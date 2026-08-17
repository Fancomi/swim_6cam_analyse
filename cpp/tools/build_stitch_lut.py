"""configs/pool_mesh.json -> cpp/models/stitch.lut：把拼接烘成逐像素查找表。

为什么烘表而不在 C++ 里光栅化三角形：拼接的几何全静态（标定固定），而"哪个画布
像素属于哪个三角形"的判定语义来自 OpenCV 的 fillConvexPoly + getAffineTransform。
在 CUDA 里重写这套判定是整条链路唯一容易静默出错的地方（边界像素差一个、
仿射差半像素，表现为接缝错位而不是报错）。这里直接调那两个函数，于是 C++ 侧
零光栅化代码、零漂移风险，运行期只做一次 gather。

产物是生成物，与 ONNX/engine 同级（不入库）。上游改了标定就重跑本脚本。

用法：
    python cpp/tools/build_stitch_lut.py                    # 默认 configs/ -> cpp/models/
    python cpp/tools/build_stitch_lut.py --ppm 100 --force
"""
import argparse
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

MAGIC = b"SWSTLUT1"
# header: magic + 8 个 u32 + crc32 + 预留
HEADER = struct.Struct("<8s8I I 28s")
CAMERA = struct.Struct("<16s4I2Q")

# pool 线的相机身份：按 pool_mesh.json 的 meshes 顺序。标定时定死，不要重排
# （CLAUDE.md「不要改 configs/pool_mesh.json 的 meshes 顺序」）。
POOL_CAMERA_IDS = ("cam3", "cam2", "cam1", "cam4", "cam5", "cam6")
SOURCE_SIZE = (3840, 2160)      # 六路原相机分辨率
POOL_PPM = 100.0                # 画布每米像素数，与 --ppm 默认值一致
POOL_NEG_V = True               # pool 的 bake 存 Y 向下，需翻转


def load_meshes(path, neg_v, neg_u):
    """mesh JSON -> 世界坐标三角形列表（原地按 neg_u/neg_v 镜像）。"""
    meshes = json.loads(Path(path).read_text(encoding="utf-8"))["meshes"]
    for mesh in meshes:
        for triangle in mesh["triangles"]:
            for vertex in triangle:
                if neg_u:
                    vertex["pos"][0] = -vertex["pos"][0]
                if neg_v:
                    vertex["pos"][1] = -vertex["pos"][1]
    return meshes


class Canvas:
    """画布几何：世界米 -> 画布像素（原点左上，y 向下）。"""

    def __init__(self, meshes, ppm):
        xs = [v["pos"][0] for m in meshes for t in m["triangles"] for v in t]
        ys = [v["pos"][1] for m in meshes for t in m["triangles"] for v in t]
        self.xmin, self.ymin, self.ppm = min(xs), min(ys), ppm
        self.width = int(round((max(xs) - self.xmin) * ppm)) + 1
        self.height = int(round((max(ys) - self.ymin) * ppm)) + 1

    def project(self, triangle):
        return np.array([[(v["pos"][0] - self.xmin) * self.ppm,
                          self.height - 1 - (v["pos"][1] - self.ymin) * self.ppm]
                         for v in triangle], np.float32)


def remap_one(mesh, canvas, src_size):
    """逐三角形烘出该相机的 (画布像素 -> 源像素) 逆映射与覆盖掩码。

    与离线参考实现同函数同顺序：getAffineTransform 求仿射、fillConvexPoly 定覆盖、
    先与画布求交再裁掩码（负起点切片会从对侧边缘取像素，既不报错也画错地方）。
    """
    height, width = canvas.height, canvas.width
    mapx = np.zeros((height, width), np.float32)
    mapy = np.zeros((height, width), np.float32)
    mask = np.zeros((height, width), np.uint8)
    src_w, src_h = src_size
    for triangle in mesh["triangles"]:
        dst = canvas.project(triangle)
        src = np.array([[v["uv"][0] * src_w, (1.0 - v["uv"][1]) * src_h]
                        for v in triangle], np.float32)
        x, y, w, h = cv2.boundingRect(dst)
        if w <= 0 or h <= 0:
            continue
        local = (dst - np.float32([x, y])).astype(np.float32)
        affine = cv2.getAffineTransform(local, src)
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        grid_x = grid_x.astype(np.float32)
        grid_y = grid_y.astype(np.float32)
        sx = affine[0, 0] * grid_x + affine[0, 1] * grid_y + affine[0, 2]
        sy = affine[1, 0] * grid_x + affine[1, 1] * grid_y + affine[1, 2]
        filled = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(filled, np.int32(local), 1)
        inside = filled > 0
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, width), min(y + h, height)
        if x1 <= x0 or y1 <= y0:
            continue
        window = (slice(y0, y1), slice(x0, x1))
        local_window = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
        inside = inside[local_window]
        mapx[window][inside] = sx[local_window][inside]
        mapy[window][inside] = sy[local_window][inside]
        mask[window][inside] = 1
    return mapx, mapy, mask


def feather_weights(masks):
    """按「到自身边界的距离」融合，逐像素归一化到和为 1。

    单覆盖处恒为 1（边缘不变暗），重叠处平滑过渡，不相交处（泳池中线）自动硬切。
    pool 的两排网格斜向大面积重叠，没有唯一的缝方向可选，所以用距离羽化而不是竖缝。
    """
    distances = []
    for mask in masks:
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        distances.append(distance * (mask > 0))
    total = np.maximum(sum(distances), 1e-6)
    return [(d / total).astype(np.float32) for d in distances]


def crop_nonzero(weight):
    """非零权重的紧包围盒。一路只覆盖画布的一部分，存全画布会白占 6 倍空间。"""
    rows, columns = np.nonzero(weight > 0.0)
    if not len(columns):
        return 0, 0, 0, 0
    return (int(columns.min()), int(rows.min()),
            int(columns.max()) + 1 - int(columns.min()),
            int(rows.max()) + 1 - int(rows.min()))


def build(mesh_json, out_path, ppm, neg_v, neg_u, camera_ids, src_size):
    """烘表并落盘；返回摘要 dict。"""
    source_bytes = Path(mesh_json).read_bytes()
    meshes = load_meshes(mesh_json, neg_v, neg_u)
    if len(camera_ids) != len(meshes):
        raise SystemExit(f"相机数不匹配：{len(camera_ids)} 个 id 对 {len(meshes)} 个 mesh")
    canvas = Canvas(meshes, ppm)
    print(f"[lut] 画布 {canvas.width}x{canvas.height} @ {ppm:g}px/m，{len(meshes)} 路")

    layers = [remap_one(mesh, canvas, src_size) for mesh in meshes]
    weights = feather_weights([layer[2] for layer in layers])

    # 画布向上取偶：H.264 的 yuv420p 要求偶数边长，奇数画布让 libx264 直接开不了
    # （--out 会在第 3 帧短写）。多出的一列/一行没有任何 lane 覆盖，kernel 那里
    # 权重和为 0 自然写黑，与上游 encoded_width/height 的约定一致。
    out_w = canvas.width + (canvas.width & 1)
    out_h = canvas.height + (canvas.height & 1)
    if (out_w, out_h) != (canvas.width, canvas.height):
        print(f"[lut] 输出补到偶数 {out_w}x{out_h}（H.264 要求），补边写黑")

    records, blobs = [], []
    # blob 偏移从 header + 相机表之后开始；每路两块（坐标、权重）连续排布
    offset = HEADER.size + len(meshes) * CAMERA.size
    total_px, covered = 0, np.zeros((canvas.height, canvas.width), np.uint8)
    for camera_id, (mapx, mapy, mask), weight in zip(camera_ids, layers, weights):
        weight = weight * (mask > 0)          # 掩码外一律 0，运行期不必再判
        x, y, w, h = crop_nonzero(weight)
        if w == 0:
            raise SystemExit(f"{camera_id} 没有任何覆盖像素，检查 mesh 与 ppm")
        sub = (slice(y, y + h), slice(x, x + w))
        # 坐标存 f32 pair：源图 3840x2160 用 u16 定点只剩 4 位小数（步长 1/16 px），
        # 而这块表一次性 132 MB 已经够小，没必要拿精度换空间
        coords = np.stack([mapx[sub], mapy[sub]], -1).astype("<f4")
        # 权重存 u16 定点（65535 = 1.0）：量化误差 1.5e-5，远小于双线性本身的误差
        wq = np.rint(np.clip(weight[sub], 0.0, 1.0) * 65535.0).astype("<u2")
        # 覆盖用权重是否为 0 表达：0 权重的像素对累加无贡献，kernel 可以无分支
        coord_bytes, weight_bytes = coords.tobytes(), wq.tobytes()
        records.append(CAMERA.pack(
            camera_id.encode("utf-8").ljust(16, b"\0"),
            x, y, w, h, offset, offset + len(coord_bytes)))
        blobs.extend((coord_bytes, weight_bytes))
        offset += len(coord_bytes) + len(weight_bytes)
        total_px += int((weight[sub] > 0).sum())
        covered |= mask
        print(f"[lut] {camera_id:6s} bbox {w}x{h}@({x},{y})  覆盖 "
              f"{int((weight[sub] > 0).sum())} px  {(len(coord_bytes) + len(weight_bytes)) / 1e6:.1f} MB")

    body = b"".join(records + blobs)
    header = HEADER.pack(MAGIC, 1, HEADER.size, out_w, out_h,
                         src_size[0], src_size[1], len(meshes), CAMERA.size,
                         zlib.crc32(body), b"\0" * 28)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(header + body)

    hole = int((covered == 0).sum())
    print(f"[lut] {out_path}  {(len(header) + len(body)) / 1e6:.1f} MB  "
          f"记录 {total_px}（{total_px / (canvas.width * canvas.height):.3f} 条/像素）")
    print(f"[lut] 未覆盖像素 {hole}（{hole / (canvas.width * canvas.height) * 100:.4f}%）"
          f"  mesh sha256 {hashlib.sha256(source_bytes).hexdigest()[:12]}")
    return {"width": out_w, "height": out_h, "cameras": len(meshes),
            "records": total_px, "holes": hole}


def main(argv=None):
    p = argparse.ArgumentParser(
        description="把泳池 mesh 标定烘成 CUDA 拼接用的逐像素查找表")
    p.add_argument("--mesh", type=Path, default=ROOT / "configs" / "pool_mesh.json")
    p.add_argument("--out", type=Path, default=ROOT / "cpp" / "models" / "stitch.lut")
    p.add_argument("--ppm", type=float, default=POOL_PPM,
                   help=f"画布每米像素数 (默认 {POOL_PPM:g}，须与 --ppm 一致)")
    p.add_argument("--cameras", default=",".join(POOL_CAMERA_IDS),
                   help="按 mesh 顺序的相机 id，逗号分隔（标定时定死，勿重排）")
    p.add_argument("--force", action="store_true", help="已存在也重烘")
    a = p.parse_args(argv)

    if a.out.exists() and not a.force:
        if a.out.stat().st_mtime >= a.mesh.stat().st_mtime:
            print(f"[lut] {a.out} 已是最新（--force 可强制重烘）")
            return
    build(a.mesh, a.out, a.ppm, POOL_NEG_V, False,
          tuple(c.strip() for c in a.cameras.split(",")), SOURCE_SIZE)


if __name__ == "__main__":
    sys.exit(main())
