"""多路视频的帧级随机读取。

Stage2 需要在同一帧号 fi 上访问多个相机（最清晰相机 + 补检用的次清晰相机
+ 调试可视化），而 cv2.VideoCapture 只有顺序游标。这里为每路视频缓存"当前
已解码的那一帧"：

  - 同一 fi 重复取用直接命中缓存，不重复解码（4K 帧解码是 Stage2 的主要
    IO 开销，补检/调试路径原本会导致同一帧被解码两三次）
  - fi 前进时连续 read() 推进，只在需要回退时才 seek（关键帧对齐的 seek
    在长视频上代价很高，正常流程里帧号单调递增，不会触发）
"""

import cv2


class MultiVideoReader:
    """按帧号读取多路视频，每路缓存当前帧。用作上下文管理器自动释放。"""

    def __init__(self, paths):
        self.caps = []
        for p in paths:
            cap = cv2.VideoCapture(p)
            if not cap.isOpened():
                self.release()
                raise RuntimeError(f"无法打开视频: {p}")
            self.caps.append(cap)
        self._pos = [-1] * len(paths)       # 各路当前缓存帧的帧号
        self._frame = [None] * len(paths)

    def read(self, idx, frame_idx):
        """取第 idx 路视频的第 frame_idx 帧，读不到（越界/解码失败）返回 None。"""
        if self._pos[idx] == frame_idx:
            return self._frame[idx]
        cap = self.caps[idx]
        if frame_idx < self._pos[idx]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self._pos[idx] = frame_idx - 1
        while self._pos[idx] < frame_idx:
            ok, frame = cap.read()
            self._pos[idx] += 1
            if not ok:
                self._frame[idx] = None
                return None
            self._frame[idx] = frame
        return self._frame[idx]

    def release(self):
        for cap in getattr(self, "caps", []):
            cap.release()
        self.caps = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def video_meta(path):
    """返回 (width, height, fps, frame_count)。fps 读不到时按 30 兜底。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    meta = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS) or 30.0,
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return meta
