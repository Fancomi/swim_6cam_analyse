#include "swim/trt_engine.h"

#include <NvOnnxParser.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sys/stat.h>
#ifdef _WIN32
#include <process.h>                 // _getpid
#define SWIM_GETPID _getpid
#else
#include <unistd.h>                  // getpid
#define SWIM_GETPID getpid
#endif

namespace swim {
namespace {

class Logger : public nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    // 错误走 stderr 并带级别前缀：构建失败时日志常被重定向，混在 stdout 里难定位
    switch (s) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:   fprintf(stderr, "[TRT/E] %s\n", msg); break;
      case Severity::kWARNING: fprintf(stderr, "[TRT/W] %s\n", msg); break;
      default: break;                       // kINFO/kVERBOSE 太啰嗦，丢弃
    }
  }
};
Logger g_logger;

struct FileStamp {
  int64_t mtime = 0, size = 0;
  /// size > 0 是必需的：0 字节的残留 onnx/engine（写盘中断、磁盘满）若被判为
  /// "存在"，就会跳过友好报错，改在 parse/deserialize 里以晦涩的错误现身。
  bool exists() const { return size > 0 && mtime != 0; }
};

FileStamp stamp_of(const std::string& p) {
  struct stat st{};
  if (stat(p.c_str(), &st) != 0) return {};
  return {static_cast<int64_t>(st.st_mtime), static_cast<int64_t>(st.st_size)};
}

std::vector<char> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  SWIM_CHECK(f.good(), "打不开文件 " + p);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

/// engine 缓存的身份戳：engine 与「ONNX 内容 + TRT 版本 + GPU 架构 + 精度 +
/// batch 上限」全部绑定，任一变化都必须重建。只比 mtime 会静默复用错的 engine
/// （换 GPU、换 --max-persons、换 --fp32 都不会碰 onnx 的 mtime）。
/// 版本用 getInferLibVersion()（运行期实际加载的 TRT）而非编译期宏 —— engine
/// 是给运行期反序列化用的，且 10.x 头文件里并无 NV_TENSORRT_VERSION 这个宏。
std::string engine_tag(const std::string& onnx, int max_b, bool fp16) {
  const FileStamp s = stamp_of(onnx);
  cudaDeviceProp prop{};
  int dev = 0;
  SWIM_CUDA(cudaGetDevice(&dev));
  SWIM_CUDA(cudaGetDeviceProperties(&prop, dev));
  char buf[128];
  snprintf(buf, sizeof buf, "sm%d%d-trt%d-b%d-%s-%llx-%llx", prop.major, prop.minor,
           getInferLibVersion(), max_b, fp16 ? "fp16" : "fp32",
           static_cast<unsigned long long>(s.mtime),
           static_cast<unsigned long long>(s.size));
  return buf;
}

/// 把身份戳插进文件名：cpp/models/pose.engine -> cpp/models/pose.<tag>.engine
std::string tagged_path(const std::string& base, const std::string& tag) {
  const size_t dot = base.find_last_of('.');
  const size_t sep = base.find_last_of("/\\");
  return (dot == std::string::npos || (sep != std::string::npos && dot < sep))
             ? base + "." + tag
             : base.substr(0, dot) + "." + tag + base.substr(dot);
}

size_t elem_size(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF:  return 2;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kBOOL:  return 1;
    default: throw std::runtime_error("不支持的 TRT DataType");
  }
}

/// 元素数。负维（未解析的动态/符号维）必须报错——当成 1 会静默少分配显存。
size_t volume(const nvinfer1::Dims& d, const std::string& who) {
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) {
    SWIM_CHECK(d.d[i] >= 0, who + " 存在未确定维度，无法计算显存大小");
    v *= static_cast<size_t>(d.d[i]);
  }
  return v;
}

/// 一个绑定在当前形状下的字节数。dtype 要从 engine 现查（不是从 Binding 缓存），
/// 分配与 set_batch 后的重算都走这里，两处口径必然一致。
size_t binding_bytes(const nvinfer1::ICudaEngine& engine, const Binding& b) {
  return volume(b.dims, b.name) *
         elem_size(engine.getTensorDataType(b.name.c_str()));
}

/// 构建 engine 并写盘。分离成函数是为了让 builder 相关对象尽早释放
/// （builder 会占约 2 GB CPU 内存）。
void build_engine(const std::string& onnx, const std::string& out,
                  const std::string& dyn_input, int min_b, int opt_b, int max_b,
                  bool fp16) {
  printf("[TRT] 构建 engine: %s -> %s (fp16=%d)\n", onnx.c_str(), out.c_str(), fp16);
  TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(g_logger));
  SWIM_CHECK(builder, "createInferBuilder 失败");
  TrtPtr<nvinfer1::INetworkDefinition> net(builder->createNetworkV2(0));
  TrtPtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*net, g_logger));

  auto buf = read_file(onnx);
  SWIM_CHECK(parser->parse(buf.data(), buf.size()), "ONNX 解析失败 " + onnx);

  TrtPtr<nvinfer1::IBuilderConfig> cfg(builder->createBuilderConfig());
  if (fp16) {
    cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
  } else {
    // kFP16 只放开"算子可用 fp16"，权重精度由 ONNX 决定。export_onnx.py 默认
    // 导 fp16 权重，此时 --fp32 得到的是"fp16 权重跑 fp32 算子"，拿它做精度
    // 对照会得出错误结论 —— 提示改用 export_onnx.py --fp32 重导。
    for (int i = 0; i < net->getNbInputs(); ++i)
      if (net->getInput(i)->getType() == nvinfer1::DataType::kHALF) {
        printf("[TRT] 提示：--fp32 只影响算子精度，%s 的输入 %s 仍是 fp16 权重。"
               "要真正的 fp32 请先跑 export_onnx.py --fp32\n",
               onnx.c_str(), net->getInput(i)->getName());
        break;
      }
  }

  if (!dyn_input.empty()) {
    // profile 非法（如 opt > max）时 buildSerializedNetwork 只会返回空指针，
    // 报"engine 构建失败"看不出因果，在这里先挡住
    SWIM_CHECK(1 <= min_b && min_b <= opt_b && opt_b <= max_b,
               "动态 batch profile 非法：须 1 <= min <= opt <= max，收到 " +
                   std::to_string(min_b) + "/" + std::to_string(opt_b) + "/" +
                   std::to_string(max_b));
    auto* prof = builder->createOptimizationProfile();
    for (int i = 0; i < net->getNbInputs(); ++i) {
      auto* t = net->getInput(i);
      if (dyn_input != t->getName()) continue;
      auto mn = t->getDimensions(), op = mn, mx = mn;
      mn.d[0] = min_b; op.d[0] = opt_b; mx.d[0] = max_b;
      prof->setDimensions(t->getName(), nvinfer1::OptProfileSelector::kMIN, mn);
      prof->setDimensions(t->getName(), nvinfer1::OptProfileSelector::kOPT, op);
      prof->setDimensions(t->getName(), nvinfer1::OptProfileSelector::kMAX, mx);
    }
    cfg->addOptimizationProfile(prof);
  }

  TrtPtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*net, *cfg));
  SWIM_CHECK(plan, "engine 构建失败");
  // 先写临时文件再 rename：中途崩溃/断电不会留下半截 engine 被下次误当缓存。
  // tmp 名带 pid：两个进程同时首次构建同一 engine 时不会互相写坏对方的临时文件
  // （rename 到同一目标是原子的，谁最后到谁生效，内容一致所以无所谓）。
  const std::string tmp = out + "." + std::to_string(SWIM_GETPID()) + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary);
    SWIM_CHECK(f.good(), "无法写入 " + tmp);
    f.write(static_cast<const char*>(plan->data()), plan->size());
    SWIM_CHECK(f.good(), "写入未完成 " + tmp + "（磁盘满？）");
  }
  std::remove(out.c_str());                   // Windows 的 rename 不覆盖已存在文件
  SWIM_CHECK(std::rename(tmp.c_str(), out.c_str()) == 0, "重命名失败 " + tmp);
  printf("[TRT] engine %.1f MB 已写入 %s\n", plan->size() / 1e6, out.c_str());
}

}  // namespace

std::unique_ptr<TrtEngine> TrtEngine::load(
    const std::string& onnx_path, const std::string& engine_path,
    const std::string& dynamic_input, int min_b, int opt_b, int max_b, bool fp16) {
  // 两种来源。开发机：有 ONNX，engine 按身份戳命名并按需构建。
  // 交付包（dist）：只带一个现成的 engine，没有 ONNX 也没有 TRT 的构建资源
  // （nvinfer_builder_resource 有 1.8 GB，不值得随包分发）。
  const bool have_onnx = stamp_of(onnx_path).exists();
  const std::string path =
      have_onnx ? tagged_path(engine_path, engine_tag(onnx_path, max_b, fp16))
                : engine_path;
  if (have_onnx) {
    if (!stamp_of(path).exists())
      build_engine(onnx_path, path, dynamic_input, min_b, opt_b, max_b, fp16);
  } else {
    SWIM_CHECK(stamp_of(path).exists(),
               "既找不到 ONNX " + onnx_path + " 也找不到预构建 engine " + path +
                   "（开发机请跑 cpp/tools/export_onnx.py；交付包应自带 engine）");
    printf("[TRT] 用预构建 engine %s（无 ONNX，本机不重建）\n", path.c_str());
  }

  auto e = std::unique_ptr<TrtEngine>(new TrtEngine());
  e->dyn_input_ = dynamic_input;
  e->runtime_.reset(nvinfer1::createInferRuntime(g_logger));
  SWIM_CHECK(e->runtime_, "createInferRuntime 失败");
  // 读盘 + 反序列化。写成 lambda 是因为要做两次（第二次是删档重建后重试），
  // 而 blob 得在每次调用结束就释放（engine 有几百 MB，不该多留一份在内存里）。
  auto deserialize = [&e, &path] {
    auto blob = read_file(path);
    e->engine_.reset(e->runtime_->deserializeCudaEngine(blob.data(), blob.size()));
  };
  deserialize();
  if (!e->engine_) {
    // 有 ONNX：身份戳一致却反序列化失败 = 文件损坏，删掉重建一次，再失败才抛。
    // 没 ONNX：engine 与本机不匹配（换了 GPU 架构或 TRT 版本），无从重建。
    SWIM_CHECK(have_onnx,
               "engine 反序列化失败 " + path +
                   "：它与本机的 GPU 架构或 TensorRT 版本不匹配。"
                   "交付包的 engine 与构建它的机器绑定，请在本机重新构建"
                   "（scripts/build.bat）后再打包");
    printf("[TRT] %s 反序列化失败，删除并重建\n", path.c_str());
    std::remove(path.c_str());
    build_engine(onnx_path, path, dynamic_input, min_b, opt_b, max_b, fp16);
    deserialize();
    SWIM_CHECK(e->engine_, "engine 反序列化失败 " + path);
  }
  // 预构建 engine 的 batch 上限由打包时决定，可能小于本次 --max-persons。
  // 不查的话会在 init_bindings 的 setInputShape 处以晦涩的失败现身。
  if (!have_onnx && !dynamic_input.empty()) {
    const auto mx = e->engine_->getProfileShape(
        dynamic_input.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
    SWIM_CHECK(mx.nbDims > 0 && mx.d[0] >= max_b,
               "预构建 engine 的 batch 上限是 " + std::to_string(mx.d[0]) +
                   "，小于 --max-persons " + std::to_string(max_b));
  }
  // kUSER_MANAGED：显存自管，避免运行时重分配
  e->ctx_.reset(e->engine_->createExecutionContext(
      nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
  SWIM_CHECK(e->ctx_, "createExecutionContext 失败");
  e->init_bindings(max_b);
  return e;
}

TrtEngine::~TrtEngine() {
  for (auto& b : bindings_)
    if (b.ptr) cudaFree(b.ptr);
  if (workspace_) cudaFree(workspace_);
}

void TrtEngine::init_bindings(int max_batch) {
  max_batch_ = max_batch;
  const int n = engine_->getNbIOTensors();
  bindings_.resize(n);
  for (int i = 0; i < n; ++i) {
    auto& b = bindings_[i];
    b.name = engine_->getIOTensorName(i);
    b.is_input = engine_->getTensorIOMode(b.name.c_str()) ==
                 nvinfer1::TensorIOMode::kINPUT;
    (b.is_input ? in_idx_ : out_idx_).push_back(i);
    // 按 max_batch 分配：动态维填 max_batch，后续 set_batch 只改逻辑尺寸
    b.dims = engine_->getTensorShape(b.name.c_str());
    if (b.dims.d[0] < 0) b.dims.d[0] = max_batch;
    b.bytes = binding_bytes(*engine_, b);
    SWIM_CHECK(cudaMalloc(&b.ptr, b.bytes) == cudaSuccess,
               "绑定 " + b.name + " 分配 " + std::to_string(b.bytes / (1 << 20)) +
                   " MB 显存失败（显存不足？可降 --max-persons）");
    SWIM_CHECK(ctx_->setTensorAddress(b.name.c_str(), b.ptr),
               std::string("setTensorAddress 失败 ") + b.name);
  }
  set_batch(max_batch);   // 先按最大形状问一次工作区大小
  workspace_bytes_ = ctx_->updateDeviceMemorySizeForShapes();
  SWIM_CHECK(workspace_bytes_ > 0, "updateDeviceMemorySizeForShapes 返回 0");
  SWIM_CHECK(cudaMalloc(&workspace_, workspace_bytes_) == cudaSuccess,
             "engine 工作区 " + std::to_string(workspace_bytes_ / (1 << 20)) +
                 " MB 分配失败（显存不足？）");
  ctx_->setDeviceMemoryV2(workspace_, static_cast<int64_t>(workspace_bytes_));
  printf("[TRT] %zu 输入 / %zu 输出, batch<=%d, 工作区 %.1f MB\n",
         in_idx_.size(), out_idx_.size(), max_batch, workspace_bytes_ / 1e6);
}

void TrtEngine::set_batch(int batch) {
  SWIM_CHECK(batch >= 1 && batch <= max_batch_, "batch 超出 profile 范围");
  if (dyn_input_.empty() || batch == cur_batch_) return;   // 形状未变则无需重设
  cur_batch_ = batch;
  for (auto& b : bindings_) {
    if (b.name != dyn_input_) continue;
    auto d = b.dims;
    d.d[0] = batch;
    SWIM_CHECK(ctx_->setInputShape(b.name.c_str(), d),
               "setInputShape 失败 " + b.name);
  }
  // 输入形状变化会传播到输出，同步各绑定的逻辑字节数（显存不动）
  for (auto& b : bindings_) {
    b.dims = ctx_->getTensorShape(b.name.c_str());
    b.bytes = binding_bytes(*engine_, b);
  }
  // 工作区只按 max_batch 分配一次。实测激活量随 batch 单调（max 是上界），但
  // 那是观察而非 API 契约 —— 加多 profile 或 weight streaming 后可能不成立，
  // 故每次改形状后核一次。workspace_bytes_==0 表示还在 init_bindings 里，跳过。
  if (workspace_bytes_) {
    const size_t need = ctx_->updateDeviceMemorySizeForShapes();
    SWIM_CHECK(need <= workspace_bytes_,
               "batch " + std::to_string(batch) + " 需要工作区 " +
                   std::to_string(need >> 20) + " MB，超过按最大 batch 预留的 " +
                   std::to_string(workspace_bytes_ >> 20) + " MB");
  }
}

void TrtEngine::enqueue(cudaStream_t stream) {
  SWIM_CHECK(ctx_->allInputDimensionsSpecified(), "输入形状未完全指定");
  SWIM_CHECK(ctx_->enqueueV3(stream), "enqueueV3 失败");
}

}  // namespace swim
