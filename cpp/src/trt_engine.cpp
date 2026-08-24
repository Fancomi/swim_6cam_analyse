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
 public:
  /// 进静音期：错误只存不打，退出静音时留着给调用方取。用于「试着反序列化交付包
  /// 自带的 engine」—— 换代显卡时它必然失败并让 TRT 打一条 Error Code 6，而那是
  /// 预期路径（下一步就用 ONNX 现烘），直接刷 [TRT/E] 会让现场以为程序崩了。
  /// 存下来的原因会并进我们自己的说明里，所以信息不丢反而更全。
  void mute(bool on) {
    std::lock_guard<std::mutex> lk(mu_);
    muted_ = on;
    if (on) first_.clear();
  }
  /// 取静音期里的第一条错误（后续多是它的连锁反应）。
  std::string take() {
    std::lock_guard<std::mutex> lk(mu_);
    std::string s = std::move(first_);
    first_.clear();
    return s;
  }

 private:
  void log(Severity s, const char* msg) noexcept override {
    if (s > Severity::kWARNING) return;     // kINFO/kVERBOSE 太啰嗦，丢弃
    std::lock_guard<std::mutex> lk(mu_);
    if (muted_ && s <= Severity::kERROR) {
      if (first_.empty()) first_ = msg;
      return;
    }
    // 错误走 stderr 并带级别前缀：构建失败时日志常被重定向，混在 stdout 里难定位
    fprintf(stderr, s <= Severity::kERROR ? "[TRT/E] %s\n" : "[TRT/W] %s\n", msg);
  }

  std::mutex  mu_;
  bool        muted_ = false;
  std::string first_;
};
Logger g_logger;

/// engine 构建进度。TRT 承诺回调可能来自内部多个线程，所以整体加锁；阶段是
/// 嵌套的树，只跟最内层那个（它才带步数），一行原地刷新 —— 构建有上百个阶段，
/// 逐行打会把启动日志刷没。回答的是现场最常问的那句「到底转完了没」。
class BuildProgress : public nvinfer1::IProgressMonitor {
 public:
  explicit BuildProgress(double t0) : t0_(t0) {}

  void phaseStart(const char* name, const char*, int32_t steps) noexcept override {
    std::lock_guard<std::mutex> lk(mu_);
    cur_ = name;
    steps_ = steps > 0 ? steps : 1;         // 文档说必为正，除零的代价太大不赌
    step_ = 0;
    draw();
  }
  bool stepComplete(const char* name, int32_t step) noexcept override {
    std::lock_guard<std::mutex> lk(mu_);
    if (cur_ == name) {
      step_ = step + 1;
      draw();
    }
    return true;                            // 返回 false 会让 TRT 中止构建
  }
  void phaseFinish(const char* name) noexcept override {
    std::lock_guard<std::mutex> lk(mu_);
    if (cur_ == name) cur_.clear();
  }

 private:
  /// 定宽 + 行尾补空格：原地刷新时短名字才不会拖着上一行的残字。
  /// 限频到 10 Hz：实测一次构建有 240 万次回调（阶段是嵌套树，每个 tactic 都
  /// 进出一次 phaseStart），逐次 printf+fflush 会把构建本身拖慢，而原地刷新的
  /// 进度条给人看，10 Hz 已经足够顺滑。
  void draw() {
    const double t = now_ms();
    if (t - drawn_ms_ < 100.0) return;
    drawn_ms_ = t;
    printf("\r[TRT] 构建中 %-32.32s %3d%%  已用 %4.0f s   ", cur_.c_str(),
           int(100.0 * step_ / steps_), (t - t0_) / 1000.0);
    fflush(stdout);
  }

  std::mutex  mu_;
  std::string cur_;
  double      t0_;
  double      drawn_ms_ = 0;
  int         steps_ = 1, step_ = 0;
};

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
  const double t0 = now_ms();
  // 首次在新架构的机器上要几分钟，且过程中没有任何 stdout —— 现场只能看着黑屏
  // 猜是不是卡死。所以先说清「要等」，再由 IProgressMonitor 原地刷进度。
  printf("[TRT] 构建 engine: %s -> %s (fp16=%d)\n", onnx.c_str(), out.c_str(), fp16);
  printf("[TRT] 首次在本机构建要几分钟（之后启动秒开）。构建期间不会出画面。\n");
  fflush(stdout);
  BuildProgress progress(t0);
  TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(g_logger));
  SWIM_CHECK(builder, "createInferBuilder 失败");
  TrtPtr<nvinfer1::INetworkDefinition> net(builder->createNetworkV2(0));
  TrtPtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*net, g_logger));

  auto buf = read_file(onnx);
  SWIM_CHECK(parser->parse(buf.data(), buf.size()), "ONNX 解析失败 " + onnx);

  TrtPtr<nvinfer1::IBuilderConfig> cfg(builder->createBuilderConfig());
  cfg->setProgressMonitor(&progress);       // 生命期覆盖 buildSerializedNetwork
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
  // 进度是原地刷新的，这里换行收尾，并给出总耗时 —— 「转了多久」是现场唯一能
  // 用来估下次开机等多久的数。
  printf("\n[TRT] engine %.1f MB 已写入 %s（耗时 %.0f s）\n", plan->size() / 1e6,
         out.c_str(), (now_ms() - t0) / 1000.0);
  fflush(stdout);
}

}  // namespace

std::unique_ptr<TrtEngine> TrtEngine::load(
    const std::string& onnx_path, const std::string& engine_path,
    const std::string& dynamic_input, int min_b, int opt_b, int max_b, bool fp16) {
  // engine 有三个可能来源，优先级从高到低：
  //   身份戳名  开发机自己烘过的，与「本机 + 本 ONNX + 本参数」严格对应，直接可用
  //   裸名      交付包自带的预烘 engine。名字不含身份戳，无从判断是否属于本机，
  //             只能试着反序列化；不匹配时有 ONNX 就现烘（全能包），没有就报错（精简包）
  //   现烘      要有 ONNX
  const bool have_onnx = stamp_of(onnx_path).exists();
  const std::string tagged =
      have_onnx ? tagged_path(engine_path, engine_tag(onnx_path, max_b, fp16))
                : engine_path;
  std::string path = tagged;
  if (!stamp_of(tagged).exists()) {
    if (stamp_of(engine_path).exists()) {
      path = engine_path;
      printf("[TRT] 用交付包自带的 engine %s%s\n", path.c_str(),
             have_onnx ? "（不匹配则用 ONNX 现烘）" : "（无 ONNX，本机不重建）");
    } else {
      SWIM_CHECK(have_onnx,
                 "既找不到 ONNX " + onnx_path + " 也找不到预构建 engine " +
                     engine_path + "（开发机请跑 cpp/tools/export_onnx.py；"
                     "交付包应自带 engine）");
      build_engine(onnx_path, tagged, dynamic_input, min_b, opt_b, max_b, fp16);
    }
  }

  auto e = std::unique_ptr<TrtEngine>(new TrtEngine());
  e->dyn_input_ = dynamic_input;
  e->runtime_.reset(nvinfer1::createInferRuntime(g_logger));
  SWIM_CHECK(e->runtime_, "createInferRuntime 失败");
  // 读盘 + 反序列化 + 体检。返回空串表示可用，否则是「为什么不可用」的短语。
  // 写成 lambda 是因为要做两次（第二次是现烘后重试），而 blob 得在每次调用结束就
  // 释放（engine 有几百 MB，不该多留一份在内存里）。
  auto try_load = [&e, &dynamic_input, max_b](const std::string& p) -> std::string {
    auto blob = read_file(p);
    // 反序列化失败在这条路上是**预期**分支（换代显卡时交付包自带的 engine 必然
    // 不匹配），TRT 会打一条 Error Code 6。静音起来，把原因并进我们的说明里，
    // 免得现场把「正在按计划现烘」看成崩溃。
    g_logger.mute(true);
    e->engine_.reset(e->runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    const std::string trt_err = g_logger.take();
    g_logger.mute(false);
    if (!e->engine_)
      return "反序列化失败（GPU 架构或 TensorRT 版本不符，也可能文件损坏）" +
             (trt_err.empty() ? std::string() : "：" + trt_err);
    // 预构建 engine 的 batch 上限由打包时决定，可能小于本次 --max-persons。
    // 不查的话会在 init_bindings 的 setInputShape 处以晦涩的失败现身。
    if (!dynamic_input.empty()) {
      const auto mx = e->engine_->getProfileShape(
          dynamic_input.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
      if (!(mx.nbDims > 0 && mx.d[0] >= max_b))
        return "batch 上限 " + std::to_string(mx.d[0]) + " 小于 --max-persons " +
               std::to_string(max_b);
    }
    return {};
  };
  std::string why = try_load(path);
  if (!why.empty()) {
    SWIM_CHECK(have_onnx,
               "预构建 engine " + path + " 不可用：" + why +
                   "。交付包的 engine 与打包机的 GPU 架构 + TensorRT 版本绑定，"
                   "请在本机重新构建（scripts/build.bat）后再打包");
    printf("[TRT] %s 不可用（%s），改用 ONNX 现烘\n", path.c_str(), why.c_str());
    // 自己烘的坏了才删；交付包自带的裸名 engine 留着（换回原机器还能用）
    if (path == tagged) std::remove(path.c_str());
    build_engine(onnx_path, tagged, dynamic_input, min_b, opt_b, max_b, fp16);
    why = try_load(tagged);
    SWIM_CHECK(why.empty(), "新烘的 engine 仍不可用 " + tagged + "：" + why);
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
