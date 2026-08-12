#include "swim/trt_engine.h"

#include <NvOnnxParser.h>

#include <cstring>
#include <fstream>
#include <sys/stat.h>

namespace swim {
namespace {

class Logger : public nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) printf("[TRT] %s\n", msg);
  }
};
Logger g_logger;

/// 文件修改时间；不存在返回 0
time_t mtime_of(const std::string& p) {
  struct stat st{};
  return stat(p.c_str(), &st) == 0 ? st.st_mtime : 0;
}

std::vector<char> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  SWIM_CHECK(f.good(), "打不开文件 " + p);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
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

size_t volume(const nvinfer1::Dims& d) {
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i] < 0 ? 1 : d.d[i]);
  return v;
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
  if (fp16) cfg->setFlag(nvinfer1::BuilderFlag::kFP16);

  if (!dyn_input.empty()) {
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
  std::ofstream f(out, std::ios::binary);
  SWIM_CHECK(f.good(), "无法写入 " + out);
  f.write(static_cast<const char*>(plan->data()), plan->size());
  printf("[TRT] engine %.1f MB 已写入\n", plan->size() / 1e6);
}

}  // namespace

std::unique_ptr<TrtEngine> TrtEngine::load(
    const std::string& onnx_path, const std::string& engine_path,
    const std::string& dynamic_input, int min_b, int opt_b, int max_b, bool fp16) {
  // engine 与 GPU 型号/TRT 版本绑定，不可跨机复制；比 onnx 旧则重建
  if (mtime_of(engine_path) < mtime_of(onnx_path))
    build_engine(onnx_path, engine_path, dynamic_input, min_b, opt_b, max_b, fp16);

  auto e = std::unique_ptr<TrtEngine>(new TrtEngine());
  e->dyn_input_ = dynamic_input;
  e->runtime_.reset(nvinfer1::createInferRuntime(g_logger));
  auto blob = read_file(engine_path);
  e->engine_.reset(e->runtime_->deserializeCudaEngine(blob.data(), blob.size()));
  SWIM_CHECK(e->engine_, "engine 反序列化失败（GPU 型号或 TRT 版本不匹配？）");
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
    b.bytes = volume(b.dims) * elem_size(engine_->getTensorDataType(b.name.c_str()));
    SWIM_CUDA(cudaMalloc(&b.ptr, b.bytes));
    SWIM_CHECK(ctx_->setTensorAddress(b.name.c_str(), b.ptr),
               std::string("setTensorAddress 失败 ") + b.name);
  }
  set_batch(max_batch);   // 先按最大形状问一次工作区大小
  workspace_bytes_ = ctx_->updateDeviceMemorySizeForShapes();
  SWIM_CHECK(workspace_bytes_ > 0, "updateDeviceMemorySizeForShapes 返回 0");
  SWIM_CUDA(cudaMalloc(&workspace_, workspace_bytes_));
  ctx_->setDeviceMemoryV2(workspace_, static_cast<int64_t>(workspace_bytes_));
  printf("[TRT] %zu 输入 / %zu 输出, batch<=%d, 工作区 %.1f MB\n",
         in_idx_.size(), out_idx_.size(), max_batch, workspace_bytes_ / 1e6);
}

void TrtEngine::set_batch(int batch) {
  SWIM_CHECK(batch >= 1 && batch <= max_batch_, "batch 超出 profile 范围");
  if (dyn_input_.empty()) return;
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
    b.bytes = volume(b.dims) * elem_size(engine_->getTensorDataType(b.name.c_str()));
  }
}

void TrtEngine::enqueue(cudaStream_t stream) {
  SWIM_CHECK(ctx_->allInputDimensionsSpecified(), "输入形状未完全指定");
  SWIM_CHECK(ctx_->enqueueV3(stream), "enqueueV3 失败");
}

}  // namespace swim
