// TensorRT engine 的薄封装：ONNX -> engine（带缓存）+ 动态 shape 绑定。
//
// 环境：TensorRT 10.11.0.33（cuda-12.0~12.9 变体）。
// 注意 TRT 11 移除了 BuilderFlag::kFP16（强类型恒开），本封装只针对 10.x；
// 若日后升级到 11.x，需改为导出 fp16 ONNX 并去掉 setFlag(kFP16)。
//
// 显存策略：用 kUSER_MANAGED 自己管 device memory —— 默认策略下
// updateDeviceMemorySizeForShapes() 会报 Error Code 3 且返回 0，
// 且每次 setInputShape 可能触发重分配，与"稳定效率"目标冲突。
#pragma once

#include <NvInfer.h>

#include <memory>
#include <string>
#include <vector>

#include "swim/common.h"

namespace swim {

/// TRT 对象的 deleter（TRT 10 用 delete 而非 destroy()）
struct TrtDeleter {
  template <typename T>
  void operator()(T* p) const { delete p; }
};
template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDeleter>;

/// 一个张量的绑定信息。
struct Binding {
  std::string      name;
  nvinfer1::Dims   dims;         // 运行时形状（batch 已确定）
  size_t           bytes = 0;    // 当前形状下的字节数
  void*            ptr   = nullptr;   // device 指针（由本类分配并持有）
  bool             is_input = false;
};

class TrtEngine {
 public:
  /// engine 缓存按身份戳命名（ONNX mtime/size + TRT 版本 + SM + 精度 + batch
  /// 上限），engine_path 只作为基名：pose.engine -> pose.<tag>.engine。
  /// 缺文件才构建，因此换 GPU / 换 --max-persons / 换 --fp32 都会各存一份，
  /// 不会互相覆盖，也不会静默复用不匹配的 engine。
  static std::unique_ptr<TrtEngine> load(
      const std::string& onnx_path, const std::string& engine_path,
      const std::string& dynamic_input = "", int min_b = 1, int opt_b = 1,
      int max_b = 1, bool fp16 = true);

  ~TrtEngine();

  /// 设定动态输入的 batch，并（重新）计算各绑定的字节数。
  /// 显存按 max_batch 一次性分配，此处只调整逻辑尺寸，不重分配。
  void set_batch(int batch);

  /// 在给定 stream 上异步执行。
  void enqueue(cudaStream_t stream);

  Binding&       input(int i = 0)  { return bindings_[in_idx_.at(i)]; }
  Binding&       output(int i = 0) { return bindings_[out_idx_.at(i)]; }
  const Binding& input(int i = 0) const  { return bindings_[in_idx_.at(i)]; }
  const Binding& output(int i = 0) const { return bindings_[out_idx_.at(i)]; }
  int  num_outputs() const { return static_cast<int>(out_idx_.size()); }
  size_t workspace_bytes() const { return workspace_bytes_; }

 private:
  TrtEngine() = default;
  void init_bindings(int max_batch);

  TrtPtr<nvinfer1::IRuntime>          runtime_;
  TrtPtr<nvinfer1::ICudaEngine>       engine_;
  TrtPtr<nvinfer1::IExecutionContext> ctx_;
  std::vector<Binding>                bindings_;
  std::vector<int>                    in_idx_, out_idx_;
  std::string                         dyn_input_;
  void*                               workspace_      = nullptr;  // engine 工作区
  size_t                              workspace_bytes_ = 0;
  int                                 max_batch_      = 1;
  int                                 cur_batch_      = 0;   // 0 = 尚未设过形状
};

}  // namespace swim
