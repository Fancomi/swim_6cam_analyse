// 子进程 + 一根管道：本项目所有外部进程（ffmpeg 解码、ffprobe 探流、ffmpeg
// 编码）都是「起一个进程，单向接一根管道」这一种形态，故只需这一个类。
//
// Windows 上不用 _popen 走像素通道：它给的匿名管道缓冲只有几 KB，ffmpeg 每写满
// 就得等我们来取，解码与搬运串成一串。实测同一条解码命令 _popen 约 16.7 ms/帧，
// 自己 CreatePipe 给 128 MB 缓冲后 13.0 ms/帧 —— 已贴住 ffmpeg CLI 本身的地板：
// `-f rawvideo -pix_fmt bgr24` 落 NUL 实测 10~13.5 ms/帧，取决于文件是否在系统
// 页缓存里（冷读要算上磁盘 IO，热读只剩解码+swscale）。
#pragma once

#include <cstdio>
#include <string>

namespace swim {

/// 子进程 + 一根管道：读它的 stdout（Mode::Read）或写它的 stdin（Mode::Write）。
///
/// 两个方向都是二进制模式：Windows 的文本模式会改写 0x0D/0x0A，读像素会毁数据、
/// 写像素会把 0x0A 撑成 CRLF。文本输出（ffprobe）用二进制读同样安全 ——
/// 唯一差别是行尾多个 '\r'，调用方本来就在剥它。
class Proc {
 public:
  enum class Mode { Read, Write };

  /// buf 是 Windows 管道容量提示（内核的容量上限，不预提交物理内存 —— 实际
  /// 占用取决于「已写入尚未被读走」的字节数，稳态下就是几个 chunk）；
  /// Linux 走 popen，该值被忽略。失败时 file()==nullptr。
  Proc(const std::string& cmd, Mode mode, unsigned buf_bytes = 1u << 20);
  ~Proc();                       // 幂等收尸，不抛

  Proc(const Proc&) = delete;
  Proc& operator=(const Proc&) = delete;

  FILE* file() const { return f_; }
  explicit operator bool() const { return f_ != nullptr; }

  /// 关管道并等子进程结束，返回其退出码；幂等，已收过返回 0。
  int close();

 private:
  FILE* f_    = nullptr;
  Mode  mode_ = Mode::Read;
#ifdef _WIN32
  /// 子进程句柄；非空表示还没收尸。用 void* 存是为了不在头文件里拉 windows.h
  /// （HANDLE 本身就是 void*）。
  void* proc_ = nullptr;
#endif
};

}  // namespace swim
