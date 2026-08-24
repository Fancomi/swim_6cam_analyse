#include "swim/proc.h"

#ifdef _WIN32
// NOMINMAX：windows.h 的 min/max 宏会打断 std:: 与 OpenCV 的模板调用
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/wait.h>
#endif

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace swim {

#ifdef _WIN32
namespace {

/// 等子进程结束并关句柄，返回它的退出码。
/// wait_ms 内没走完就直接终止（此时返回 0：那是我们造成的，报出去会让调用方
/// 以为编解码失败）。wait_ms=INFINITE 表示无限期等。
int reap(HANDLE h, DWORD wait_ms) {
  DWORD code = 0;
  if (WaitForSingleObject(h, wait_ms) != WAIT_OBJECT_0) TerminateProcess(h, 1);
  else if (!GetExitCodeProcess(h, &code)) code = 0;
  CloseHandle(h);
  return int(code);
}

/// 读到 EOF 时子进程已在退出，给它这么久自己走完；仍在写就终止。
constexpr DWORD kReadDrainMs = 100;

}  // namespace
#endif

Proc::Proc(const std::string& cmd, Mode mode, unsigned buf_bytes) : mode_(mode) {
#ifdef _WIN32
  const bool rd_mode = mode == Mode::Read;
  SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};   // 给子进程那端要可继承
  HANDLE rd = nullptr, wr = nullptr;
  if (!CreatePipe(&rd, &wr, &sa, DWORD(buf_bytes))) return;
  // 父进程自己留的那端不给子进程：读模式留读端，写模式留写端。
  // 漏掉这步时子进程也持有父端句柄，读模式就永远等不到 EOF。
  HANDLE mine  = rd_mode ? rd : wr;
  HANDLE their = rd_mode ? wr : rd;
  SetHandleInformation(mine, HANDLE_FLAG_INHERIT, 0);

  STARTUPINFOA si{};
  si.cb = sizeof si;
  si.dwFlags = STARTF_USESTDHANDLES;
  // 只替换用到的那一端，另外两个标准句柄照旧继承父进程 ——
  // 尤其 hStdError 必须保留，否则 ffmpeg 的报错就没地方去了。
  si.hStdInput  = rd_mode ? GetStdHandle(STD_INPUT_HANDLE)  : their;
  si.hStdOutput = rd_mode ? their : GetStdHandle(STD_OUTPUT_HANDLE);
  si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
  PROCESS_INFORMATION pi{};
  std::vector<char> line(cmd.begin(), cmd.end());     // CreateProcessA 要可写缓冲
  line.push_back('\0');
  if (!CreateProcessA(nullptr, line.data(), nullptr, nullptr, TRUE, 0, nullptr,
                      nullptr, &si, &pi)) {
    CloseHandle(rd);
    CloseHandle(wr);
    return;                                          // f_ 保持 nullptr
  }
  CloseHandle(their);                                 // 只留子进程那份
  CloseHandle(pi.hThread);

  // 把 HANDLE 包成 FILE*，让上层的 fread/fwrite 与 Linux 分支共用一份代码。
  // fd/FILE* 接管 mine 的所有权，之后只能由 fclose 释放（不能再 CloseHandle）。
  // 这两步失败时子进程已经起来了，必须就地收掉：析构看 f_==nullptr 就直接返回，
  // 指望它清理等于漏一个进程。
  const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(mine),
                                 rd_mode ? (_O_RDONLY | _O_BINARY)
                                         : (_O_WRONLY | _O_BINARY));
  if (fd < 0) { CloseHandle(mine); reap(pi.hProcess, kReadDrainMs); return; }
  f_ = _fdopen(fd, rd_mode ? "rb" : "wb");
  if (!f_)    { _close(fd);        reap(pi.hProcess, kReadDrainMs); return; }
  proc_ = pi.hProcess;                                // 交给 close()
#else
  (void)buf_bytes;                  // popen 不给缓冲控制，Linux 上也没这个瓶颈
  f_ = popen(cmd.c_str(), mode_ == Mode::Read ? "r" : "w");
#endif
}

Proc::~Proc() { close(); }

int Proc::close() {
  if (!f_) return 0;                // 幂等：已收过
#ifdef _WIN32
  auto h = static_cast<HANDLE>(proc_);
  proc_ = nullptr;
  FILE* f = f_;
  f_ = nullptr;
  // 构造成功时 f_ 与 proc_ 必然同时非空，这里只是别让手滑改坏时传 nullptr 进内核
  if (!h) { fclose(f); return 0; }
  if (mode_ == Mode::Read) {
    // **必须先处置子进程再 fclose 读端**：先关我们这端的话，仍在写的 ffmpeg 会
    // 拿到管道错误并把 "Error muxing a packet / Error writing trailer" 一串刷到
    // stderr —— 提前结束（--max-frames）时那是必然发生的假报错，会盖住真日志。
    const int code = reap(h, kReadDrainMs);
    fclose(f);
    return code;
  }
  // 写模式相反：先 fclose 让子进程读到 EOF（ffmpeg 要靠它写完 moov），再无限期等。
  // 这里不能设超时：收尾耗时取决于它还压着多少帧没编码，等不够就是把视频写坏。
  fclose(f);
  return reap(h, INFINITE);
#else
  FILE* f = f_;
  f_ = nullptr;
  const int st = pclose(f);          // pclose 自身就是「关管道 + waitpid」
  return WIFEXITED(st) ? WEXITSTATUS(st) : st;
#endif
}

}  // namespace swim
