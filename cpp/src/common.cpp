#include "swim/common.h"

#include <algorithm>
#include <vector>

namespace swim {

void Timers::report(const std::string& title, double total_ms, int64_t frames) const {
  std::lock_guard<std::mutex> lk(mu_);
  std::vector<std::pair<std::string, std::pair<double, int64_t>>> v(stat_.begin(),
                                                                   stat_.end());
  std::sort(v.begin(), v.end(),
            [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
  printf("\n[%s] 总计 %.1f s / %lld 帧 = %.1f ms/帧 (%.1f fps)\n", title.c_str(),
         total_ms / 1000.0, static_cast<long long>(frames),
         frames ? total_ms / frames : 0.0,
         total_ms > 0 ? frames * 1000.0 / total_ms : 0.0);
  printf("  %-18s %10s %8s %10s\n", "阶段", "累计(s)", "占比", "每帧(ms)");
  for (const auto& [k, s] : v)
    printf("  %-18s %10.2f %7.1f%% %10.2f\n", k.c_str(), s.first / 1000.0,
           total_ms > 0 ? 100.0 * s.first / total_ms : 0.0,
           frames ? s.first / frames : 0.0);
}

}  // namespace swim
