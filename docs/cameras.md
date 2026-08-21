# 六路 ZCam 相机接入

面向在现场把六台 ZCam 接进 `--cam-dir` 的人。**离线片段跑通不需要读这页。**
命令在 `scripts/cams.sh`（`probe` / `setup` / `verify` / `list` / `run`），
这里只写相机侧的口径与实测约束。

实测机型 **E2-M4（`model":"echom4"`，固件 1.0.8）**，一台 `192.168.1.199`。
前身系统的现场配置在 `sport-detect-haotian/run_swim_xlj_zcams.sh`（8 路
`192.168.3.101-108`）与 `bash/run_zcam.sh`，本页的取值与它对齐。

## 1. 两路流的分工

一台相机同时提供两路编码流，用途不同、能改的参数也不同：

| | stream0（主流） | stream1（子流） |
|---|---|---|
| 分辨率/帧率 | **不能直接设** —— 由全局 `movfmt` 决定 | `stream_setting?index=stream1&width=&height=&fps=` |
| 能设什么 | `venc`（h264/h265）、`bitrate` | `venc` / `width` / `height` / `fps` / `bitrate` |
| 我们用哪路 | **就是它**：4K 原生，拼接要的分辨率 | 备用（降分辨率给别的消费者） |

旧脚本里 `# stream0不能720` 那句注释说的就是这件事：想要 720p 主流做不到，
只能把 `movfmt` 设成 1080P 再用 stream1 降到 720。

**约束（来自 `rtsp-h264-stitcher/profiles/PROFILES.md`，本机复核一致）**：
stream1 的分辨率不能大于 stream0；改 stream1 的尺寸前它必须处于 `idle`，
正在 `streaming` 时先把 `send_stream` 切到另一路让它退下来。`cams.sh setup`
在 `CAM_STREAM=1` 时就是这么做的。

## 2. movfmt：4K + 30fps 怎么设

```
GET /ctrl/set?movfmt=4KP29.97
```

格式是 `<分辨率>P<帧率>`。`4K`=3840×2160，`C4K`=4096×2160，`S16`=Super16 裁切。
帧率是 NTSC 值：`29.97` = 30000/1001，`59.94` = 60000/1001。本机 `/info` 报的
合法值有 90 多个，4K 段包含 `4KP23.98 / 4KP29.97 / 4KP59.94 / 4KP25 / 4KP50 /
4KP24`，另有 `(Low Noise)` 与 `WDR` 变体。

**录制中改不动**。查询会看到只读标记，写入静默失败：

```
$ curl "http://<ip>/ctrl/get?k=movfmt"
{"code":0,...,"ro":1,"value":"4KP59.94",...}      ← ro:1

$ curl "http://<ip>/ctrl/mode?action=query"
{"code":0,"desc":"","msg":"rec_ing"}              ← 正在录制

$ curl "http://<ip>/ctrl/set?movfmt=4KP29.97"
{"code":-1,"desc":"","msg":""}                    ← 拒绝，且不报原因
```

所以 `cams.sh setup` 会先 `ctrl/rec?action=stop` 再改（`CAM_KEEP_REC=1` 可跳过，
但那样只能调码率）。**改完必须回读校验** —— 相机对不合法的组合一律静默拒绝，
不查就会以为设上了。

数据集 `20260730-4k-raw` 是 **4K@59.94** 录的，而我们要 30fps。这不影响拼接
（LUT 只关心 3840×2160 的分辨率），但两者的时间轴不同，不要拿同一份基线比划水数。

**设上了要按拉流验，不能只看回读。** 本机 199 实测 `movfmt=4KP29.97` 生效后，
`ffprobe` 报 `h264 3840x2160 29.97fps`，`ffmpeg -c copy` 抓 10.02 s 得 296 帧
（≈29.5 fps，与 29.97 吻合）。`cams.sh verify` 现在就是这么查的：分辨率、编码
（必须 8bit h264/hevc —— 10bit 出 P010，拼接 kernel 只认 NV12）、帧率三项一起报。

帧率一项**只告警不拦**，因为流自报值跟不上 `movfmt` 切换是常态（改成 4KP29.97 后
`avg_frame_rate` 仍可能报 59.94）。时间轴走错帧率会让速度整体翻倍，所以
`cams.sh run` 默认把 `CAM_FMT` 里的帧率下给二进制的 `--fps`（`CAM_FPS=0` 关掉，
自己传 `--fps` 则以你为准）。`--fps` 只改时间轴与离线路限速，不碰解码，
语义见 `cpp/README.md`。

## 3. 拉流

**RTSP 只有一个挂载点** `rtsp://<ip>/live_stream`。实测 `/live_stream1`、
`/stream0`、`/stream1`、`/live/sub` 等全部 404。它给出的是 `send_stream` 当前选中
的那一路，所以"拉子流"= 把相机切到 `send_stream=Stream1`，URL 不变。

```
$ curl "http://<ip>/ctrl/get?k=send_stream"
{...,"value":"Stream0","opts":["Stream0","Stream1","Stream0_with_backup"]}
```

前身项目还有一条 libssp 的路（`zcam_ssp://<ip>:9999?ssp_stream=main`，
`STREAM_MAIN`↔stream0、`STREAM_SEC`↔stream1），它带每帧时间戳，多机对齐更准。
本项目暂时只用 RTSP —— 六路各自独立 demux，时间对齐留给后续（见第 6 节）。

**同一台相机最多约 4 个并发 RTSP 会话**。实测 6 个独立进程同时拉：前 4 个正常
（各约 620 帧/12 s），第 5、6 个直接失败拿不到帧。六路各一台相机时无碍，但要注意
别同时开多个消费者；`cams.sh verify` 因此是串行的。

## 4. 已实测通过的

| 项 | 结果 |
|---|---|
| HTTP 探测 / 参数回读 | 通 |
| RTSP + NVDEC 硬解 | 通，3840×2160，10 s 收 594~600 帧（相机 60fps，即跟得住） |
| 接进拼接链路 | 通。1 路真实 RTSP + 5 路离线片段，画布 5002×2102，**30.3 fps 稳态 82 秒无中断** |
| 画布正确性 | 现场那一路的 bbox 区域有真实画面，其余五路正常，接缝无错位 |
| 全离线回归 | 不受影响：3000 帧 / 91 track / 379 次划水，两次运行逐字节相同 |
| **单路掉线重连** | 通。本机 TCP 转发制造 10 s 断流：画布不停，重连后自动续上，见下 |

**掉线重连实测**（一台真机 199 经 `127.0.0.1:8554` 裸 TCP 转发，转发进程在第 15 s
切断会话并拒接 10 s）：断流后约 1 s 打出「断流，开始重连」，画布**继续出帧**
（帧率从 30.3 缓降到 25.8 fps —— 顶住的旧帧不走限速，掉线期间时间轴按真实墙钟走），
恢复后第 10 回尝试成功，整段共「重连 1 次、整帧沿用旧画面 228 帧」（≈7.6 s，
与 10 s 断流窗口相符，差值是转发恢复到 RTSP 握手完成的时间）。
重连只重开 demux + 解码器，hw device context 与 surface 池不动，所以一次成功的
重连约 1 秒；分辨率若在重连后变了会直接报错（LUT 按单一源尺寸烘的）。

单路拉流不是瓶颈：NVDEC 跟得住相机的 60fps（`speed=1.06x`）。端到端 30.3 fps 的
限制来自六路合起来的 NVDEC 吞吐 + 推理，与离线六路现拼的 38 fps 同源
（见 `cpp/README.md` 的性能一节）。相机若设成 30fps，单路负担减半。

## 5. 踩过的坑（都已修在代码里）

1. **`stimeout` 已改名 `timeout`** —— 新版 ffmpeg 的 rtsp demuxer 只认后者。
   不下发任何超时的后果就是旧脚本注释里那句「连不上会卡住」：无限期阻塞。
   代码里两个都下发，识别不了的键会留在 dict 里被忽略，不是错误。
2. **不要加 `fflags=nobuffer` 与 `reorder_queue_size=0`**。看着能降延迟，实测
   （4K@60，400 帧）会让 RTP 重排失效 → `corrupt decoded frame` +
   `error while decoding MB`，而且更慢（8.7 s vs 7.2 s）。延迟控制交给下游的
   预取深度。
3. **直播路不能背压**。生产者一旦停止读 socket，TCP 接收缓冲堆积、相机侧发送
   阻塞，5 秒后 `av_read_frame` 返回 **ETIMEDOUT(-138)**；若把它当 EOF，整条链路
   会先掉到 3 fps 再静默停住。修法两条：读失败在 live 下重试（连续 3 次才判断流，
   之后转重连而不是收摊），队列满时**丢最旧的一帧**而不是等待。
   重试次数从 20 降到 3 是因为重连便宜了（约 1 s），几十次重试只是白等一分钟。
4. **离线路与直播路混跑时，离线路必须限速**。这是最隐蔽的一条：离线片段会以
   200+ fps 满速解码，把 NVDEC 与 libav 内部占满，直播路的 `av_read_frame` 拿不到
   调度，于是同样触发 -138。用最小复现确认过：同进程内只读不解 rtsp 完全正常
   （109 pkt/s 稳定），一旦并行 5 路满速 cuvid 解码，直播路 5 秒超时一次、20 秒只
   拿到 37 帧。现在混合来源时离线路按各自帧率 `sleep_until` 限速，全离线时不限速。
5. **RTSP 带一路 1536 kb/s 的 PCM 音频**。只在 `av_read_frame` 里 unref 不够，
   libavformat 仍会为它解析排队；用 `AVDISCARD_ALL` 显式丢弃。
6. **每路各建一份 hw device context**（都带 `AV_CUDA_USE_PRIMARY_CONTEXT`，
   所以底层仍是同一个 CUDA primary context，指针可被 kernel 直接读）。共用一个
   `AVBufferRef` 会让六路争同一把 hwctx 锁。

7. **重连时不能在消费者侧空转等**。`StitchFrameSource::next()` 每路只 `pop()` 一次，
   拿不到就用该路上一帧的 `av_frame_ref`（同一张 surface，不拷像素）顶住；一整帧
   都没有新画面时按帧率 sleep 一帧。写成「循环重试直到有帧」会占满 CPU 且让
   q/ESC 停不下来（改的时候自己写错过一次）。

## 6. 还没做的

- **多机时间对齐**。现在六路各自 demux，没有共享时钟。ZCam 的 SSP 协议每帧带
  时间戳，前身项目用它对齐；RTSP 路要靠 RTP 时间戳或相机侧的 EzLink/PixelLink
  同步。六台真机到位后再定。
- **六路真机验证**。现在只有一台，六路拓扑是用「1 路真实 + 5 路离线」和「6 路
  指向同一台」两种方式间接验的。后者受相机 4 会话上限限制，跑不通属预期。
  掉线重连也只在单路上验过 —— 逻辑是每路独立的（重连在各自解码线程里做），
  但「多路同时掉」的 surface 占用峰值没有实测。
