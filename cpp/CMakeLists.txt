# swim_analyse C++ 实时分析（Plan C：画布 detect + 画布 RTMPose，全程 GPU 驻留）
#
# 依赖：CUDA 12.x、TensorRT 10.11.0.33、OpenCV 4.x
# Linux:
#   cmake -B build -DTRT_ROOT=/opt/trt/TensorRT-10.11.0.33 && cmake --build build -j
# Windows（VS 2022 + CUDA + TRT 官方 zip）:
#   cmake -B build -G "Visual Studio 17 2022" -DTRT_ROOT=C:/TensorRT-10.11.0.33 \
#         -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
#   cmake --build build --config Release
cmake_minimum_required(VERSION 3.18)
project(swim_analyse LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# kernels.cu 间接用到 std::array/constexpr，而 CMake 记录的 CUDA 默认标准是 14
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# 与下面链接的 CUDA::cudart 保持一致：VS 的 CUDA 集成默认静态 runtime，
# 不显式指定会让静态与动态两份 cudart 同时进链接行
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# CMAKE_BUILD_TYPE 只对单配置生成器有效，VS 下是空操作
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_CONFIGURATION_TYPES)
  message(STATUS "多配置生成器：构建必须加 --config Release，否则拿到的是 Debug 二进制（慢一个量级）")
endif()

# H800=90, RTX40xx=89, RTX30xx=86；台式机按需覆盖 -DCMAKE_CUDA_ARCHITECTURES=89
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 89 90)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)

# ── TensorRT（无官方 find module，手动定位）───────────────────────────────
# Windows 版的库名带版本后缀（nvinfer_10.lib / nvonnxparser_10.lib），
# Linux 版是 libnvinfer.so，因此两套名字都要找。
set(TRT_ROOT "" CACHE PATH "TensorRT 10.11 根目录")
# find_path/find_library 的结果进 cache，命中后不再重查；本机多版本 TRT 并存，
# 换 -DTRT_ROOT= 重新 configure 必须先清掉旧结果，否则会静默沿用上次的版本
if(NOT "${TRT_ROOT}" STREQUAL "${TRT_ROOT_LAST}")
  unset(TRT_INCLUDE CACHE)
  foreach(lib nvinfer nvonnxparser)
    unset(TRT_${lib} CACHE)
  endforeach()
  set(TRT_ROOT_LAST "${TRT_ROOT}" CACHE INTERNAL "上次 configure 使用的 TRT_ROOT")
endif()
# 给了 TRT_ROOT 就只在它下面找（否则可能命中系统里的另一个版本）；
# 没给则退回系统路径（Linux 的 apt 包装在 /usr 下）
if(TRT_ROOT)
  set(TRT_INC_PATH PATHS ${TRT_ROOT}/include NO_DEFAULT_PATH)
  set(TRT_LIB_PATH PATHS ${TRT_ROOT}/lib ${TRT_ROOT}/lib64 NO_DEFAULT_PATH)
else()
  set(TRT_INC_PATH HINTS /usr/include/x86_64-linux-gnu /usr/include)
  set(TRT_LIB_PATH HINTS /usr/lib/x86_64-linux-gnu)
endif()
find_path(TRT_INCLUDE NvInfer.h ${TRT_INC_PATH})
foreach(lib nvinfer nvonnxparser)
  find_library(TRT_${lib} NAMES ${lib} ${lib}_10 ${TRT_LIB_PATH})
  if(NOT TRT_${lib})
    message(FATAL_ERROR "找不到 ${lib}，请用 -DTRT_ROOT=<TensorRT 根目录> 指定")
  endif()
  list(APPEND TRT_LIBS ${TRT_${lib}})
endforeach()
if(NOT TRT_INCLUDE)
  message(FATAL_ERROR "找不到 NvInfer.h，请用 -DTRT_ROOT=<TensorRT 根目录> 指定")
endif()
message(STATUS "TensorRT: ${TRT_INCLUDE}")

add_executable(swim_analyse
  src/main.cpp src/pipeline.cpp src/trt_engine.cpp
  src/frame_source.cpp src/metrics.cpp src/common.cpp src/kernels.cu)

target_include_directories(swim_analyse PRIVATE include ${TRT_INCLUDE})
target_link_libraries(swim_analyse PRIVATE
  ${TRT_LIBS} CUDA::cudart ${OpenCV_LIBS})

if(UNIX)
  target_link_libraries(swim_analyse PRIVATE dl pthread)
endif()

# 运行期要找到 TRT 的动态库：不拷 DLL（TRT lib 目录 2.3 GB），只把路径写进
# VS 调试环境 / ELF 的 RPATH，省掉每次手动设 PATH / LD_LIBRARY_PATH
if(TRT_ROOT)
  if(MSVC)
    set_target_properties(swim_analyse PROPERTIES
      VS_DEBUGGER_ENVIRONMENT "PATH=${TRT_ROOT}/lib;%PATH%")
  else()
    set_target_properties(swim_analyse PROPERTIES BUILD_RPATH "${TRT_ROOT}/lib")
  endif()
endif()
# TRT 头文件自身有 deprecated 声明，屏蔽以免淹没真实告警
if(MSVC)
  # 源码是 UTF-8 无 BOM：不给 /source-charset 的话 MSVC 按 ACP(936) 解中文注释，
  # 会把注释里的字节吃进语法（与 .ps1 无 BOM 的坑同源）。
  # 执行字符集保持 UTF-8，运行期由 main() 调 SetConsoleOutputCP(CP_UTF8) 对齐控制台。
  #
  # /utf-8 只能救 cl，救不了 nvcc：nvcc 的 EDG 前端在 ACP=936 的机器上按 ANSI 解
  # .cu 源码，中文**字面量**会被吞掉右引号（加 BOM 也无效），因此 kernels.cu 里的
  # 字面量一律 ASCII、只有注释用中文（注释不进语法，安全）。
  target_compile_options(swim_analyse PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8 /wd4996>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/wd4996>)
else()
  target_compile_options(swim_analyse PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>)
endif()
