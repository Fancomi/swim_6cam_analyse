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
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# H800=90, RTX40xx=89, RTX30xx=86；台式机按需覆盖 -DCMAKE_CUDA_ARCHITECTURES=89
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 89 90)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio imgcodecs)

# ── TensorRT（无官方 find module，手动定位）───────────────────────────────
# Windows 版的库名带版本后缀（nvinfer_10.lib / nvonnxparser_10.lib），
# Linux 版是 libnvinfer.so，因此两套名字都要找。
set(TRT_ROOT "" CACHE PATH "TensorRT 10.11 根目录")
find_path(TRT_INCLUDE NvInfer.h
  HINTS ${TRT_ROOT}/include /usr/include/x86_64-linux-gnu /usr/include)
foreach(lib nvinfer nvonnxparser)
  find_library(TRT_${lib} NAMES ${lib} ${lib}_10
    HINTS ${TRT_ROOT}/lib ${TRT_ROOT}/lib64 /usr/lib/x86_64-linux-gnu)
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
# TRT 头文件自身有 deprecated 声明，屏蔽以免淹没真实告警
if(MSVC)
  # 源码是 UTF-8 无 BOM：不给 /source-charset 的话 MSVC 按 ACP(936) 解中文注释，
  # 会把注释里的字节吃进语法（与 .ps1 无 BOM 的坑同源）。
  # 执行字符集保持 UTF-8，运行期由 main() 调 SetConsoleOutputCP(CP_UTF8) 对齐控制台。
  target_compile_options(swim_analyse PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8 /wd4996>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/wd4996>)
else()
  target_compile_options(swim_analyse PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>)
endif()
