cmake_minimum_required(VERSION 3.24)
project(LotusRain65536CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

add_executable(lotus_rain_sim
    src/main.cu
    src/audio_synth.cpp
)

target_include_directories(lotus_rain_sim PRIVATE src)

target_compile_options(lotus_rain_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr -diag-suppress=20012>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/O2 /fp:fast /W3 /EHsc /utf-8>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/O2,/W3,/EHsc,/utf-8>
)
