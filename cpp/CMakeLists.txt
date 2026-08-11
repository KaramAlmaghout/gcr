cmake_minimum_required(VERSION 3.16)
project(gcr_cpp_benchmark LANGUAGES CXX)

add_executable(gcr_benchmark src/gcr_benchmark.cpp)
target_compile_features(gcr_benchmark PRIVATE cxx_std_17)

if(MSVC)
  target_compile_options(gcr_benchmark PRIVATE /O2 /W4)
else()
  target_compile_options(gcr_benchmark PRIVATE -O3 -Wall -Wextra -pedantic)
endif()
