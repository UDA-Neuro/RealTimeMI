cmake_minimum_required(VERSION 3.16)
project(RealTimeMI LANGUAGES CXX)

enable_testing()
add_executable(streaming_pac_test tests/StreamingPacTest.cpp)
target_include_directories(streaming_pac_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(streaming_pac_test PRIVATE cxx_std_17)
add_test(NAME streaming_pac COMMAND streaming_pac_test)
