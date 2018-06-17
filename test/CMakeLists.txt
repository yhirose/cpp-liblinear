cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(
  ..
  ./vendor/liblinear
  ./vendor/Catch2/single_include
  ./vendor/cpp-peglib
  ./vendor/cpp-mmaplib)

add_executable(
  test-main
  ./vendor/liblinear/linear.cpp
  ./vendor/liblinear/tron.cpp
  ./vendor/liblinear/blas/daxpy.c
  ./vendor/liblinear/blas/ddot.c
  ./vendor/liblinear/blas/dnrm2.c
  ./vendor/liblinear/blas/dscal.c
  test.cc)

enable_testing()
add_test(
  NAME TestMain
  COMMAND test-main
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
