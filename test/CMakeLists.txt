cmake_minimum_required(VERSION 3.1)

project(cpp-liblinear)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Debug or Release")
endif()

enable_language(CXX)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

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
