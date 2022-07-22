project(scrfdncnn)

cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.5.1-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc imgcodecs )

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20210720-android-vulkan-shared/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(scrfdncnn SHARED scrfdncnn.cpp scrfd.cpp ndkcamera.cpp ast_rawosd.c)

target_link_libraries(scrfdncnn ncnn ${OpenCV_LIBS} camera2ndk mediandk)
