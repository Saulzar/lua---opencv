CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

SET(src opencv.cpp)
SET(luasrc init.lua lena.jpg affine.lua util.lua test.lua)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

ADD_TORCH_PACKAGE(opencv "${src}" "${luasrc}" "Image Processing")
TARGET_LINK_LIBRARIES(opencv luaT TH ${OpenCV_LIBS})
