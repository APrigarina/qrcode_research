cmake_minimum_required(VERSION 2.8)
project(sample)
find_package(OpenCV)
add_executable(${PROJECT_NAME} test.cpp )
message("Libs: ${OpenCV_LIBS}")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )