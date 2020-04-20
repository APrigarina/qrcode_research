cmake_minimum_required(VERSION 3.2.2)
project(qrcode_research)


#find_package(Boost REQUIRED COMPONENTS program_options filesystem)
find_package(OpenCV 4 REQUIRED)

#set(THREADS_PREFER_PTHREAD_FLAG ON)
#find_package(Threads REQUIRED)

#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Options. Turn on with 'cmake -Dmyvarname=ON'.
#option(test "Build all tests." ON) # Makes boolean 'test' available.
#enable_testing()

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} qrcode_v2.cpp)

target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS} )
