cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(openinoTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED)

# Find OPENCV automatically if built from dockerfile
if (DEFINED ENV{build_from_docker_file})
    find_package(OpenCV 4 REQUIRED)
    find_package(Threads REQUIRED)
# Else force the path to opencv here if executed on Windows
else ()
    find_package(OpenCV REQUIRED PATHS C:/Users/sasso/Downloads/new/opencv/build )
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
# Get all character text files  from the 'lang' directory
file(GLOB LANG_FILES "lang/*")

file(GLOB MODELS "models/*")


# Set the source file path
set(SOURCE_FILE_PATH "test.jpg")

# Set the destination file path in the binary directory
set(DEST_FILE_PATH ${CMAKE_CURRENT_BINARY_DIR})

# Copy the file to the binary directory
configure_file(${SOURCE_FILE_PATH} ${DEST_FILE_PATH} COPYONLY)

# Copy files obtained from FILE_PATHS_GLOB_1
foreach(FILE_PATH ${LANG_FILES})
    get_filename_component(FILE_NAME ${FILE_PATH} NAME)  # Get the file name
    file(COPY ${FILE_PATH} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach()

# Copy files obtained from FILE_PATHS_GLOB_2
foreach(FILE_PATH ${MODELS})
    get_filename_component(FILE_NAME ${FILE_PATH} NAME)  # Get the file name
    file(COPY ${FILE_PATH} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach()

add_executable(openvinoTest OpenvinoExample.cpp 
			 src/OpenvinoModel.cpp
			 src/CRAFT.cpp
			 src/CRNN.cpp)
target_include_directories(openvinoTest PUBLIC "include/")
target_link_libraries(openvinoTest ${OpenCV_LIBS} ${MKL} PRIVATE openvino::runtime)
set_property(TARGET openvinoTest PROPERTY CXX_STANDARD 17)

if (MSVC)
  set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT openvinoTest)
  add_custom_command(TARGET openvinoTest
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     $<TARGET_FILE_DIR:openvinoTest>)
endif (MSVC)