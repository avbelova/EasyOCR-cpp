cmake_minimum_required(VERSION 3.14)
project(openvinoTest)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Make openvinoTest a startup project in MSVS
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT openvinoTest)

# Find OpenVINO package (use the latest version's simplified package find)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)  # Load only the inference runtime component
set(OpenCV_DIR "C:/OpenCV/opencv/build")
find_package(OpenCV REQUIRED)

# Include OpenVINO and OpenCV directories
include_directories(${CMAKE_SOURCE_DIR}/include ${OpenVINO_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
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

# Add the executable for your application
add_executable(${PROJECT_NAME} OpenvinoExample.cpp
                               src/OpenvinoModel.cpp
                               src/CRAFT.cpp
                               src/CRNN.cpp)

# Link OpenVINO and OpenCV libraries to your target
target_link_libraries(${PROJECT_NAME} openvino::runtime ${OpenCV_LIBS})

# Print diagnostic information (Optional, useful for troubleshooting)
message(STATUS "OpenVINO include dirs: ${OpenVINO_INCLUDE_DIRS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
