# EasyOCR-cpp OpenVINO

### Custom C++ implementation of [EasyOCR](https://github.com/JaidedAI/EasyOCR) with [OpenVINO](https://github.com/openvinotoolkit/openvino) backend
### Built and tested on Windows 11, openvino2024.4.0 and OpenCV 4.6

This C++ project implements the pre/post processing to run a OCR pipeline consisting of a text detector [CRAFT](https://arxiv.org/abs/1904.01941), and a CRNN based text recognizer. Unlike the EasyOCR python which is API based, this repo provides a set of classes to show how you can integrate OCR in any C++ program for maximum flexibility. The openvinoExample.cpp main program highlights how to utilize all elements of the EasyOCR-cpp pipeline. Because a test program is only provided, make sure to configure your input image within OpenvinoExample.cpp if you only plan to utilize the test program. 


## Setup with MSVS

### Dependencies
Install [OpenVINO](https://docs.openvino.ai/2024/get-started/install-openvino.html?PACKAGE=OPENVINO_BASE&VERSION=v_2024_4_0&OP_SYSTEM=WINDOWS&DISTRIBUTION=ARCHIVE)

Use OpenCV Windows installer and unzip (v4.6) - > [OpenCV libs](https://opencv.org/releases/)

Make sure to change the location in the  [Makefile](https://github.com/avbelova/EasyOCR-cpp/blob/afc2090b6d32dda4461d3a361abb7eaa80116ff9/CMakeLists.txt#L11) for OpenCV to point to your OpenCV build dir 

Create a build directory within the repo, cd to it and run
```
cmake  ..
```


This will generate a solution within the build folder you can open up in Visual Studio. **Make sure to use the Release config when building**

### Running

Configure your input image  [here](https://github.com/avbelova/EasyOCR-cpp/blob/afc2090b6d32dda4461d3a361abb7eaa80116ff9/OpenvinoExample.cpp#L14). Currently the test program is using the test.jpg which comes in the repo.

Set up OpenVINO environment:
```
C:\Program Files (x86)\Intel\openvino_2024.4.0\setupvars.bat 
```
Set up OpenCV environment 
```
C:\OpenCV\opencv\build\setup_vars_opencv4.cmd
```

Launch from command-line, or within Visual Studio after building.

**Since its designed to be used in a C++ program, text is not being written to disk at the moment** An output image will be generated in the main repo dir containing an annotated version of the input image with detection bounding boxes


