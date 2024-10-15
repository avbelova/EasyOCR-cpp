# EasyOCR-cpp OpenVINO

### Custom C++ implementation of [EasyOCR](https://github.com/JaidedAI/EasyOCR) with [OpenVINO](https://github.com/openvinotoolkit/openvino) backend
### Built and tested on Windows 11, openvino2024.4.0 and OpenCV 4.6

This C++ project implements the pre/post processing to run a OCR pipeline consisting of a text detector [CRAFT](https://arxiv.org/abs/1904.01941), and a CRNN based text recognizer. Unlike the EasyOCR python which is API based, this repo provides a set of classes to show how you can integrate OCR in any C++ program for maximum flexibility. The openvinoExample.cpp main program highlights how to utilize all elements of the EasyOCR-cpp pipeline. Because a test program is only provided, make sure to configure your input image within OpenvinoExample.cpp if you only plan to utilize the test program. 


## Setup with MSVS

### Dependencies
Install [OpenVINO](https://docs.openvino.ai/2024/get-started/install-openvino.html?PACKAGE=OPENVINO_BASE&VERSION=v_2024_4_0&OP_SYSTEM=WINDOWS&DISTRIBUTION=ARCHIVE)

Use OpenCV Windows installer and unzip (v4.6) - > [OpenCV libs](https://opencv.org/releases/)

Make sure to change the location in the  [Makefile](https://github.com/avbelova/EasyOCR-cpp/blob/afc2090b6d32dda4461d3a361abb7eaa80116ff9/CMakeLists.txt#L11) for OpenCV to point to your OpenCV build dir 

Set up OpenVINO environment:
```
C:\Program Files (x86)\Intel\openvino_2024.4.0\setupvars.bat 
```
Set up OpenCV environment 
```
C:\OpenCV\opencv\build\setup_vars_opencv4.cmd
```
Create a build directory within the repo, cd to it and run cmake

```
mkdir build
cd build
cmake  ..
```


This will generate a solution within the build folder you can open up in Visual Studio. **Make sure to use the Release config when building**

### Running

Configure your recognition model, input image and inference device [here](https://github.com/avbelova/EasyOCR-cpp/blob/0754743a0128266dc624964d01d45e2147b290fe/OpenvinoExample.cpp#L13C3-L15C32). Configure a characters list for your language [here](https://github.com/avbelova/EasyOCR-cpp/blob/0754743a0128266dc624964d01d45e2147b290fe/src/CRNN.cpp#L10) By default the openvinoTest program is using the english recognition model, test.jpg as an input image which comes in the repo and running inference on CPU. 

Launch from command-line, or within Visual Studio after building. **Don't forget to source environment variables for both OpenVINO and OpenCV as described above**

### Adding more languages support

This repo contains a recognition model for English [recognition_model.xml](https://github.com/avbelova/EasyOCR-cpp/blob/openvino-integration/models/recognition_model.xml) and for the most popular european languages based on latin symbols (German, French, Inalian, Spanish, etc.) [recognition_model_latin.xml](https://github.com/avbelova/EasyOCR-cpp/blob/openvino-integration/models/recognition_model_latin.xml). Please note that for the inference with OpenVINO both <model>.xml and <model>.bin files are required, they should have the same name and be placed in the same folder, but in a code you can specify only path to the .xml file. For both languages there are corresponding language characters files: [english_g2_characters.txt](https://github.com/avbelova/EasyOCR-cpp/blob/openvino-integration/lang/english_g2_characters.txt) and [latin_char.txt](https://github.com/avbelova/EasyOCR-cpp/blob/openvino-integration/lang/latin_char.txt).

**If you need models for more languages, you can get them already in OpenVINO format following these steps:** 
1. Create and activate python virtual environment:
   ```
   python -m venv env
   env\Scripts\activate
   ```
2. Install a patched Python EasyOCR version:
   ```
   pip install git+https://github.com/avbelova/EasyOCR.git@model-convert-and-save
   ```
3. Run EasyOCR with the needed language as usuall in Python. For example the following code gets a Chineese recognition model:
   ```
   import cv2
   import easyocr

   img=cv2.imread("chinese.jpg")
   reader = easyocr.Reader(['ch_sim'], gpu="ov_cpu")
   result = reader.readtext(img, detail = 0)
   print(result)
   ```
4. Find a recogntion model in OpenVINO format in the directory from where you run EasyOCR in the previous step.
5. Don't forget to obtain a character list for your model.

