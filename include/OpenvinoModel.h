#ifndef TORCHMODEL_H
#define TORCHMODEL_H
#include "openvino/openvino.hpp"
#include "string"
#include <opencv2/opencv.hpp>

class OpenVINOModel
{
public:
	OpenVINOModel();
	~OpenVINOModel();
	bool loadModel(const std::string& modelPath, const std::string& device);
	ov::Tensor predict(const ov::Tensor& input);
	ov::Tensor convertToTensor(const cv::Mat& img, bool normalize = false, bool color = true);
	cv::Mat convertToMat(const ov::Tensor& output, bool isFloat, bool permute, bool bgr, bool color);
	cv::Mat loadMat(const std::string file, bool grey, bool rgb);
	ov::CompiledModel compiled_model;
};
#endif
