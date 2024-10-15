#ifndef CRNN_H
#define CRNN_H
#include "string"
#include "OpenvinoModel.h"
#include "CRAFT.h"
#include <opencv2/opencv.hpp>
struct TextResult
{
	std::string text;
	float confidence;
	BoundingBox coords;
};

class CRNNModel : public OpenVINOModel {

public:

	CRNNModel();
	std::vector<TextResult> recognize(std::vector<BoundingBox>& dets, cv::Mat& img);
	ov::Tensor preProcess(cv::Mat& det);
	ov::Tensor normalize(cv::Mat& processed);
	std::string greedyDecode(std::vector<int>& encoded);
	ov::Tensor softmax(ov::Tensor& input, int dim);
	//stores the last computed ratio (resize/rescale) from input image. 
	float ratio;
	std::vector<char> characters;
	void print_tensor(ov::Tensor& tensor);
};
#endif