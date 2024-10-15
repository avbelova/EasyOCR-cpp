
#include "OpenvinoModel.h"

OpenVINOModel::OpenVINOModel()
{
}

OpenVINOModel::~OpenVINOModel()
{
}

bool OpenVINOModel::loadModel(const std::string& modelPath, const std::string& device)
{
	bool success = false;
	try
	{
		ov::Core core;
		std::shared_ptr<ov::Model> model = core.read_model(modelPath.c_str());
		ov::preprocess::PrePostProcessor ppp(model);
		ppp.input(0).preprocess().convert_layout({ 0, 3, 1, 2 });
		model = ppp.build();
		compiled_model = core.compile_model(model, device);
		success = true;

	}
	catch (std::exception& e)
	{
		std::cout << "ERRORS";
		std::cout << e.what();
	}
	return success;
}


ov::Tensor OpenVINOModel::predict(const ov::Tensor& input)
{
	ov::InferRequest ireq = this->compiled_model.create_infer_request();
	ireq.set_input_tensor(0, input);
	ireq.infer();
	const ov::Tensor& output_tensor = ireq.get_output_tensor(0);
	return output_tensor;
	
}


ov::Tensor OpenVINOModel::convertToTensor(const cv::Mat& img, bool normalize, bool color)
{
	cv::Mat c = img.clone();
	if (color)
	{
		cv::cvtColor(c, c, cv::COLOR_BGR2RGB);
	}

	float scale = (normalize) ? 1.0 / 255.0 : 1.0;
	int channels = c.channels();
	auto colorRead = (channels == 3) ? CV_32FC3 : CV_32FC1;
	c.convertTo(c, colorRead, scale);

	ov::Shape input_shape = { 1, size_t(c.rows), size_t(c.cols), size_t(channels) };
	ov::element::Type input_type = ov::element::f32;
	ov::Tensor converted = ov::Tensor(input_type, input_shape, c.data);
	ov::Tensor res = ov::Tensor(input_type, input_shape);
	converted.copy_to(res);
	return res;
}

cv::Mat OpenVINOModel::loadMat(const std::string file, bool grey, bool rgb)
{
	auto readMode = (grey) ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
	cv::Mat returnMat = cv::imread(file, readMode);
	return returnMat;
}


cv::Mat OpenVINOModel::convertToMat(const ov::Tensor& output, bool isFloat, bool permute, bool bgr, bool color)
{
	int height = output.get_shape()[1];
	int width = output.get_shape()[2];
	int channels = output.get_shape()[0];
	auto dataType = (channels == 3) ? CV_8UC3 : CV_8UC1;
	cv::Mat outputMat = cv::Mat(cv::Size(width, height), dataType, output.data()); //+channels
	if (bgr)
		cv::cvtColor(outputMat, outputMat, cv::COLOR_RGB2BGR);
	return outputMat.clone();
}
