#include "CRNN.h"
#include <iostream>
#include <fstream>
#include <tuple>


CRNNModel::CRNNModel() : OpenVINOModel()
{
	// eventually read from a config!
	std::string filename = "english_g2_characters.txt";
	std::ifstream file(filename);
	if (!file.is_open()) 
	{
		std::cerr << "Error: Unable to open file " << filename << std::endl;
	}

	std::string line;
	std::getline(file, line);
	// add blank token
	this->characters.push_back(' ');
	// Convert string to vector of characters
	for (char c : line) {
		this->characters.push_back(c);
	}

	file.close();
}

float resizeComputeRatio(cv::Mat& img, int modelHeight)
{
	float ratio = float(img.cols) / float(img.rows);
	if (ratio < 1.0)
	{
		ratio = 1.0 / ratio;
		cv::resize(img, img, cv::Size(modelHeight, int(modelHeight * ratio)));
	}
	else
	{

		cv::resize(img, img, cv::Size(int(modelHeight * ratio), modelHeight));

	}

	return ratio;
}


std::string CRNNModel::greedyDecode(std::vector<int>& encoded)
{
	std::string text="";
	for (int i = 0; i < encoded.size(); i++)
	{
		if (encoded[i]!= 0 && encoded[i] != encoded[i + 1])
		{
			text.push_back(this->characters[encoded[i]]);

		}
	}
	return text;
}



ov::Tensor CRNNModel::preProcess(cv::Mat& det)
{
	// Default model height used in easyOCR
	float ratio = resizeComputeRatio(det, 64);
	double alpha = 1.28;
	double beta = 0;
	//cv::equalizeHist(det, det);
	//det.convertTo(det, -1, alpha, beta);

	//at least 128 in length
	auto processedTensor = this->normalize(det);
	return processedTensor;
}

std::vector<TextResult> CRNNModel::recognize(std::vector<BoundingBox>& dets, cv::Mat& img)
{
	// returns max width for padding and resize
	std::vector<ov::Tensor> processed;
	std::vector<TextResult> results;
	for (auto& x : dets)
	{
		TextResult res;
		cv::Mat det = img(cv::Rect(x.topLeft.x, x.topLeft.y, (x.bottomRight.x - x.topLeft.x), (x.bottomRight.y - x.topLeft.y))).clone();
		if (det.rows < 5)
			continue;

		ov::Tensor processedTensor = this->preProcess(det);
		//auto ss = std::chrono::high_resolution_clock::now();
		ov::Tensor output = this->predict(processedTensor);
		/*auto ee = std::chrono::high_resolution_clock::now();
		auto difff = ee - ss;
		*/
		//std::cout << "TOTAL INFERENCE RECORNGITON TIME " << std::chrono::duration <double, std::milli>(difff).count() << " ms" << std::endl;

		//post process and decode
		auto confidence = this->softmax(output, 2);
		float* confidence_data = confidence.data<float>();

		ov::Shape shape = confidence.get_shape();
		
		std::vector<float> maxes;
		std::vector<int> indices;
		int counter = 0;
		for (int i=0; i< confidence.get_shape()[1]; i++)
		{
			float max = confidence_data[counter];
			int idx = 0;
			for (int j = 0; j < confidence.get_shape()[2]; j++)
			{
				if (confidence_data[counter] > max)
				{
					max = confidence_data[counter];
					idx = j;
				}
				counter++;
			}
			maxes.push_back(max);
			indices.push_back(idx);
		}

		std::string text = this->greedyDecode(indices);
		res.text = text;
        res.confidence = *confidence.data<float>();
		res.coords = x;
		results.push_back(res);
	}
	return results;
}

ov::Tensor CRNNModel::normalize(cv::Mat& processed)
{
	auto converted = this->convertToTensor(processed.clone(), true, false); 
	float* converted_data = converted.data<float>();
	for (int i = 0; i < converted.get_size(); i++)
	{
		converted_data[i] = (converted_data[i] - (.5) / (.5));
	}

	return converted;
}
ov::Tensor CRNNModel::softmax(ov::Tensor& input, int dim) 
{
	ov::Shape shape = input.get_shape();
	float* input_data = input.data<float>();

	ov::Tensor output(input.get_element_type(), shape);
	float* output_data = output.data<float>();

	size_t batch_size = shape[0]; 
	size_t num_elements_per_axis = shape[dim]; 

	for (size_t i = 0; i < input.get_size(); ++i) {
		output_data[i] = std::exp(input_data[i]);
	}

	for (size_t b = 0; b < batch_size; ++b) {
		for (size_t j = 0; j < num_elements_per_axis; ++j) {
			float sum_exp = 0.0f;
			for (size_t k = 0; k < num_elements_per_axis; ++k) {
				sum_exp += output_data[b * num_elements_per_axis + k];
			}

			for (size_t k = 0; k < num_elements_per_axis; ++k) {
				output_data[b * num_elements_per_axis + k] /= sum_exp;
			}
		}
	}

	return output;
}
void CRNNModel::print_tensor(ov::Tensor& tensor)
{
	float* data_ptr = tensor.data<float>();
	int index = 0;
	for (int i = 0; i < tensor.get_shape()[1]; i++) 
	{
		for (int j = 0; j < tensor.get_shape()[2]; j++) 
		{
			std::cout << data_ptr[index] << " ";
			index++;
		}
		std::cout << std::endl;
	}



}