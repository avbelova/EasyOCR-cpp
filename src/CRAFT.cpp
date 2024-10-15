#include "CRAFT.h"
#include <algorithm>
#include <fstream>
#include <sstream>

HeatMapRatio CraftModel::resizeAspect(cv::Mat& img)
{
	HeatMapRatio output;
	try {
		cv::Mat processed;
		int channels = img.channels();
		float height = img.rows;
		float width = img.cols;
		float targetSize = std::max(height, width);
		const int canvasSize = 2560;
		int targetH, targetW;
		if (targetSize > canvasSize)
		{
			targetSize = canvasSize;
		}
		float ratio = targetSize / std::max(height, width);
		targetH = int(height * ratio);
		targetW = int(width * ratio);
		cv::resize(img, img, cv::Size(targetW, targetH));

		int h32 = targetH;
		int w32 = targetW;
		if (targetH % 32 != 0)
		{
			h32 = targetH + (32 - targetH % 32);
		}

		if (targetW % 32 != 0)
		{
			w32 = targetW + (32 - targetW % 32);
		}

		cv::Mat resized = cv::Mat::zeros(h32, w32, CV_32FC3);
		cv::Range colRange = cv::Range(0, cv::min(resized.cols, img.cols)); //select maximum allowed cols
		cv::Range rowRange = cv::Range(0, cv::min(resized.rows, img.rows)); //select maximum allowed cols
		img(rowRange, colRange).clone().copyTo(resized(rowRange, colRange));
		cv::Size heatMapSize = cv::Size(int(targetW / 2), int(targetH / 2));
		output.ratio = ratio;
		output.heatMapSize = heatMapSize;
		output.img = resized;
		this->ratio = ratio;
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
	return output;
}
std::vector<BoundingBox> CraftModel::mergeBoundingBoxes(std::vector<BoundingBox>& dets, float distanceThresh, int height, int width)
{
	// represents how much we change the top left Y
	std::sort(dets.begin(), dets.end(), boxSorter());
	bool merge = NULL;
	std::vector<BoundingBox> merged;
	int newTopLeft;
	bool firstRun = true;
	for (int i = 0; i < dets.size(); i++)
	{
		cv::Point newBottomRight = dets[i].bottomRight;
		int minY = 0;
		int maxY = 0;
		if (i < dets.size())
		{
			float x = dets[i].bottomRight.x;
			float xPrime = dets[i + 1].topLeft.x;
			float ratio = x / xPrime;
			bool isNegative = false;
			if (x - xPrime < 0) isNegative = true;
			float w = dets[i].bottomRight.x - dets[i].topLeft.x;
			float h = dets[i].bottomRight.y - dets[i].topLeft.y;
			if (width > 5 * height)
			{
				// box is a line, skip merging
				merge = false;
				continue;
			}
			//merge box, store point
			if (ratio > distanceThresh && ratio < 1.4 && std::abs(dets[i].bottomRight.y - dets[i + 1].bottomRight.y) < 20)
			{
				newBottomRight = dets[i + 1].bottomRight;
				if (dets[i + 1].topLeft.y < dets[i].topLeft.y)
				{
					if (minY > dets[i + 1].topLeft.y)
					{
						minY = dets[i + 1].topLeft.y;
					}
				}
				else {
					if (dets[i + 1].bottomRight.y > maxY)
					{
						maxY = dets[i + 1].bottomRight.y;
					}

				}
				merge = true;
			}
			else
			{
				newBottomRight = dets[i].bottomRight;
				merge = false;
			}
			if (firstRun)
			{
				// store first
				newTopLeft = i;
				firstRun = false;
			}
			//cant merge anymore other box is too far, build new box
			if (!merge)
			{
				BoundingBox newBox;
				newBox.topLeft = dets[newTopLeft].topLeft;
				newBox.topLeft.y -= minY;
				//margin
				newBox.topLeft.y *= .998;
				newBox.topLeft.x *= .998;
				if (newBox.topLeft.y < 0)
				{
					newBox.topLeft.y = 0;
				}

				if (newBox.topLeft.x < 0)
				{
					newBox.topLeft.x = 0;
				}
				newBox.topLeft.x = int(newBox.topLeft.x);
				newBox.topLeft.y = int(newBox.topLeft.y);

				newBox.bottomRight = newBottomRight;
				newBox.bottomRight.y += maxY;
				// margin
				newBox.bottomRight.y *= 1.003;
				newBox.bottomRight.x *= 1.003;

				if (newBox.bottomRight.y > height)
				{
					newBox.bottomRight.y = height - 1;
				}

				if (newBox.bottomRight.x > width)
				{
					newBox.bottomRight.x = width - 5;
				}
				newBox.bottomRight.x = int(newBox.bottomRight.x);
				newBox.bottomRight.y = int(newBox.bottomRight.y);
				merged.push_back(newBox);
				// move top left box to next box
				newTopLeft = i + 1;
				minY = 0;
				maxY = 0;
			}
		}
	}
	return merged;
}

std::vector<BoundingBox> CraftModel::getBoundingBoxes(const ov::Tensor& input, const ov::Tensor& output, float textThresh, float linkThresh, float lowText)
{
	std::vector<BoundingBox> detBoxes;
	const ov::Shape output_shape = output.get_shape();
	auto output_size = output.get_size();
	float* output_data = output.data<float>();
	std::vector<uint8_t> linkMapData = {};
	std::vector<uint8_t> textMapData = {};
	for (size_t i = 0; i < output_size; i+=2)
	{	
		float scaled_value = output_data[i] * 255.0f;
		float cla = std::max(0.0f, std::min(255.0f, scaled_value));
		uint8_t casted = static_cast<uint8_t>(cla);
		linkMapData.push_back(casted);
	}

	for (size_t i = 1; i < output_size; i += 2)
		{	
			float scaled_value = output_data[i] * 255.0f;
			float cla = std::max(0.0f, std::min(255.0f, scaled_value));
			textMapData.push_back(cla);
		}
	const ov::Tensor linkMapTensor(ov::element::u8, { 1, output_shape[1],output_shape[2] }, linkMapData.data());
	const ov::Tensor textMapTensor(ov::element::u8, { 1, output_shape[1],output_shape[2] }, textMapData.data());
	const cv::Mat linkMap = convertToMat(linkMapTensor, true, true, false, false).clone();
	const cv::Mat textMap = convertToMat(textMapTensor, true, true, false, false).clone();
	auto tempTextMap = textMap.clone();
	int r = linkMap.rows;
	int c = linkMap.cols;
	cv::Mat linkScore, textScore;
	cv::threshold(linkMap, linkScore, (linkThresh * 255), 255, 0);
	cv::threshold(textMap, textScore, (lowText * 255), 255, 0);
	cv::Mat outputScore = linkScore.clone() + textScore.clone();
	cv::min(cv::max(outputScore, 0), 255, outputScore);
	outputScore.convertTo(outputScore, CV_8UC3);
	cv::Mat labels, stats, centroids;
	std::vector<int> mapper;
	int numLabels = cv::connectedComponentsWithStats(outputScore, labels, stats, centroids, 4, CV_32S);
	for (int i = 1; i < numLabels; i++)
	{
		int area = stats.at<int>(i, cv::CC_STAT_AREA);
		if (area < 10)
			continue;
		cv::Mat segMap = cv::Mat::zeros(textMap.size(), CV_8UC1);
		cv::Mat mask = (labels == i);
		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(textMap, &minVal, &maxVal, &minLoc, &maxLoc, mask);
		mapper.push_back(i);
		segMap.setTo(255, labels == i);
		cv::Mat linkMask = (linkScore == 1) & (textScore == 0);
		segMap.setTo(0, linkMask);
		int x = stats.at<int>(i, cv::CC_STAT_LEFT);
		int y = stats.at<int>(i, cv::CC_STAT_TOP);
		int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
		int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
		int niter = int(sqrt(area * cv::min(w, h) / (w * h)) * 2);
		int sx = x - niter, ex = x + w + niter + 1, sy = y - niter, ey = y + h + niter + 1;
		// boundary check
		if (sx < 0) sx = 0;
		if (sy < 0) sy = 0;
		if (ex >= c) ex = c;
		if (ey >= r) ey = r;
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1 + niter, 1 + niter));
		cv::dilate(segMap(cv::Range(sy, ey), cv::Range(sx, ex)), segMap(cv::Range(sy, ey), cv::Range(sx, ex)), kernel);
		// make box
		cv::Mat nonZeroCoords;
		cv::findNonZero(segMap, nonZeroCoords);
		cv::Mat npContours = nonZeroCoords.reshape(2, nonZeroCoords.total()).t();
		cv::Mat boxMat;
		cv::RotatedRect rectangle = cv::minAreaRect(npContours);
		cv::boxPoints(rectangle, boxMat);
		cv::Mat box = boxMat;
		BoundingBox detection;
		std::vector<cv::Point> points;
		for (int i = 0; i < 4; i++)
		{
			float colVal;
			float rowVal = box.at<float>(i, 0);
			for (int j = 0; j < 1; j++)
			{
				colVal = box.at<float>(i, 1);
			}
			cv::Point point(rowVal, colVal);
			points.push_back(point);
		}
		std::sort(points.begin(), points.end(), pointSorter());
		detection.topLeft.x = (points[0].x * 2);
		detection.topLeft.y = (points[0].y * 2);
		detection.bottomRight.x = (points[3].x * 2);
		detection.bottomRight.y = (points[3].y * 2);
		if (detection.bottomRight.y < detection.topLeft.y)
		{
			std::cout << "ALIGNEMENT ISSUE" << std::endl;
		}
		// align diamond-shape
		float w1 = cv::norm(box.row(0) - box.row(1));
		float h1 = cv::norm(box.row(1) - box.row(2));
		float box_ratio = std::max(w1, h1) / (std::min(w1, h1) + 1e-5);
		/*
		if (std::abs(1 - box_ratio) <= 0.1) {
			double l, r, t, b;
			cv::minMaxLoc(np_contours.col(0), &l, &r);
			cv::minMaxLoc(np_contours.col(1), &t, &b);
			//box = (cv::Mat_<float>(4, 2) << l, t, r, t, r, b, l, b);
			detection.topLeft.x = int(l) * 2;
			detection.topLeft.y = int(t) * 2;
			detection.bottomRight.x = int(r) * 2;
			detection.bottomRight.y = int(b) * 2;

		}
		*/
		detBoxes.push_back(detection);
		//std::cout << "BOUNDING BOX: " << box << std::endl;

	}
	// # uncomment to see raw output written to disk
	//cv::imwrite("output-heatmap.jpg", outputScore);
	return detBoxes;
}

cv::Mat CraftModel::normalize(const cv::Mat& img)
{
	std::vector<cv::Mat> channels(3);
	cv::Mat output;
	// split img:
	split(img, channels);
	channels[0] = (channels[0] - (.485 * 255)) / (.229 * 255);
	channels[1] = (channels[1] - (.456 * 255)) / (.224 * 255);
	channels[2] = (channels[2] - (.406 * 255)) / (.225 * 255);
	merge(channels, output);
	return output;
}

ov::Tensor CraftModel::preProcess(const cv::Mat& matInput)
{
	//Normalize the input using mean + std values from easyOCR
	cv::Mat normedMatInput = this->normalize(matInput.clone()).clone();
	ov::Tensor input = this->convertToTensor(normedMatInput);
	return input;
}

std::vector<BoundingBox> CraftModel::runDetector(ov::Tensor& input, bool merge)
{
	int height = input.get_shape()[1];
	int width = input.get_shape()[2];
	auto output = this->predict(input);
	auto ss = std::chrono::high_resolution_clock::now();
	auto detections = this->getBoundingBoxes(input, output);
	if (merge)
		detections = this->mergeBoundingBoxes(detections, .97, height, width);
	//auto ee = std::chrono::high_resolution_clock::now();
	//auto difff = ee - ss;
	//std::cout << "TOTAL preprocessing TIME " << std::chrono::duration <double, std::milli>(difff).count() << " ms" << std::endl;
	return detections;
}