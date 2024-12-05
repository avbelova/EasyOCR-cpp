#include "OpenVINOModel.h"
#include "CRAFT.h"
#include "CRNN.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    CraftModel detection;
    CRNNModel recognition;

    std::string det = "CRAFT-detector.xml";
    std::string rec = "recognition_model.xml";
    std::string filePath = "test.jpg";
    std::string device = "CPU";

    auto check_det = detection.loadModel(det, device);
    auto check_rec = recognition.loadModel(rec, device);

    cv::Mat matInput = detection.loadMat(filePath, false, true).clone();
    HeatMapRatio processed = detection.resizeAspect(matInput);
    cv::Mat clone = processed.img.clone();
    cv::Mat grey = processed.img.clone();
    grey.convertTo(grey, CV_8UC1);
    cv::cvtColor(grey, grey, cv::COLOR_BGR2GRAY);
    clone.convertTo(clone, CV_8UC3);
    ov::Tensor input = detection.preProcess(processed.img.clone());
    auto ss = std::chrono::high_resolution_clock::now();
    std::vector<BoundingBox> dets = detection.runDetector(input, true);
    std::vector<TextResult> results = recognition.recognize(dets, grey);
    auto ee = std::chrono::high_resolution_clock::now();
    auto difff = ee - ss;
    int count = 0;
    for (auto x : dets)
    {
        rectangle(clone, x.topLeft, x.bottomRight, cv::Scalar(0, 255, 0));
        putText(clone, std::to_string(count), (x.bottomRight + x.topLeft) / 2, cv::FONT_HERSHEY_COMPLEX, .6, cv::Scalar(100, 0, 255));
        count++;

    }
    for (auto& result : results)
    {
        std::cout << "LOCATION: " << result.coords.topLeft << " " << result.coords.bottomRight << std::endl;
        std::cout << "TEXT: " << result.text << std::endl;
        std::cout << "CONFIDENCE " << result.confidence << std::endl;
        std::cout << "################################################" << std::endl;
    }
    std::cout << "TOTAL INFERENCE TIME " << std::chrono::duration <double, std::milli>(difff).count() << " ms" << std::endl;

    return 0;
}

