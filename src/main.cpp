#include <iostream>
#include <string>
#include <bits/stdc++.h>
#include <experimental/filesystem>
#include <iterator>
#include <queue>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;

int process_frame(Mat left, Mat right, string name);
int create_combined_output(Mat left, Mat right, string name);
int process_dataset();
bool startsWith(string mainStr, string toMatch);

int DPI=96;
string DATASET = "data/1";
string DATASET_LEFT = DATASET+"/left/";
string DATASET_RIGHT = DATASET+"/right/";
string DATASET_DISPARITIES = DATASET+"/disparities/";
string DATASET_COMBINED = DATASET+"/combined/";

constexpr float CONFIDENCE_THRESHOLD = 0;
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 80;

const cv::Scalar colors[] =
{
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};
const auto NUM_COLORS = sizeof(colors)/sizeof(colors[0]);

int main()
{
	process_dataset();
	return 0;
}

int process_frame(Mat left, Mat right, string name)
{
	int kernel_size = 3;
	cv::Mat smooth_left;
	cv::GaussianBlur(left, smooth_left, Size(kernel_size,kernel_size), 1.5);
	cv::Mat smooth_right;
	cv::GaussianBlur(right, smooth_right, Size(kernel_size, kernel_size), 1.5);

	int window_size = 9;
	int numDisparities = 96;
	int blockSize = 7;
	int P1 = 8 * 3 * std::pow(window_size, 2);
	int P2 = 32 * 3 * std::pow(window_size,2);
	int disp12MaxDiff = 1;
	int uniquenessRatio = 16;
	int speckleRange = 2;
	int mode = StereoSGBM::MODE_SGBM_3WAY;
	auto left_matcher = StereoSGBM::create(
	    numDisparities,
	    blockSize,
	    P1,
	    P2,
	    disp12MaxDiff,
	    uniquenessRatio,
	    speckleRange,
	    mode
	);

	auto right_matcher = ximgproc::createRightMatcher(left_matcher);

	auto wls_filter = ximgproc::createDisparityWLSFilter(left_matcher);
	wls_filter->setLambda(80000);
	wls_filter->setSigmaColor(1.2);

	cv::Mat disparity_left;
	left_matcher->compute(smooth_left, smooth_right, disparity_left);
	Mat disparity_right;
	right_matcher->compute(smooth_right, smooth_left, disparity_right);

	cv::Mat wls_image;
	wls_filter->filter(disparity_left, smooth_left, wls_image, disparity_right);
	normalize(wls_image, wls_image, 0, 255, NORM_MINMAX);

	cv::namedWindow("WLS Image");
	cv::imshow("WLS Image", disparity_left);
	cv::waitKey(0);
	create_combined_output(left, right, "/home/mitrankshahhh/Documents/Autonomous Systems/stereo_depth_estimator/Output.png");
}

int create_combined_output(Mat left, Mat right, string name)
{
	cv::Mat combined;
	hconcat(left, right, combined);
	std::cout<<"1st hconcat done."<<endl;
	cv::namedWindow("Disparity");
	cv::imshow("Disparity", combined);
	cv::waitKey(0);
}

bool startsWith(string mainStr, string toMatch)
{
    if(mainStr.find(toMatch) == 0)
		{
      return true;
		}
    else
		{
			return false;
		}
}

int process_dataset()
{
	string left_image_path = "/home/mitrankshahhh/Documents/Autonomous Systems/stereo_depth_estimator/img_L.png";
	string right_image_path = "/home/mitrankshahhh/Documents/Autonomous Systems/stereo_depth_estimator/img_R.png";
  auto left_image = imread(left_image_path, IMREAD_COLOR);
	cv::namedWindow("Left Image");
	cv::imshow("Left Image", left_image);
	cv::waitKey(0);
	auto right_image = imread(right_image_path, IMREAD_COLOR);
	cv::namedWindow("Right Image");
	cv::imshow("Right Image", right_image);
	cv::waitKey(0);
	process_frame(left_image, right_image, "left_image_sgbm");

	std::vector<std::string> class_names;
  {
      std::ifstream class_file("coco.names");
      if (!class_file)
      {
          std::cerr << "failed to open coco.names\n";
          return 0;
      }
      std::string line;
      while (std::getline(class_file, line))
          class_names.push_back(line);
  }

  auto net = cv::dnn::readNetFromDarknet("yolov4.cfg", "yolov4.weights");
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
  auto output_names = net.getUnconnectedOutLayersNames();

  cv::Mat frame, blob;
  frame = left_image;
  cv::namedWindow("Input");
  cv::imshow("Input Image", frame);
  cv::waitKey(0);
  std::vector<cv::Mat> detections;

  cv::dnn::blobFromImage(frame);
  auto total_start = std::chrono::steady_clock::now();
  cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(608, 608), cv::Scalar(), true, false, CV_32F);
  net.setInput(blob);

  auto dnn_start = std::chrono::steady_clock::now();
  net.forward(detections, output_names);
  auto dnn_end = std::chrono::steady_clock::now();

  std::vector<int> indices[NUM_CLASSES];
  std::vector<cv::Rect> boxes[NUM_CLASSES];
  std::vector<float> scores[NUM_CLASSES];

  for (auto& output : detections)
  {
      const auto num_boxes = output.rows;
      for (int i = 0; i < num_boxes; i++)
      {
          auto x = output.at<float>(i, 0) * frame.cols;
          auto y = output.at<float>(i, 1) * frame.rows;
          auto width = output.at<float>(i, 2) * frame.cols;
          auto height = output.at<float>(i, 3) * frame.rows;
          cv::Rect rect(x - width/2, y - height/2, width, height);

          for (int c = 0; c < NUM_CLASSES; c++)
          {
              auto confidence = *output.ptr<float>(i, 5 + c);
              if (confidence >= CONFIDENCE_THRESHOLD)
              {
                  boxes[c].push_back(rect);
                  scores[c].push_back(confidence);
              }
          }
      }
  }

  for (int c = 0; c < NUM_CLASSES; c++)
	{
		cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);
	}
	for (int c= 0; c < NUM_CLASSES; c++)
  {
    for (size_t i = 0; i < indices[c].size(); ++i)
    {
        const auto color = colors[c % NUM_COLORS];
        auto idx = indices[c][i];
        const auto& rect = boxes[c][idx];
        cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

        std::ostringstream label_ss;
        label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
        auto label = label_ss.str();

        int baseline;
        auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
        cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
    }
  }

  auto total_end = std::chrono::steady_clock::now();

  float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
  float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
  std::ostringstream stats_ss;
  stats_ss << std::fixed << std::setprecision(2);
  stats_ss << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
  auto stats = stats_ss.str();

  int baseline;
  auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
  cv::rectangle(frame, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
  cv::putText(frame, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

  cv::namedWindow("output");
  cv::imshow("output", frame);
  cv::waitKey(0);
}
