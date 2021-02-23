#include <iostream>
#include <string>
#include <bits/stdc++.h>
#include <experimental/filesystem>
#include <assert.h>
#include <armadillo>
#include <matplotlibcpp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

using namespace std;
using namespace cv;
namespace plt = matplotlibcpp;
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

int main()
{
	//if (__name__ == __main__)
	//{
		process_dataset();
	//}
	return 0;
}

int process_frame(Mat left, Mat right, string name)
{
	int kernel_size = 3;
	cv::Mat smooth_left;
	GaussianBlur(left, smooth_left, Size(kernel_size,kernel_size), 1.5);
	cv::Mat smooth_right;
	GaussianBlur(right, smooth_right, Size(kernel_size, kernel_size), 1.5);

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

	Mat disparity_left;
	left_matcher->compute(smooth_left, smooth_right, disparity_left);
	Mat disparity_right;
	right_matcher->compute(smooth_right, smooth_left, disparity_right);

	Mat wls_image;
	wls_filter->filter(disparity_left, smooth_left, wls_image, disparity_right);
	normalize(wls_image, wls_image, 0, 255, NORM_MINMAX);

	plt::figure();
	// auto ax = plt::Axes(fig, [0., 0., 1., 1.]);
	// ax::set_axis_off();
	// fig::add_axes(ax);
	// plt::imshow(wls_image, "jet");
	// plt::savefig(DATASET_DISPARITIES+name);
	// plt::close();
	create_combined_output(left, right, name);
}

int create_combined_output(Mat left, Mat right, string name)
{
	cv::Mat combined;
	hconcat(left, right, combined);
	cv::Mat combined_image;
	hconcat(combined, imread(DATASET_DISPARITIES+name), combined_image);
	imwrite(DATASET_COMBINED+name, combined);
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
	string left_image_path = "/home/mitrankshahhh/Documents/DV/stereo_depth_estimator/img_L.png";
//[f for f in os.listdir(DATASET_LEFT) if not f.startswith('.')]
	// for (const auto & entry : fs::directory_iterator(DATASET_LEFT))
	// {
	// 	auto check_l1 = entry.DATASET_LEFT();
	// 	auto check_l2 = '.';
	// 	bool left_start = startsWith(check_l1, check_l2);
	// 	if (left_start == false)
	// 	{
	// 		left_images.push_back(entry.DATASET_LEFT());
	// 	}
	// }
	string right_image_path = "/home/mitrankshahhh/Documents/DV/stereo_depth_estimator/img_R.png";
//[f for f in os.listdir(DATASET_RIGHT) if not f.startswith('.')]
	// for (const auto & entry : fs::directory_iterator(path))
	// {
	// 	auto check_r1 = entry.DATASET_RIGHT();
	// 	auto check_r2 = '.';
	// 	bool right_start = startsWith(check_r1, check_r2);
	// 	if (right_start == false)
	// 	{
	// 		right_images.push_back(entry.DATASET_RIGHT());
	// 	}
	// }
	//assert(left_images.size() == right_images.size());
	// int l1 = sizeof(left_images)/sizeof(left_images[0]);
	// int r1 = sizeof(right_images)/sizeof(right_images[0]);
	// sort(left_images, left_images+l1);
	// sort(right_images, right_images+r1);
	// for (int i : left_images.size())
	// {
	// 	auto left_image_path = DATASET_LEFT+left_images[i];
	// 	auto right_image_path = DATASET_RIGHT+right_images[i];
		auto left_image = imread(left_image_path, IMREAD_COLOR);
		auto right_image = imread(right_image_path, IMREAD_COLOR);
		process_frame(left_image, right_image, "left_image_sgbm");
	//}
}
