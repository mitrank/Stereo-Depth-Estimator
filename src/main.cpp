#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <armadillo>
#include <matplotlibcpp.h>
#include <experimental/filesystem>
#include <assert.h>
#include <bits/stdc++.h>

using namespace std;
using namespace cv;
namespace plt = matplotlibcpp;
namespace fs = experimental/filesystem;

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

int process_frame(Mat left, Mat right, string name):
{
	int kernel_size = 3;
	auto smooth_left = GaussianBlur(left, Size(kernel_size,kernel_size), 1.5);
	auto smooth_right = GaussianBlur(right, Size(kernel_size, kernel_size), 1.5);

	int window_size = 9;
	auto left_matcher = StereoSGBM::create(
	    int numDisparities=96,
	    int blockSize=7,
	    int P1=8*3*window_size**2,
	    int P2=32*3*window_size**2,
	    int disp12MaxDiff=1,
	    int uniquenessRatio=16,
	    int speckleRange=2,
	    int mode=StereoSGBM::MODE_SGBM_3WAY
	);

	auto right_matcher = ximgproc::createRightMatcher(left_matcher);

	auto wls_filter = ximgproc::createDisparityWLSFilter(matcher_left=left_matcher);
	wls_filter::setLambda(80000);
	wls_filter::setSigmaColor(1.2);

	//disparity_left = int16(left_matcher.compute(smooth_left, smooth_right));
	//disparity_right = int16(right_matcher.compute(smooth_right, smooth_left));

	auto wls_image = wls_filter.filter(disparity_left, smooth_left, None, disparity_right);
	auto wls_image = normalize(src=wls_image, dst=wls_image, beta=0, alpha=255, norm_type=NORM_MINMAX);
	//wls_image = uint8(wls_image);

	auto fig = plt::figure(figsize=(wls_image.shape[1]/DPI, wls_image.shape[0]/DPI), dpi=DPI, frameon=False);
	auto ax = plt::Axes(fig, [0., 0., 1., 1.]);
	ax::set_axis_off();
	fig::add_axes(ax);
	plt::imshow(wls_image, 'jet');
	plt::savefig(DATASET_DISPARITIES+name);
	plt::close();
	create_combined_output(left, right, name);
}

int create_combined_output(Mat left, Mat right, string name):
{
	Mat combined = join_cols(left, right, imread(DATASET_DISPARITIES+name));
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
	string left_images = [];
//[f for f in os.listdir(DATASET_LEFT) if not f.startswith('.')]
	for (const auto & entry : fs::directory_iterator(DATASET_LEFT))
	{
		string check_l1 = entry.DATASET_LEFT();
		string check_l2 = '.';
		bool left_start = startsWith(check_l1, check_l2);
		if (left_start == false)
		{
			left_images.push_back(entry.DATASET_LEFT());
		}
	}
	string right_images = [];
//[f for f in os.listdir(DATASET_RIGHT) if not f.startswith('.')]
	for (const auto & entry : fs::directory_iterator(path))
	{
		string check_r1 = entry.DATASET_RIGHT();
		string check_r2 = '.';
		bool right_start = startsWith(check_r1, check_r2);
		if (right_start == false)
		{
			right_images.push_back(entry.DATASET_RIGHT());
		}
	}
	assert(left_images.size() == right_images.size());
	int l1 = sizeof(left_images)/sizeof(left_images[0]);
	int r1 = sizeof(right_images)/sizeof(right_images[0]);
	sort(left_images, left_images+l1);
	sort(right_images, right_images+r1);
	for (int i : left_images.size())
	{
		left_image_path = DATASET_LEFT+left_images[i];
		right_image_path = DATASET_RIGHT+right_images[i];
		left_image = imread(left_image_path, IMREAD_COLOR);
		right_image = imread(right_image_path, IMREAD_COLOR);
		process_frame(left_image, right_image, left_images[i]);
	}
}
