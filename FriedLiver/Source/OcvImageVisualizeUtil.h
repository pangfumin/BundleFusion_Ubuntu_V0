#pragma once

#include <opencv2/core/core.hpp>

cv::Mat1b deviceFloatGrayToHostUcharGray(const float* d_gray, const int& width, const int& height);

cv::Mat3b colorizedDeviceFloatDepthImage(const float* d_depht, const int& width, const int& height);