#include "OcvImageVisualizeUtil.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include "mLibCuda.h"

cv::Mat1b deviceFloatGrayToHostUcharGray(const float* d_gray, const int& width, const int& height) {
    cv::Mat1b uchar_image(height, width);
    cv::Mat1f float_image(height, width);
    MLIB_CUDA_SAFE_CALL(cudaMemcpy(float_image.ptr<float>(),d_gray,
                                       sizeof(float)*width*height,
                                       cudaMemcpyDeviceToHost));
    float_image *= 255;
    float_image.convertTo(uchar_image, CV_8UC1);
    return uchar_image;
}

cv::Mat3b colorizedDeviceFloatDepthImage(const float* d_depht, const int& width, const int& height) {
    cv::Mat3b colorized(height, width);
    colorized.setTo(cv::Vec3b(0,0,0));
    cv::Mat1f float_image(height, width);
    cv::Mat1s short_image(height, width);
    MLIB_CUDA_SAFE_CALL(cudaMemcpy(float_image.ptr<float>(),d_depht,
                                sizeof(float)*width*height,
                                   cudaMemcpyDeviceToHost));

//    float_image *= 1000;
//    float_image.convertTo(short_image, CV_16SC1);
    float min_d, max_d, scale;

    min_d = 100000.0f; max_d = -100000.0f;
    for (int i = 0; i < float_image.rows; i++) {
        for (int j = 0; j < float_image.cols; j++) {
            float d = float_image.at<float>(i,j);
            if (d > 0.05) {
                if (d < min_d) min_d = d;
                if (d > max_d) max_d = d;
            }
        }
    }

    scale = ((max_d - min_d) != 0) ? 1.0f / (max_d - min_d) : 1.0f / max_d;

    std::cout << min_d << " " <<max_d << " " << scale << std::endl;

    for (int i = 0; i < float_image.rows; i++) {
        for (int j = 0; j < float_image.cols; j++) {
            float sourceVal = float_image.at<float>(i,j);
            if (sourceVal < 0.05) continue;
            sourceVal = (sourceVal - min_d) * scale;

            colorized.at<cv::Vec3b>(i,j)[0] = (uchar)((1 - sourceVal) * 255.0f);
            colorized.at<cv::Vec3b>(i,j)[1] = (uchar)((1 - sourceVal) * 255.0f);
            colorized.at<cv::Vec3b>(i,j)[2] = (uchar)((sourceVal) * 255.0f);
        }
    }

    return colorized;
}