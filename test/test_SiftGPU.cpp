#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "CUDAImageUtil.h"
#include "SiftGPU.h"


class TestSiftGPU {
public:
    TestSiftGPU(const cv::Mat4b& rgb_image, const cv::Mat1s& depth_image):
            width_(rgb_image.cols), height_(rgb_image.rows)
    {

        cv::Mat1f depth_image_float;
        depth_image.convertTo(depth_image_float, CV_32FC1, 1/1000.0);


        // malloc
        MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_rgb_, sizeof(uchar4)*width_*height_));
        MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_intensity_, sizeof(float)*width_*height_));
        MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depth_, sizeof(float)*width_*height_));
        // transfer
        MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_rgb_, rgb_image.ptr<cv::Vec4b>(),
                                       sizeof(uchar4)*width_*height_, cudaMemcpyHostToDevice));

        MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_depth_, depth_image_float.ptr<float>(),
                                       sizeof(float)*width_*height_, cudaMemcpyHostToDevice));

        // rgb to gray
        CUDAImageUtil::resampleToIntensity(d_intensity_, width_, height_,
                                           d_rgb_, width_, height_);

        /// DEBUG
        // transfer back
//        cv::Mat1f intensity_float(height_, width_);
//        MLIB_CUDA_SAFE_CALL(cudaMemcpy(intensity_float.ptr<float>(),d_intensity_,
//                                       sizeof(float)*width_*height_,
//                                       cudaMemcpyDeviceToHost));
//
//
//        intensity_float *= 255;
//        cv::Mat1b intensity_uchar(height_, width_);
//        intensity_float.convertTo(intensity_uchar, CV_8UC1);
//
//        cv::imshow("rgba_image", intensity_uchar);
//        cv::waitKey();

    }

    void runSift() {

    }



    ~TestSiftGPU() {
        MLIB_CUDA_SAFE_CALL(cudaFree(d_rgb_));
        MLIB_CUDA_SAFE_CALL(cudaFree(d_intensity_));
        MLIB_CUDA_SAFE_CALL(cudaFree(d_depth_));
    };

private:
    int width_, height_;

    // device
    uchar4*					d_rgb_;
    float*					d_intensity_;
    float*                  d_depth_;

};


int main () {
    std::string rgb_image_file = "/home/pang/dataset/bundlefusion/copyroom/frame-000000.color.jpg";
    std::string depth_image_file = "/home/pang/dataset/bundlefusion/copyroom/frame-000000.depth.png";


    cv::Mat rgb_image = cv::imread(rgb_image_file, CV_LOAD_IMAGE_COLOR);
    cv::Mat depth_image = cv::imread(depth_image_file, CV_LOAD_IMAGE_UNCHANGED);

    cv::cvtColor(rgb_image, rgb_image, CV_RGB2BGR);
    cv::Mat rgba_image;
    cv::cvtColor(rgb_image, rgba_image, CV_RGB2BGRA);




    TestSiftGPU testSiftGpu(rgba_image, depth_image);






    return 0;
}
