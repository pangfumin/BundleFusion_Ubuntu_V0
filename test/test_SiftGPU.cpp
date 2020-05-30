#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "CUDAImageUtil.h"
#include "OcvImageVisualizeUtil.h"
#include "SiftGPU.h"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/no



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


        float s_sensorDepthMax = 4.0f;	//maximum sensor depth in meter
        float s_sensorDepthMin = 0.1f;	//minimum sensor depth in meter
        sift_ = new SiftGPU;
        sift_->SetParams(width_, height_, false, 150, s_sensorDepthMin, s_sensorDepthMax);
        sift_->InitSiftGPU();

    }

    void runSift() {
//        cv::Mat1b uchar_gray_image = deviceFloatGrayToHostUcharGray(d_intensity_, width_, height_);
//        cv::Mat3b colored_depth = colorizedDeviceFloatDepthImage(d_depth_, width_, height_);
//        cv::imshow("colored_depth", uchar_gray_image);
//        cv::waitKey();

        int success = sift_->RunSIFT(d_intensity_, d_depth_);
        if (!success) throw MLIB_EXCEPTION("Error running SIFT detection");

        SIFTImageGPU cur_kp_des;
        MLIB_CUDA_SAFE_CALL(cudaMalloc(&cur_kp_des.d_keyPoints, sizeof(SIFTKeyPoint)*1024));
        MLIB_CUDA_SAFE_CALL(cudaMalloc(&cur_kp_des.d_keyPointDescs, sizeof(SIFTKeyPointDesc)*1024));

        unsigned int numKeypoints = sift_->GetKeyPointsAndDescriptorsCUDA(cur_kp_des, d_depth_, 1024);
        std::cout << "success: " << success << " " << numKeypoints << std::endl;


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

    //
    SiftGPU*				sift_;
    cv::Sif


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
    testSiftGpu.runSift();






    return 0;
}
