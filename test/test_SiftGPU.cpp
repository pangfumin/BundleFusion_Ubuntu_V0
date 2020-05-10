#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "SiftGPU.h"

int main () {
    std::string image_file = "/home/pang/software/BundleFusion_Ubuntu_V0/test/img1.png";

//    SiftGPU sift;
//    char* myargv[4] ={ "-fo", "-1", "-v", "1"};
//    sift.ParseParam(4, myargv);

    SiftGPU*				m_sift;
    m_sift = new SiftGPU;
    float s_sensorDepthMax = 4.0f;	//maximum sensor depth in meter
    float s_sensorDepthMin = 0.1f;	//minimum sensor depth in meter

    int widthSift = 1920, heightSift = 1080;
    m_sift->SetParams(widthSift, heightSift, false, 150, s_sensorDepthMin, s_sensorDepthMax);
    m_sift->InitSiftGPU();



    return 0;
}
