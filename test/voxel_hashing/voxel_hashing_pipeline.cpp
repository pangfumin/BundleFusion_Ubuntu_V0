#include <iostream>
#include "voxel_hashing.h"
#include "GlobalAppState.h"
#include "GlobalBundlingState.h"
#include "RGBDSensor.h"
#include "SensorDataReader.h"
#include "OcvImageVisualizeUtil.h"
#include "CUDAImageManager.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "voxel_hashing.h"

int main() {

    std::string fileNameDescGlobalApp;
    std::string fileNameDescGlobalBundling;

    std::cout << "usage: DepthSensing [fileNameDescGlobalApp] [fileNameDescGlobalTracking]" << std::endl;
    fileNameDescGlobalApp = "zParametersDefault.txt";
    fileNameDescGlobalBundling = "zParametersBundlingDefault.txt";

    std::cout << VAR_NAME(fileNameDescGlobalApp) << " = " << fileNameDescGlobalApp << std::endl;
    std::cout << VAR_NAME(fileNameDescGlobalBundling) << " = " << fileNameDescGlobalBundling << std::endl;
    std::cout << std::endl;

    ParameterFile parameterFileGlobalApp(fileNameDescGlobalApp);

    GlobalAppState::getInstance().readMembers(parameterFileGlobalApp);

    GlobalAppState::getInstance().readMembers(parameterFileGlobalApp);

    //Read the global camera tracking state
    ParameterFile parameterFileGlobalBundling(fileNameDescGlobalBundling);

    GlobalBundlingState::getInstance().readMembers(parameterFileGlobalBundling);


    //Read the global camera tracking state
//    ParameterFile parameterFileGlobalBundling(fileNameDescGlobalBundling);

    RGBDSensor* g_RGBDSensor = NULL;
    g_RGBDSensor = new SensorDataReader;
    g_RGBDSensor->createFirstConnected();

    int width = g_RGBDSensor->getDepthWidth();
    int height = g_RGBDSensor->getDepthHeight();
    int cnt = 0;

    CUDAImageManager* g_imageManager = new CUDAImageManager(GlobalAppState::get().s_integrationWidth, GlobalAppState::get().s_integrationHeight,
                                          GlobalBundlingState::get().s_widthSIFT, GlobalBundlingState::get().s_heightSIFT,
                                          g_RGBDSensor, true);

    VoxelHashingPipeline voxelHashingPipeline(g_RGBDSensor, g_imageManager);

    while (g_RGBDSensor->isReceivingFrames()) {


        voxelHashingPipeline.process();



        int width = g_RGBDSensor->getDepthWidth();
        int height = g_RGBDSensor->getDepthHeight();

        auto frame = g_imageManager->getLastIntegrateFrame();
        auto* color_cpu = frame.getColorFrameCPU();
        cv::Mat imageWrapper(height, width, CV_8UC4, const_cast<uchar4* >(color_cpu));
        //
        //            std::cout << "frame: " << g_imageManager->getCurrFrameNumber() << std::endl;

        std::cout << "integrate " << std::endl;
        cv::cvtColor(imageWrapper, imageWrapper, cv::COLOR_RGBA2BGRA);
        cv::imshow("color", imageWrapper);
        int c = cv::waitKey();
        if (c == 27) {
            break;
        }


    }


    std::string filename = "/home/pang/test.ply";
    bool overwrite = true;
    voxelHashingPipeline.StopScanningAndExtractIsoSurfaceMC(filename, overwrite);








    return 0;

}