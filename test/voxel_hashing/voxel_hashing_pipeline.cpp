#include <iostream>
#include "voxel_hashing.h"
#include "GlobalAppState.h"
#include "GlobalBundlingState.h"
#include "RGBDSensor.h"
#include "SensorDataReader.h"
#include "OcvImageVisualizeUtil.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
    while (g_RGBDSensor->isReceivingFrames()) {
        std::cout << cnt ++ << std::endl;
        g_RGBDSensor->processDepth();

        vec4uc* color = new vec4uc;
        color = g_RGBDSensor->getColorRGBX();

        cv::Mat imageWrapper(height, width, CV_8UC4, const_cast<vec4uc* >(color));

        cv::cvtColor(imageWrapper, imageWrapper, cv::COLOR_RGBA2BGRA);
        cv::imshow("color", imageWrapper);
        cv::waitKey(2);





        mat4f T = g_RGBDSensor->getRigidTransform();
        std::cout << T << std::endl;

    }











    return 0;

}