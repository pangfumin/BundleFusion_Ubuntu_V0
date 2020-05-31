#ifndef __VOXEL_HASHING_H__
#define __VOXEL_HASHING_H__
//#include ""
#include "GlobalAppState.h"
#include "GlobalBundlingState.h"
#include "RGBDSensor.h"
#include "SensorDataReader.h"
#include "OcvImageVisualizeUtil.h"
#include "CUDAImageManager.h"

#include "DepthSensing/CUDASceneRepHashSDF.h"
#include "DepthSensing/CUDARayCastSDF.h"
#include "DepthSensing/CUDAMarchingCubesHashSDF.h"
#include "DepthSensing/CUDAHistogramHashSDF.h"
#include "DepthSensing/CUDASceneRepChunkGrid.h"

class VoxelHashingPipeline {
public:
    VoxelHashingPipeline(RGBDSensor* rgbdSensor, CUDAImageManager* cudaImageManager);

    void process ();


    void StopScanningAndExtractIsoSurfaceMC(const std::string& filename, bool overwriteExistingFile /*= false*/);

    void renderToCvImage(const mat4f& transform, cv::Mat& image);

public:
    void integrate(const DepthCameraData& depthCameraData, const mat4f& transformation);

    CUDAImageManager*			g_CudaImageManager = NULL;
    RGBDSensor*					g_depthSensingRGBDSensor = NULL;
    CUDASceneRepHashSDF*		g_sceneRep = NULL;
    CUDARayCastSDF*				g_rayCast = NULL;
    CUDAMarchingCubesHashSDF*	g_marchingCubesHashSDF = NULL;
    CUDAHistrogramHashSDF*		g_historgram = NULL;
    CUDASceneRepChunkGrid*		g_chunkGrid = NULL;

    DepthCameraParams			g_depthCameraParams;

};

#endif