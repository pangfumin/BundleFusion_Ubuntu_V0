#include "voxel_hashing.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

VoxelHashingPipeline::VoxelHashingPipeline(RGBDSensor* rgbdSensor, CUDAImageManager* cudaImageManager):
g_CudaImageManager(cudaImageManager), g_depthSensingRGBDSensor(rgbdSensor){


    g_sceneRep = new CUDASceneRepHashSDF(CUDASceneRepHashSDF::parametersFromGlobalAppState(GlobalAppState::get()));
    //g_rayCast = new CUDARayCastSDF(CUDARayCastSDF::parametersFromGlobalAppState(GlobalAppState::get(), g_CudaImageManager->getColorIntrinsics(), g_CudaImageManager->getColorIntrinsicsInv()));
    g_rayCast = new CUDARayCastSDF(CUDARayCastSDF::parametersFromGlobalAppState(GlobalAppState::get(), g_CudaImageManager->getDepthIntrinsics(), g_CudaImageManager->getDepthIntrinsicsInv()));


    g_marchingCubesHashSDF = new CUDAMarchingCubesHashSDF(CUDAMarchingCubesHashSDF::parametersFromGlobalAppState(GlobalAppState::get()));
    g_historgram = new CUDAHistrogramHashSDF(g_sceneRep->getHashParams());

    g_depthCameraParams.fx = g_CudaImageManager->getDepthIntrinsics()(0, 0);//TODO check intrinsics
    g_depthCameraParams.fy = g_CudaImageManager->getDepthIntrinsics()(1, 1);
    g_depthCameraParams.mx = g_CudaImageManager->getDepthIntrinsics()(0, 2);
    g_depthCameraParams.my = g_CudaImageManager->getDepthIntrinsics()(1, 2);
    g_depthCameraParams.m_sensorDepthWorldMin = GlobalAppState::get().s_renderDepthMin;
    g_depthCameraParams.m_sensorDepthWorldMax = GlobalAppState::get().s_renderDepthMax;
    g_depthCameraParams.m_imageWidth = g_CudaImageManager->getIntegrationWidth();
    g_depthCameraParams.m_imageHeight = g_CudaImageManager->getIntegrationHeight();
    DepthCameraData::updateParams(g_depthCameraParams);


}

void VoxelHashingPipeline::process () {
    g_CudaImageManager->process();

    DepthCameraData depthCameraData(g_CudaImageManager->getLastIntegrateFrame().getDepthFrameGPU(), g_CudaImageManager->getLastIntegrateFrame().getColorFrameGPU());

    mat4f transformation = mat4f::identity();
    integrate(depthCameraData, transformation);
}



void VoxelHashingPipeline::integrate(const DepthCameraData& depthCameraData, const mat4f& transformation)
{
    mat4f g_transformWorld = mat4f::identity();

//    if (GlobalAppState::get().s_streamingEnabled) {
//        vec4f posWorld = transformation*vec4f(GlobalAppState::getInstance().s_streamingPos, 1.0f); // trans laggs one frame *trans
//        vec3f p(posWorld.x, posWorld.y, posWorld.z);
//
//        g_chunkGrid->streamOutToCPUPass0GPU(p, GlobalAppState::get().s_streamingRadius, true, true);
//        g_chunkGrid->streamInToGPUPass1GPU(true);
//    }

    if (GlobalAppState::get().s_integrationEnabled) {
        unsigned int* d_bitMask = NULL;
        if (g_chunkGrid) d_bitMask = g_chunkGrid->getBitMaskGPU();

        g_sceneRep->integrate(g_transformWorld * transformation, depthCameraData, g_depthCameraParams, d_bitMask);//here is the problem

//        g_sceneRep->debugHash();
    }
    //else {
    //	//compactification is required for the ray cast splatting
    //	g_sceneRep->setLastRigidTransformAndCompactify(transformation);	//TODO check this
    //}
}

void VoxelHashingPipeline::StopScanningAndExtractIsoSurfaceMC(const std::string& filename, bool overwriteExistingFile /*= false*/)
{
    //g_sceneRep->debugHash();
    //g_chunkGrid->debugCheckForDuplicates();


    Timer t;


    g_marchingCubesHashSDF->clearMeshBuffer();
    if (!GlobalAppState::get().s_streamingEnabled) {

        //g_chunkGrid->stopMultiThreading();
        //g_chunkGrid->streamInToGPUAll();
        g_marchingCubesHashSDF->extractIsoSurface(g_sceneRep->getHashData(), g_sceneRep->getHashParams(), g_rayCast->getRayCastData());
        //g_chunkGrid->startMultiThreading();
    }


    const mat4f& rigidTransform = mat4f::identity();//g_lastRigidTransform
    g_marchingCubesHashSDF->saveMesh(filename, &rigidTransform, overwriteExistingFile);

    std::cout << "Mesh generation time " << t.getElapsedTime() << " seconds" << std::endl;

    //g_sceneRep->debugHash();
    //g_chunkGrid->debugCheckForDuplicates();
}
