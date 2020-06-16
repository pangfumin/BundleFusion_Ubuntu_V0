#ifndef  __RGB_IMAGE_FRAME_H__
#define  __RGB_IMAGE_FRAME_H__

#include "RGBDSensor.h"
#include "CUDAImageUtil.h"
#include "CUDAImageCalibrator.h"
#include "GlobalBundlingState.h"
#include "TimingLog.h"

#include <cuda_runtime.h>

class RgbImageFrame {
public:
    RgbImageFrame(unsigned int width, unsigned int height,
                  const uchar4 *rgb_image, const float* depth_image,
            bool isOnCPU, bool isOnGPU)
    {

        s_width = width;
        s_height = height;
        s_bIsOnGPU = isOnGPU;
        s_bIsOnCPU = isOnCPU;

        if (s_bIsOnGPU) {
            MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_dev_depth, sizeof(float)*width*height));
            MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_dev_color, sizeof(uchar4)*width*height));

            MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_dev_depth, depth_image, sizeof(float)*width*height,  cudaMemcpyHostToDevice));
            MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_dev_color, rgb_image, sizeof(uchar4)*width*height,  cudaMemcpyHostToDevice));


        }

        if(s_bIsOnCPU) {
            m_host_depth = new float[width*height];
            m_host_color = new uchar4[width*height];

            memcpy(m_host_depth, const_cast<float*>(depth_image), sizeof(float)*width*height);
            memcpy(m_host_color, rgb_image, sizeof(uchar4)*width*height);
        }
    }

    ~RgbImageFrame() {

        if (s_bIsOnGPU) {
            MLIB_CUDA_SAFE_FREE(m_dev_depth);
            MLIB_CUDA_SAFE_FREE(m_dev_color);
        }

        if(s_bIsOnCPU) {
            SAFE_DELETE_ARRAY(m_host_color);
            SAFE_DELETE_ARRAY(m_host_depth);
        }


    }
//    void globalFree()
//    {
//        if (!s_bIsOnGPU) {
//            MLIB_CUDA_SAFE_FREE(s_depthIntegrationGlobal);
//            MLIB_CUDA_SAFE_FREE(s_colorIntegrationGlobal);
//        }
//        else {
//            SAFE_DELETE_ARRAY(s_depthIntegrationGlobal);
//            SAFE_DELETE_ARRAY(s_colorIntegrationGlobal);
//        }
//    }
//
//
//    void alloc() {
//        if (s_bIsOnGPU) {
//            MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_depthIntegration, sizeof(float)*s_width*s_height));
//            MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_colorIntegration, sizeof(uchar4)*s_width*s_height));
//        }
//        else {
//            m_depthIntegration = new float[s_width*s_height];
//            m_colorIntegration = new uchar4[s_width*s_height];
//        }
//    }
//
//
//    void free() {
//        if (s_bIsOnGPU) {
//            MLIB_CUDA_SAFE_FREE(m_depthIntegration);
//            MLIB_CUDA_SAFE_FREE(m_colorIntegration);
//        }
//        else {
//            SAFE_DELETE_ARRAY(m_depthIntegration);
//            SAFE_DELETE_ARRAY(m_colorIntegration);
//        }
//    }
//
//
    const float* getDepthFrameGPU() {	//be aware that only one depth frame is globally valid at a time
        if (s_bIsOnGPU) {
            return m_dev_depth;
        }
        else {
             return NULL;
        }
    }
    const uchar4* getColorFrameGPU() {	//be aware that only one depth frame is globally valid at a time
        if (s_bIsOnGPU) {
            return m_dev_color;
        }
        else {
            return NULL;
        }
    }
//
    const float* getDepthFrameCPU() {
        if (s_bIsOnCPU) {
               return m_host_depth;
        }
        else {
           return NULL;
        }
    }
    const uchar4* getColorFrameCPU() {
        if (s_bIsOnCPU) {
              return m_host_color;
        }
        else {
            return NULL;
        }
    }

private:
    float*	m_dev_depth;
    uchar4*	m_dev_color;

    float* m_host_depth;
    uchar4*	m_host_color;

    bool			s_bIsOnGPU;
    bool			s_bIsOnCPU;
    unsigned int s_width;
    unsigned int s_height;

//    float*		s_depthIntegrationGlobal;
//    uchar4*		s_colorIntegrationGlobal;
////    ManagedRGBDInputFrame*	s_activeColorGPU;
////    ManagedRGBDInputFrame*	s_activeDepthGPU;
//
//    float*		s_depthIntegrationGlobalCPU;
//    uchar4*		s_colorIntegrationGlobalCPU;
////    ManagedRGBDInputFrame*	s_activeColorCPU;
////    ManagedRGBDInputFrame*	s_activeDepthCPU;

};
#endif //   __RGB_IMAGE_FRAME_H__