/**
 * @file moai4ss.hpp.h
 * @brief Descriptions(TODO.(Jamin. )) ...
 * @author Jamin. (metoak), junming.chen@metoak.net
 * @version 0.0.1
 * @date 05/25/2022 15:42:57
 * @copyright Copyright © 2022 metoak(Beijing), LLC. all rights reserved.
 */
#ifndef MOAI4SS_HPP_65gPuBI
#define MOAI4SS_HPP_65gPuBI
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <moai.hpp>
#include <string>
#include <mocppbase.hpp>

#if (MO_NET_USE_DPU)
#include <glog/logging.h>
#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <vitis/ai/collection_helper.hpp>
#include <xir/graph/graph.hpp>
#else
#include <ncnn/layer.h>
#include <ncnn/net.h>
#endif

#define MOAI4SS_LABEL_NUM 18
extern std::string gAI4SSLabel[MOAI4SS_LABEL_NUM];
extern cv::Vec3b gAI4SSLabelColor[MOAI4SS_LABEL_NUM];

class MOAI4SS : public MOAI, public MOIniBase {
    /// virtual member-function from MOAI
   public:
    virtual void init_param();
    virtual int32_t infprocesspre(const uint8_t *pData, int32_t width, int32_t height, int32_t channel, int32_t type);
    virtual int32_t infprocessing();
    virtual int32_t infprocesspost();

   private:
#if (MO_NET_USE_DPU)
    // DpuRunner实例化
    std::unique_ptr<vart::RunnerExt> ptrCssRunner;

    // add more of your self-define members
    uint8_t *pLabelMapper;

    // DpuRunner support
    std::unique_ptr<xir::Graph> ptrCssGraph;

    // 输入特征图的属性
    uint16_t inputHeight;
    uint16_t inputWidth;
    uint16_t inputChannel;

    // 输出特征图的属性
    uint16_t outputHeight;
    uint16_t outputWidth;
    uint16_t outputChannel;

    uint32_t nWeightVersion;

    char sPathModelParam[256];

    // 输入输出特征图缓冲区
    std::vector<vart::TensorBuffer *> aInputTensorBuffers;
    std::vector<vart::TensorBuffer *> vOutputTensorBuffers;

    // 获得输入定点缩放因子
    const xir::Tensor *input_tensor;
    float input_scale;

    // 获得 output_scale
    const xir::Tensor *output_tensor;
    float output_scale;
    int8_t *pAIOutputData;

    void post_process_impl();

    bool bLogTurnOn;
    bool bLogPrint;
    bool bLogWrite;
    char sLogFilePath[256];

#else
    typedef struct _mo_yolo_scale_params_ {
        float r;
        int dw;
        int dh;
        int new_unpad_w;
        int new_unpad_h;
        bool flag;
    } YOLOScaleParams;

    char sPathModelParam[256];
    char sPathModelBin[256];
    bool bUseGpu;
    int32_t nModeType;
    ncnn::Net Net;  // the network model for inference
    ncnn::Mat mInput;
    ncnn::Mat mOutput;
    YOLOScaleParams mpScaleParam;
    bool moLoadAiModel();  // load the model config and params
    void resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs, int target_height, int target_width,
                        YOLOScaleParams &scale_params);
    int32_t moForwardInference(const cv::Mat &mSrcImage, cv::Mat &mOutImage);

#endif
   public:
    MOAI4SS();

    ~MOAI4SS();
};
#endif /** end of MOAI4SS_HPP_65gPuBI define **/
