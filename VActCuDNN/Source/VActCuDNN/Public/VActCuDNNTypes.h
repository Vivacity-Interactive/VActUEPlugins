#pragma once

THIRD_PARTY_INCLUDES_START
#include "cudnn.h"	
#include "cuda_runtime.h"
THIRD_PARTY_INCLUDES_END

#include "CoreMinimal.h"
#include "VActCuDNNTypes.generated.h"

UENUM()
enum class ECuAlgorithm
{
    ConvFwd,
    ConvBwdFilter,
    ConvBwdData,
    RNN,
    CTCLoss,
    // PPOFwd,
    // PPOBwdFilter,
    // PPOBwdData,
    // DyNN
};

UENUM()
enum class ECuResult
{
    ConvFwd = ECuAlgorithm::ConvFwd,
    ConvBwdData = ECuAlgorithm::ConvBwdData,
};

USTRUCT()
struct VACTCUDNN_API FCuContext
{
    GENERATED_BODY()
    
    _VACT_SIZE_T SizeOf;
    cudnnHandle_t Handle;
    void* Bytes;
};

USTRUCT()
struct VACTCUDNN_API FCuTensor
{
    GENERATED_BODY()

    _VACT_SIZE_T SizeOf;
    cudnnTensorDescriptor_t Desc;
    cudnnTensorFormat_t Format;
    cudnnDataType_t Type;
    void* Bytes;
};

USTRUCT()
struct VACTCUDNN_API FCuConvolution
{
    GENERATED_BODY()

    _VACT_SIZE_T SizeOf;
    cudnnConvolutionDescriptor_t Desc;
    cudnnConvolutionMode_t Mode;
    cudnnDataType_t Type;
    void* Bytes;
};

USTRUCT()
struct VACTCUDNN_API FCuKernel
{
    GENERATED_BODY()

    _VACT_SIZE_T SizeOf;
    cudnnFilterDescriptor_t Desc;
    cudnnTensorFormat_t Format;
    cudnnDataType_t Type;
    void* Bytes;
};

USTRUCT()
struct VACTCUDNN_API FCuAlgorithm
{
    GENERATED_BODY()

    _VACT_SIZE_T SizeOf;
    cudnnAlgorithmDescriptor_t Desc;
    ECuAlgorithm Type;
    union 
    {
        cudnnConvolutionFwdAlgo_t ConvFwd;
        cudnnConvolutionBwdFilterAlgo_t ConvBwdFilter;
        cudnnConvolutionBwdDataAlgo_t ConvBwdData;
        cudnnRNNAlgo_t ConvRNN;
        cudnnCTCLossAlgo_t ConvCTCLoss;
    };
    void* Bytes;
};

USTRUCT()
struct VACTCUDNN_API FCuResult
{
    GENERATED_BODY()
    
    _VACT_SIZE_T SizeOf;
    int32 Count;
    int32 MaxCount;
    ECuResult Type;
    union
    {
        cudnnConvolutionFwdAlgoPerf_t ConvFwd;
        cudnnConvolutionBwdDataAlgoPerf_t ConvBwdData;
        void* Bytes;
    };
};

USTRUCT()
struct VACTCUDNN_API FCuActivation
{
    GENERATED_BODY()
    float Alpha[1];
    float Beta[1];
    cudnnActivationDescriptor_t Desc;
    cudnnActivationMode_t Mode;
    cudnnNanPropagation_t Propagation;
};

USTRUCT()
struct VACTCUDNN_API FCuDevice
{
    GENERATED_BODY()

    int32 Id;
    int32 Index;
    cudaDeviceProp Specs;
};

USTRUCT()
struct VACTCUDNN_API FCuInstance
{
    GENERATED_BODY()

    _VACT_SIZE_T SizeOf;
    cudnnConvolutionDescriptor_t Desc;
    void* Bytes;
};

USTRUCT()
struct VACTCUDNN_API FCuCPU
{
    GENERATED_BODY()
    
    _VACT_SIZE_T SizeOf;
    void* Bytes;
};