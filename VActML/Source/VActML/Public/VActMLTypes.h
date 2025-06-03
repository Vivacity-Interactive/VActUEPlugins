#pragma once

THIRD_PARTY_INCLUDES_START
#include "cudnn.h"	
#include "cuda_runtime.h"
THIRD_PARTY_INCLUDES_END

#include "CoreMinimal.h"
#include "VActMLTypes.generated.h"

UENUM()
enum class EMLAlgorithm
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
enum class EMLResult
{
    ConvFwd = EMLAlgorithm::ConvFwd,
    ConvBwdData = EMLAlgorithm::ConvBwdData,
};

USTRUCT()
struct VACTML_API FMLContext
{
    GENERATED_BODY()
    
    _VACT_SIZE_T SizeOf;
    cudnnHandle_t Handle;
    void* Bytes;
};

USTRUCT()
struct VACTML_API FMLTensor
{
    GENERATED_BODY()

    _VACT_SIZE_T SizeOf;
    cudnnTensorDescriptor_t Desc;
    cudnnTensorFormat_t Format;
    cudnnDataType_t Type;
    void* Bytes;
};

USTRUCT()
struct VACTML_API FMLConvolution
{
    GENERATED_BODY()

    _VACT_SIZE_T SizeOf;
    cudnnConvolutionDescriptor_t Desc;
    cudnnConvolutionMode_t Mode;
    cudnnDataType_t Type;
    void* Bytes;
};

USTRUCT()
struct VACTML_API FMLKernel
{
    GENERATED_BODY()

    _VACT_SIZE_T SizeOf;
    cudnnFilterDescriptor_t Desc;
    cudnnTensorFormat_t Format;
    cudnnDataType_t Type;
    void* Bytes;
};

USTRUCT()
struct VACTML_API FMLAlgorithm
{
    GENERATED_BODY()

    _VACT_SIZE_T SizeOf;
    cudnnAlgorithmDescriptor_t Desc;
    EMLAlgorithm Type;
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
struct VACTML_API FMLResult
{
    GENERATED_BODY()
    
    _VACT_SIZE_T SizeOf;
    int32 Count;
    int32 MaxCount;
    EMLResult Type;
    union
    {
        cudnnConvolutionFwdAlgoPerf_t ConvFwd;
        cudnnConvolutionBwdDataAlgoPerf_t ConvBwdData;
        void* Bytes;
    };
};

USTRUCT()
struct VACTML_API FMLActivation
{
    GENERATED_BODY()
    float Alpha[1];
    float Beta[1];
    cudnnActivationDescriptor_t Desc;
    cudnnActivationMode_t Mode;
    cudnnNanPropagation_t Propagation;
};

USTRUCT()
struct VACTML_API FMLDevice
{
    GENERATED_BODY()

    int32 Id;
    int32 Index;
    cudaDeviceProp Specs;
};

USTRUCT()
struct VACTML_API FMLInstance
{
    GENERATED_BODY()

    _VACT_SIZE_T SizeOf;
    cudnnConvolutionDescriptor_t Desc;
    void* Bytes;
};

USTRUCT()
struct VACTML_API FMLCPU
{
    GENERATED_BODY()
    
    _VACT_SIZE_T SizeOf;
    void* Bytes;
};

UENUM()
enum class EMLFormatSafetensor
{
    Default = 0,
    JSon = 0,
    Binary = 1,
    VActBinary = 2
};

UENUM()
enum class EMLFormatSafetensorDType
{
    F64 = CUDNN_DATA_DOUBLE,
    F32 = CUDNN_DATA_FLOAT,
    F16 = CUDNN_DATA_HALF,
    BF16 = CUDNN_DATA_BFLOAT16,
    I64 = CUDNN_DATA_INT64,
    I32 = CUDNN_DATA_INT32,
    I16 = CUDNN_DATA_INT32,
    I8 = CUDNN_DATA_INT8,
    U8 = CUDNN_DATA_UINT8,
    BOOL = CUDNN_DATA_BOOLEAN
};

//         {N}    [{DType   , {K     , Shape   }, {Begin,  End   }}] >> [{Key     , Value   }] >> Body Bytes
//Header {uint64} [{[char]\0, {uint32, [uint32]}, {uint64, uint64}}] >> [{[char]\0, [char]\0}] >> Body [uint8]

//         {N}    "{ Tensor : {    DType      ,      Shape    ,      Offset          , ...,        MetaData { Key   : Value   } }" >> Body Bytes
//Header {uint64} "{ string : {"dtype": string, "shape": [int], "offset": [int, int]}, ..., "__metadata__": { string: string } }" >> Body [uint8]
USTRUCT()
struct VACTML_API FMLFormatSafetensorHeader
{
    GENERATED_BODY()

    TArray<FMLTensor> Tensors;
    TMap<FString, FString> Meta;
};

//         {N}    [{DType,  {K     , Shape   }, {Begin,  Count }}] >> [{Key     , Value   }] >> Body Bytes
//Header {uint64} [{uint32, {uint32, [uint32]}, {uint64, uint64}}] >> [{[char]\0, [char]\0}] >> Body [uint8]
