#pragma once

#define _CHECK_CUDNN(ExpresionIn)															\
{																							\
	cudnnStatus_t Status = (ExpresionIn);													\
	bool bFail = Status != CUDNN_STATUS_SUCCESS;											\
	if (bFail)																				\
	{																						\
		const FString ErrorMsg(cudnnGetErrorString(Status));								\
		UE_LOG(LogTemp, Fatal, TEXT("%s"), *ErrorMsg); 										\
	}																						\
}																							

#define _CHECK_CUDA(ExpresionIn)															\
{																							\
	cudaError_t Status = (ExpresionIn);														\
	bool bFail = Status != cudaSuccess;														\
	if (bFail)																				\
	{																						\
		const FString ErrorMsg(cudaGetErrorString(Status));									\
		UE_LOG(LogTemp, Fatal, TEXT("%s"), *ErrorMsg); 										\
	}																						\
}

#define _VACT_SIZE_T SIZE_T

#include "VActCuDNNTypes.h"

#include "CoreMinimal.h"
#include "VActCuDNN.generated.h"

struct FCuContext;
struct FCuTensor;
struct FCuConvolution;
struct FCuKernel;
struct FCuAlgorithm;
struct FCuResult;
//struct FCuWorkspace;
//struct FCuPooling;
//struct FCuLRN;
struct FCuActivation;
//struct FCuTransformer;
//struct FCuReduce;
//struct FCuLoss;
//struct FCuTransform;

struct FCuDevice;
struct FCuInstance;

USTRUCT()
struct FVActCuDNN
{
	GENERATED_BODY()

	static void _Unsafe_Create(FCuDevice& Into, int32 Index);

	static FCuDevice CreateDevice(int32 Index);

	static void _Unsafe_Create(TArray<FCuDevice>& Into);

	static TArray<FCuDevice> CreateDevice();

	static void _Unsafe_Create(FCuContext& Into);

	static FCuContext CreateContext();
	
	//static void _Unsafe_Create(FCuTensor& Into, int32 Width, int32 Size, int32 Count = 1);

	static void _Unsafe_Create(FCuTensor& Into, int32 Width, int32 Height, int32 Size, int32 Count = 1);
	
	static FCuTensor CreateTensor(int32 Width, int32 Height, int32 Size, int32 Count = 1);

	//static void _Unsafe_Create(FCuTensor& Into, int32 N, const TArray<int32> Dims, const TArray<int32> Strids);

	//static void _Unsafe_Create(FCuKernel& Into, int32 Width, int32 SizeOut, int32 SizeIn);
	
	static void _Unsafe_Create(FCuKernel& Into, int32 Width, int32 Height, int32 SizeOut, int32 SizeIn);
	
	static FCuKernel CreateKernel(int32 Width, int32 Height, int32 SizeOut, int32 SizeIn);

	//static void _Unsafe_Create(FCuKernel& Into, int32 N, const TArray<int32> Dims);

	//static void _Unsafe_Create(FCuConvolution& Into, int32 Pad = 1, int32 Stride = 1, int32 Dilation);

	static void _Unsafe_Create(FCuConvolution& Into, int32 PadWidth = 1, int32 PadHeight = 1, int32 StrideWidth = 1, int32 StrideHeight = 1, int32 DilationWidth = 1, int32 DilationHeight = 1);
	
	static FCuConvolution CreateConvolution(int32 PadWidth = 1, int32 PadHeight = 1, int32 StrideWidth = 1, int32 StrideHeight = 1, int32 DilationWidth = 1, int32 DilationHeight = 1);

	//static void _Unsafe_Create(FCuConvolution& Into, int32 N, const TArray<int32> Pads, const TArray<int32> Strids, const TArray<int32> Dilations);

	static void _Unsafe_Create(FCuAlgorithm& Into, FCuContext& Context, FCuTensor& Input, FCuKernel& Kernel, FCuConvolution& Convolution, FCuTensor& Output, FCuResult& Result, int32 Count = 1, bool bAllowShringking = true);
	
	static FCuAlgorithm CreateAlgorithm(FCuContext& Context, FCuTensor& Input, FCuKernel& Kernel, FCuConvolution& Convolution, FCuTensor& Output, FCuResult& Result, int32 LimitCount = 1, bool bAllowShringking = true);

	static void _Unsafe_Create(FCuActivation& Into, FCuTensor& Input, FCuTensor& Output, float Alpha = 1.0f, float Beta = 0.0f, double Gamma = 0.0);
	
	static FCuActivation CreateActivation(FCuTensor& Input, FCuTensor& Output, float Alpha = 1.0f, float Beta = 0.0f, double Gamma = 0.0);

	static void _Unsafe_Create(FCuInstance& Into, FCuContext& Context, FCuTensor& Input, FCuKernel& Kernel, FCuConvolution& Convolution, FCuTensor& Output, FCuAlgorithm& Algorithm, TArray<float> KernelData, int32 DefaultByte = 0);
	
	static FCuInstance CreateInstance(FCuContext& Context, FCuTensor& Input, FCuKernel& Kernel, FCuConvolution& Convolution, FCuTensor& Output, FCuAlgorithm& Algorithm, TArray<float> KernelData, int32 DefaultByte = 0);

	//static void _Unsafe_Create(FCuInstance& Into, FCuContext& Context, TArray<float> InputData, TArray<float> OutputData, FCuConvolution& Convolution, FCuAlgorithm& Algorithm, int32 Size, int32 Count = 1);
	
	static void _Unsafe_Use(const FCuDevice& Device);
	
	static void Use(const FCuDevice& Device);

	static void _Unsafe_Forward(const FCuContext& Context, FCuActivation& Activation, FCuTensor& Input, FCuTensor& Output);

	static void Forward(const FCuContext& Context, FCuActivation& Activation, FCuTensor& Input, FCuTensor& Output);

	// static void _Unsafe_Use(FCuInstance& Instance);
	
	// static FCuInstance& Use(FCuInstance& Instance);

	// static void _Unsafe_Train(FCuWorkspace& Into, FCuContext& Context);


	static void _Unsafe_Destroy(FCuKernel& Kernel, bool bData = true);

	static void Destroy(FCuKernel& Kernel, bool bData = true);

	static void _Unsafe_Destroy(FCuTensor& Tensor, bool bData = true);

	static void Destroy(FCuTensor& Tensor, bool bData = true);

	static void _Unsafe_Destroy(FCuConvolution& Convolution, bool bData = true);

	static void Destroy(FCuConvolution& Convolution, bool bData = true);

	static void _Unsafe_Destroy(FCuAlgorithm& Algorithm, bool bData = true);

	static void Destroy(FCuAlgorithm& Algorithm, bool bData = true);

	// static void _Unsafe_Destroy(FCuWorkspace& Workspace, bool bData = true);

	// static void Destroy(FCuWorkspace& Workspace, bool bData = true);

	static void _Unsafe_Destroy(FCuContext& Context, bool bData = true);

	static void Destroy(FCuContext& Context, bool bData = true);

	// static void _Unsafe_Destroy(FCuInstance& Instance, bool bData = true, bool bContext = false);

	// static void Destroy(FCuInstance& Instance, bool bData = true, bool bContext = false);

	static void _Unsafe_Destroy(FCuDevice& Device, bool bData = true);

	static void Destroy(FCuDevice& Device, bool bData = true);

	static void _Unsafe_Destroy(FCuResult& Result, bool bData = true);

	static void Destroy(FCuResult& Result, bool bData = true);

	static void _Unsafe_Destroy(FCuActivation& Activation, bool bData = true);

	static void Destroy(FCuActivation& Activation, bool bData = true);

	static void _Unsafe_CPU_ConvFwd(const FCuContext& Context, FCuActivation& Activation, FCuTensor& Input, FCuTensor& Output, FCuCPU& CPU);

	static void _Unsafe_CPU_RNN(const FCuContext& Context, FCuActivation& Activation, FCuTensor& Input, FCuTensor& Output, FCuCPU& CPU);

	static void _Unsafe_CPU_CTCLoss(const FCuContext& Context, FCuActivation& Activation, FCuTensor& Input, FCuTensor& Output, FCuCPU& CPU);

	//static void _Unsafe_CPU_PPOFwd(const FCuContext& Context, FCuActivation& Activation, FCuTensor& Input, FCuTensor& Output, FCuCPU& CPU);

	//static void _Unsafe_CPU_DyNN(const FCuContext& Context, FCuActivation& Activation, FCuTensor& Input, FCuTensor& Output, FCuCPU& CPU);

};