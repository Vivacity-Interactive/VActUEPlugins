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

#include "VActMLTypes.h"

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"
#include "VActML.generated.h"

struct FMLContext;
struct FMLTensor;
struct FMLConvolution;
struct FMLKernel;
struct FMLAlgorithm;
struct FMLResult;
//struct FMLWorkspace;
//struct FMLPooling;
//struct FMLLRN;
struct FMLActivation;
//struct FMLTransformer;
//struct FMLReduce;
//struct FMLLoss;
//struct FMLTransform;

struct FMLDevice;
struct FMLInstance;


class FVActMLModule : public IModuleInterface
{
public:

	/** IModuleInterface implementation */
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;

private:
	/** Handle to the test dll we will load */
	void* CudaLib;

	void* CuDNNLib;
	void* CuDNNOpsTrainLib;
	void* CuDNNOpsInferLib;
	void* CuDNNCNNTrainLib;
	void* CuDNNCNNInferLib;
	void* CuDNNAdvInferLib;
	void* CuDNNAdvTrainLib;
};

USTRUCT()
struct FVActML
{
	GENERATED_BODY()

#if WITH_EDITORONLY_DATA
	static void _Debug_Cuda_Test();

	static void _Debug_CuDNN_Test_0();

	static void _Debug_CuDNN_Test_1();

	static void _Debug_VActML_Test_0();

	static void _Debug_VActML_Test_1();
#endif;

	static void _Unsafe_Create(FMLDevice& Into, int32 Index);

	static FMLDevice CreateDevice(int32 Index);

	static void _Unsafe_Create(TArray<FMLDevice>& Into);

	static TArray<FMLDevice> CreateDevice();

	static void _Unsafe_Create(FMLContext& Into);

	static FMLContext CreateContext();
	
	//static void _Unsafe_Create(FMLTensor& Into, int32 Width, int32 Size, int32 Count = 1);

	static void _Unsafe_Create(FMLTensor& Into, int32 Width, int32 Height, int32 Size, int32 Count = 1);
	
	static FMLTensor CreateTensor(int32 Width, int32 Height, int32 Size, int32 Count = 1);

	//static void _Unsafe_Create(FMLTensor& Into, int32 N, const TArray<int32> Dims, const TArray<int32> Strids);

	//static void _Unsafe_Create(FMLKernel& Into, int32 Width, int32 SizeOut, int32 SizeIn);
	
	static void _Unsafe_Create(FMLKernel& Into, int32 Width, int32 Height, int32 SizeOut, int32 SizeIn);
	
	static FMLKernel CreateKernel(int32 Width, int32 Height, int32 SizeOut, int32 SizeIn);

	//static void _Unsafe_Create(FMLKernel& Into, int32 N, const TArray<int32> Dims);

	//static void _Unsafe_Create(FMLConvolution& Into, int32 Pad = 1, int32 Stride = 1, int32 Dilation);

	static void _Unsafe_Create(FMLConvolution& Into, int32 PadWidth = 1, int32 PadHeight = 1, int32 StrideWidth = 1, int32 StrideHeight = 1, int32 DilationWidth = 1, int32 DilationHeight = 1);
	
	static FMLConvolution CreateConvolution(int32 PadWidth = 1, int32 PadHeight = 1, int32 StrideWidth = 1, int32 StrideHeight = 1, int32 DilationWidth = 1, int32 DilationHeight = 1);

	//static void _Unsafe_Create(FMLConvolution& Into, int32 N, const TArray<int32> Pads, const TArray<int32> Strids, const TArray<int32> Dilations);

	static void _Unsafe_Create(FMLAlgorithm& Into, FMLContext& Context, FMLTensor& Input, FMLKernel& Kernel, FMLConvolution& Convolution, FMLTensor& Output, FMLResult& Result, int32 Count = 1, bool bAllowShringking = true);
	
	static FMLAlgorithm CreateAlgorithm(FMLContext& Context, FMLTensor& Input, FMLKernel& Kernel, FMLConvolution& Convolution, FMLTensor& Output, FMLResult& Result, int32 LimitCount = 1, bool bAllowShringking = true);

	static void _Unsafe_Create(FMLActivation& Into, FMLTensor& Input, FMLTensor& Output, float Alpha = 1.0f, float Beta = 0.0f, double Gamma = 0.0);
	
	static FMLActivation CreateActivation(FMLTensor& Input, FMLTensor& Output, float Alpha = 1.0f, float Beta = 0.0f, double Gamma = 0.0);

	static void _Unsafe_Create(FMLInstance& Into, FMLContext& Context, FMLTensor& Input, FMLKernel& Kernel, FMLConvolution& Convolution, FMLTensor& Output, FMLAlgorithm& Algorithm, TArray<float> KernelData, int32 DefaultByte = 0);
	
	static FMLInstance CreateInstance(FMLContext& Context, FMLTensor& Input, FMLKernel& Kernel, FMLConvolution& Convolution, FMLTensor& Output, FMLAlgorithm& Algorithm, TArray<float> KernelData, int32 DefaultByte = 0);

	//static void _Unsafe_Create(FMLInstance& Into, FMLContext& Context, TArray<float> InputData, TArray<float> OutputData, FMLConvolution& Convolution, FMLAlgorithm& Algorithm, int32 Size, int32 Count = 1);
	
	static void _Unsafe_Use(const FMLDevice& Device);
	
	static void Use(const FMLDevice& Device);

	static void _Unsafe_Forward(const FMLContext& Context, FMLActivation& Activation, FMLTensor& Input, FMLTensor& Output);

	static void Forward(const FMLContext& Context, FMLActivation& Activation, FMLTensor& Input, FMLTensor& Output);

	// static void _Unsafe_Use(FMLInstance& Instance);
	
	// static FMLInstance& Use(FMLInstance& Instance);

	// static void _Unsafe_Train(FMLWorkspace& Into, FMLContext& Context);


	static void _Unsafe_Destroy(FMLKernel& Kernel, bool bData = true);

	static void Destroy(FMLKernel& Kernel, bool bData = true);

	static void _Unsafe_Destroy(FMLTensor& Tensor, bool bData = true);

	static void Destroy(FMLTensor& Tensor, bool bData = true);

	static void _Unsafe_Destroy(FMLConvolution& Convolution, bool bData = true);

	static void Destroy(FMLConvolution& Convolution, bool bData = true);

	static void _Unsafe_Destroy(FMLAlgorithm& Algorithm, bool bData = true);

	static void Destroy(FMLAlgorithm& Algorithm, bool bData = true);

	// static void _Unsafe_Destroy(FMLWorkspace& Workspace, bool bData = true);

	// static void Destroy(FMLWorkspace& Workspace, bool bData = true);

	static void _Unsafe_Destroy(FMLContext& Context, bool bData = true);

	static void Destroy(FMLContext& Context, bool bData = true);

	// static void _Unsafe_Destroy(FMLInstance& Instance, bool bData = true, bool bContext = false);

	// static void Destroy(FMLInstance& Instance, bool bData = true, bool bContext = false);

	static void _Unsafe_Destroy(FMLDevice& Device, bool bData = true);

	static void Destroy(FMLDevice& Device, bool bData = true);

	static void _Unsafe_Destroy(FMLResult& Result, bool bData = true);

	static void Destroy(FMLResult& Result, bool bData = true);

	static void _Unsafe_Destroy(FMLActivation& Activation, bool bData = true);

	static void Destroy(FMLActivation& Activation, bool bData = true);

	static void _Unsafe_CPU_ConvFwd(const FMLContext& Context, FMLActivation& Activation, FMLTensor& Input, FMLTensor& Output, FMLCPU& CPU);

	static void _Unsafe_CPU_RNN(const FMLContext& Context, FMLActivation& Activation, FMLTensor& Input, FMLTensor& Output, FMLCPU& CPU);

	static void _Unsafe_CPU_CTCLoss(const FMLContext& Context, FMLActivation& Activation, FMLTensor& Input, FMLTensor& Output, FMLCPU& CPU);

	//static void _Unsafe_CPU_PPOFwd(const FMLContext& Context, FMLActivation& Activation, FMLTensor& Input, FMLTensor& Output, FMLCPU& CPU);

	//static void _Unsafe_CPU_DyNN(const FMLContext& Context, FMLActivation& Activation, FMLTensor& Input, FMLTensor& Output, FMLCPU& CPU);

};