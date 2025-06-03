#include "VActML.h"

#include "HAL/UnrealMemory.h"

#if WITH_EDITORONLY_DATA
#include "Math/UnrealMathUtility.h"
#include <stdlib.h>
#endif;

#include "Misc/MessageDialog.h"
#include "Modules/ModuleManager.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"

// #include "ThirdParty/Cuda/Public/Cuda/cuda_runtime.h"
// #include "ThirdParty/cudnn/Public/cudnn/cudnn.h"

#define LOCTEXT_NAMESPACE "FVActMLModule"

#define _VACT_LOADLIB(LibOut, LibPathIn, LibNameIn)											\
{																							\
	LibOut = !LibPathIn.IsEmpty() ? FPlatformProcess::GetDllHandle(*LibPathIn) : nullptr;	\
	bool bFail = !(LibOut);																	\
	if (bFail)																				\
	{																						\
		UE_LOG(LogTemp, Fatal, TEXT("VActML \"%s\" failed to load \"%s\"")					\
			, LibNameIn, *LibPathIn);														\
	}																						\
	else																					\
	{																						\
		UE_LOG(LogTemp, Display, TEXT("VActML \"%s\" successfully loaded \"%s\"")			\
			, LibNameIn, *LibPathIn);														\
	}																						\
}

#define _VACT_FREELIB(LibIn)																\
{																							\
	FPlatformProcess::FreeDllHandle(LibIn);													\
	LibIn = nullptr;																		\
}																							\

void FVActMLModule::StartupModule()
{
	FString BaseDir = IPluginManager::Get().FindPlugin("VActML")->GetBaseDir();

	FString CudaLibPath;

	FString CuDNNLibPath;
	FString CuDNNOpsInferLibPath;
	FString CuDNNOpsTrainLibPath;
	FString CuDNNCNNInferLibPath;
	FString CuDNNCNNTrainLibPath;
	FString CuDNNAdvInferLibPath;
	FString CuDNNAdvTrainLibPath;

#if PLATFORM_WINDOWS
	CudaLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/Cuda/Win64/cudart64_12.dll"));

	CuDNNLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Win64/cudnn64_8.dll"));
	CuDNNOpsInferLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Win64/cudnn_ops_infer64_8.dll"));
	CuDNNOpsTrainLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Win64/cudnn_ops_train64_8.dll"));
	CuDNNCNNInferLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Win64/cudnn_cnn_infer64_8.dll"));
	CuDNNCNNTrainLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Win64/cudnn_cnn_train64_8.dll"));
	CuDNNAdvInferLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Win64/cudnn_adv_infer64_8.dll"));
	CuDNNAdvTrainLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Win64/cudnn_adv_train64_8.dll"));
#elif PLATFORM_MAC
	CudaLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/Cuda/Mac/cudart64_12.dylib"));

	CuDNNLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Mac/cudnn64_8.dylib"));
	CuDNNOpsInferLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Mac/cudnn_ops_infer64_8.dylib"));
	CuDNNOpsTrainLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Mac/cudnn_ops_train64_8.dylib"));
	CuDNNCNNInferLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Mac/cudnn_cnn_infer64_8.dylib"));
	CuDNNCNNTrainLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Mac/cudnn_cnn_train64_8.dylib"));
	CuDNNAdvInferLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Mac/cudnn_adv_infer64_8.dylib"));
	CuDNNAdvTrainLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Mac/cudnn_adv_train64_8.dylib"));
#elif PLATFORM_LINUX
	CudaLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/Cuda/Linux/cudart64_12.so"));

	CuDNNLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Linux/cudnn64_8.so"));
	CuDNNOpsInferLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Linux/cudnn_ops_infer64_8.so"));
	CuDNNOpsTrainLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Linux/cudnn_ops_train64_8.so"));
	CuDNNCNNInferLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Linux/cudnn_cnn_infer64_8.so"));
	CuDNNCNNTrainLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Linux/cudnn_cnn_train64_8.so"));
	CuDNNAdvInferLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Linux/cudnn_adv_infer64_8.so"));
	CuDNNAdvTrainLibPath = FPaths::Combine(*BaseDir, TEXT("Source/ThirdParty/CuDNN/Linux/cudnn_adv_train64_8.so"));
#endif

	_VACT_LOADLIB(CudaLib, CudaLibPath, TEXT("Cuda"));

	_VACT_LOADLIB(CuDNNLib, CuDNNLibPath, TEXT("CuDNN"));
	_VACT_LOADLIB(CuDNNOpsInferLib, CuDNNOpsInferLibPath, TEXT("CuDNN"));
	_VACT_LOADLIB(CuDNNOpsTrainLib, CuDNNOpsTrainLibPath, TEXT("CuDNN"));
	_VACT_LOADLIB(CuDNNCNNInferLib, CuDNNCNNInferLibPath, TEXT("CuDNN"));
	_VACT_LOADLIB(CuDNNCNNTrainLib, CuDNNCNNTrainLibPath, TEXT("CuDNN"));
	_VACT_LOADLIB(CuDNNAdvInferLib, CuDNNAdvInferLibPath, TEXT("CuDNN"));
	_VACT_LOADLIB(CuDNNAdvTrainLib, CuDNNAdvTrainLibPath, TEXT("CuDNN"));

#if WITH_EDITORONLY_DATA
	//FVActML::_Debug_Cuda_Test();
	//FVActML::_Debug_CuDNN_Test_0();
	//FVActML::_Debug_CuDNN_Test_1();
	//FVActML::_Debug_VActML_Test_0();
	//FVActML::_Debug_VActML_Test_1();
#endif;
}

void FVActMLModule::ShutdownModule()
{
	_VACT_FREELIB(CudaLib);

	_VACT_FREELIB(CuDNNLib);
	_VACT_FREELIB(CuDNNOpsTrainLib);
	_VACT_FREELIB(CuDNNOpsInferLib);
	_VACT_FREELIB(CuDNNCNNInferLib);
	_VACT_FREELIB(CuDNNCNNTrainLib);
	_VACT_FREELIB(CuDNNAdvTrainLib);
	_VACT_FREELIB(CuDNNAdvInferLib);
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FVActMLModule, VActML)

#if WITH_EDITORONLY_DATA

void FVActML::_Debug_Cuda_Test()
{
	const int32 Num = 4;
	
	cudaDeviceProp Specs;
	
	int32 Device = 0, NumDevices = 0;
	
	_VACT_SIZE_T SizeOf = sizeof(float) * Num;
	
	float Data[Num] = {1.0f, 1.0f, 1.0f, 1.0f};
	void* Bytes = nullptr;

	_CHECK_CUDA(cudaGetDeviceCount(&NumDevices));
	
	_CHECK_CUDA(cudaSetDevice((NumDevices > 1)));

	_CHECK_CUDA(cudaGetDevice(&Device));
	_CHECK_CUDA(cudaGetDeviceProperties(&Specs, Device));

	_CHECK_CUDA(cudaMalloc(&Bytes, SizeOf));
	_CHECK_CUDA(cudaMemcpy(Bytes, &Data, SizeOf, cudaMemcpyHostToDevice))
	_CHECK_CUDA(cudaFree(Bytes));

	UE_LOG(LogTemp, Display, TEXT("VActML Cuda Test Success"));
}

void FVActML::_Debug_CuDNN_Test_0()
{
	cudnnHandle_t Handle = nullptr;

	cudnnDataType_t Type = CUDNN_DATA_FLOAT;
	cudnnTensorFormat_t Format = CUDNN_TENSOR_NCHW;
	cudnnTensorDescriptor_t InputDesc = nullptr, OutputDesc = nullptr;
	cudnnActivationDescriptor_t ActivationDesc = nullptr;
	cudnnActivationMode_t Mode = CUDNN_ACTIVATION_SIGMOID;
    cudnnNanPropagation_t Propagate = CUDNN_NOT_PROPAGATE_NAN;
	
	int32 NumDevices, Count = 1, Size = 1, Height = 4, Width = 4, Num = Count * Size * Height * Width;

	_VACT_SIZE_T SizeOf = Num * sizeof(float);
	
	double Gamma = 0.0;
	float Alpha[1] = { 1.0f }, Beta[1] = { 0.0f }, * Input, * Output;
	
	_CHECK_CUDA(cudaGetDeviceCount(&NumDevices));
	_CHECK_CUDA(cudaSetDevice((NumDevices > 1)));

	_CHECK_CUDNN(cudnnCreate(&Handle));
	_CHECK_CUDNN(cudnnCreateTensorDescriptor(&InputDesc));
	_CHECK_CUDNN(cudnnCreateTensorDescriptor(&OutputDesc));
	_CHECK_CUDNN(cudnnCreateActivationDescriptor(&ActivationDesc));
	
	_CHECK_CUDA(cudaMallocManaged(&Input, SizeOf));
	_CHECK_CUDA(cudaMallocManaged(&Output, SizeOf));

	for(int Index = 0; Index < Num; ++Index)
	{ 
		Input[Index] = Index * 1.0f; 
		Output[Index] = 0.0f;
	}

	_CHECK_CUDNN(cudnnSetTensor4dDescriptor(InputDesc, Format, Type, Count, Size, Height, Width));
	_CHECK_CUDNN(cudnnSetTensor4dDescriptor(OutputDesc, Format, Type, Count, Size, Height, Width));
	_CHECK_CUDNN(cudnnSetActivationDescriptor(ActivationDesc, Mode, Propagate, Gamma));
	_CHECK_CUDNN(cudnnActivationForward(Handle, ActivationDesc, Alpha, InputDesc, Input, Beta, OutputDesc, Output));
	
	_CHECK_CUDNN(cudnnDestroyTensorDescriptor(InputDesc));
	_CHECK_CUDNN(cudnnDestroyTensorDescriptor(OutputDesc));
	_CHECK_CUDNN(cudnnDestroyActivationDescriptor(ActivationDesc));
	_CHECK_CUDNN(cudnnDestroy(Handle));

	_CHECK_CUDA(cudaFree(Input));
	_CHECK_CUDA(cudaFree(Output));

	UE_LOG(LogTemp, Display, TEXT("VActML CuDNN Test 0 Success"));
}

void FVActML::_Debug_CuDNN_Test_1()
{
	cudnnHandle_t Handle = nullptr;
	
	cudnnTensorDescriptor_t InputDesc = nullptr, OutputDesc = nullptr;
	cudnnFilterDescriptor_t KernelDesc = nullptr;
	cudnnConvolutionDescriptor_t ConvolutionDesc = nullptr;
	cudnnActivationDescriptor_t ActivationDesc = nullptr;
	
	cudnnConvolutionFwdAlgoPerf_t Results;
	
	cudnnConvolutionFwdAlgo_t Algorithm = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
	cudnnDataType_t Type = CUDNN_DATA_FLOAT;
	cudnnTensorFormat_t Format = CUDNN_TENSOR_NCHW;
	cudnnConvolutionMode_t ConvMode = CUDNN_CROSS_CORRELATION;
	cudnnActivationMode_t Mode = CUDNN_ACTIVATION_SIGMOID;
    cudnnNanPropagation_t Propagate = CUDNN_NOT_PROPAGATE_NAN;

	cudaMemcpyKind CopyTo = cudaMemcpyHostToDevice;
	cudaMemcpyKind CopyFrom = cudaMemcpyDeviceToHost;

	int32 NumDevices, Count = 1, Size = 3, Height = 11, Width = 11;
	int32 ImageNum = Height * Width * Size, KernelNum = Size * Size * Size * Size, Num = Count * ImageNum;
	int32 PadH = 1, PadW = 1, StrideH = 1, StrideW = 1, DilationH = 1, DilationW = 1;
	int32 AlgoNum = 0, RequestNum = 1;
	
	_VACT_SIZE_T WorkspaceSizeOf = 0, SizeOf = Num * sizeof(float), KernelSizeOf = KernelNum * sizeof(float), ImageSizeOf = ImageNum * sizeof(float);
	
	void* Workspace = nullptr;
	float* Input = nullptr, * Output = nullptr, * Kernel = nullptr;
	
	float* ImageX = (float*) malloc(ImageSizeOf);
	float* KernelX = (float*) malloc(KernelSizeOf);
	float* OutputX = (float*) malloc(ImageSizeOf);

	float Alpha = 1.0f, Beta = 0.0f;
	double Gamma = 0.0;

	for (int32 Index = 0; Index < Num; ++Index)
	{
		ImageX[Index] = FMath::FRandRange(0.0f, 1.0f);
	}

	for (int32 Index = 0; Index < KernelNum / 2; ++Index)
	{
		KernelX[Index * 2] = FMath::FRandRange(0.0f, 1.0f);
		KernelX[Index * 2 + 1] = FMath::FRandRange(-1.0f, 0.0f);
	}

	_CHECK_CUDA(cudaGetDeviceCount(&NumDevices));
	_CHECK_CUDA(cudaSetDevice((NumDevices > 1)));

	_CHECK_CUDNN(cudnnCreate(&Handle));

	_CHECK_CUDNN(cudnnCreateTensorDescriptor(&InputDesc));
	_CHECK_CUDNN(cudnnCreateTensorDescriptor(&OutputDesc));
	_CHECK_CUDNN(cudnnCreateFilterDescriptor(&KernelDesc));
	_CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&ConvolutionDesc));
	_CHECK_CUDNN(cudnnCreateActivationDescriptor(&ActivationDesc));

	_CHECK_CUDNN(cudnnSetTensor4dDescriptor(InputDesc, Format, Type, Count, Size, Height, Width));
	_CHECK_CUDNN(cudnnSetTensor4dDescriptor(OutputDesc, Format, Type, Count, Size, Height, Width));
	_CHECK_CUDNN(cudnnSetFilter4dDescriptor(KernelDesc, Type, Format, Size, Size, Size, Size));
	_CHECK_CUDNN(cudnnSetConvolution2dDescriptor(ConvolutionDesc, PadH, PadW, StrideH, StrideW, DilationH, DilationW, ConvMode, Type));
	_CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(Handle, InputDesc, KernelDesc, ConvolutionDesc, OutputDesc, Algorithm, &RequestNum, &Results));
	_CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(Handle, InputDesc, KernelDesc, ConvolutionDesc, OutputDesc, Algorithm, &WorkspaceSizeOf));
	_CHECK_CUDNN(cudnnSetActivationDescriptor(ActivationDesc, Mode, Propagate, Gamma));

	_CHECK_CUDA(cudaMalloc(&Workspace, WorkspaceSizeOf));
	_CHECK_CUDA(cudaMalloc(&Input, SizeOf));
	_CHECK_CUDA(cudaMalloc(&Output, SizeOf));
	_CHECK_CUDA(cudaMalloc(&Kernel, KernelSizeOf));

	_CHECK_CUDA(cudaMemcpy(Input, ImageX, SizeOf, CopyTo));
	_CHECK_CUDA(cudaMemset(Output, 0, SizeOf));
	_CHECK_CUDA(cudaMemcpy(Kernel, KernelX, KernelSizeOf, CopyTo));

	UE_LOG(LogTemp, Display, TEXT("VActML Test 1 Convolution"));
	_CHECK_CUDNN(cudnnConvolutionForward(Handle, &Alpha, InputDesc, Input, KernelDesc, Kernel, ConvolutionDesc, Algorithm, Workspace, WorkspaceSizeOf, &Beta, OutputDesc, Output));	
	_CHECK_CUDA(cudaMemcpy(OutputX, Output, ImageSizeOf, CopyFrom));

	UE_LOG(LogTemp, Display, TEXT("VActML Test 1 Activation"));
	_CHECK_CUDNN(cudnnActivationForward(Handle, ActivationDesc, &Alpha, OutputDesc, Output, &Beta, OutputDesc, Output));
	_CHECK_CUDA(cudaMemcpy(OutputX, Output, ImageSizeOf, CopyFrom));

	free(ImageX);
	free(OutputX);
	free(KernelX);

	_CHECK_CUDA(cudaFree(Kernel));
	_CHECK_CUDA(cudaFree(Input));
	_CHECK_CUDA(cudaFree(Output));
	_CHECK_CUDA(cudaFree(Workspace));

	_CHECK_CUDNN(cudnnDestroyTensorDescriptor(InputDesc));
	_CHECK_CUDNN(cudnnDestroyTensorDescriptor(OutputDesc));
	_CHECK_CUDNN(cudnnDestroyFilterDescriptor(KernelDesc));
	_CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(ConvolutionDesc));
	_CHECK_CUDNN(cudnnDestroyActivationDescriptor(ActivationDesc));

	_CHECK_CUDNN(cudnnDestroy(Handle));

	UE_LOG(LogTemp, Display, TEXT("VActML CuDNN Test 1 Success"));
}

void FVActML::_Debug_VActML_Test_0()
{
	const int32 Width = 4, Height = 4, Size = 1;
	TArray<float> InputData, OutputData;

	FMLDevice Device = CreateDevice(-1);
	UE_LOG(LogTemp, Display, TEXT("VActML Test 0 Using Device %s(%d, %d)"), ANSI_TO_TCHAR(Device.Specs.name), Device.Index, Device.Id);
	Use(Device);

	FMLContext Context = CreateContext();
	// TODO Find Way To Use TArray as data
	FMLTensor Input = CreateTensor(Width, Height, Size);
	FMLTensor Output = CreateTensor(Width, Height, Size);
	FMLActivation Activation = CreateActivation(Input, Output);

	Forward(Context, Activation, Input, Output);

	Destroy(Input);
	Destroy(Output);
	Destroy(Activation);
	Destroy(Context);
	
	UE_LOG(LogTemp, Display, TEXT("VActML Test 0 Success"));
}

void FVActML::_Debug_VActML_Test_1()
{
	//UTexture* InputData, * OutputData, KernelData;
	const int32 Width = 3, Height = 3, Size = 1;
	TArray<float> InputData, OutputData, KernelData;

	FMLResult Result;

	FMLDevice Device = CreateDevice(-1);
	UE_LOG(LogTemp, Display, TEXT("VActML Test 1 Using Device %s(%d, %d)"), ANSI_TO_TCHAR(Device.Specs.name), Device.Index, Device.Id);
	Use(Device);

	FMLContext Context = CreateContext();
	FMLTensor Input = CreateTensor(Width, Height, Size);
	FMLTensor Output = CreateTensor(Width, Height, Size);
	FMLKernel Kernel = CreateKernel(Size, Size, Size, Size);
	FMLConvolution Convolution = CreateConvolution();
	FMLAlgorithm Algorithm = CreateAlgorithm(Context, Input, Kernel, Convolution, Output, Result);

	const float KernelTemplate[Height][Width] = { { 1.0f, 1.0f, 1.0f }, { 1.0f, -8.0f, 1.0f }, { 1.0f, 1.0f, 1.0f } };
	
	int32 nHeight = Size;
	int32 nChannel = nHeight * Size;
	int32 nKernel = nChannel * Size;
	KernelData.SetNum(nKernel * Size);
	for (int32 iKernel = 0; iKernel < Size; ++iKernel)
	{
		for (int32 iChannel = 0; iChannel < Size; ++iChannel)
		{
			for (int32 iHeight = 0; iHeight < Height; ++iHeight)
			{
				for (int32 iWidth = 0; iWidth < Width; ++iWidth)
				{
					int32 Index = iKernel * nKernel + iChannel * nChannel + iHeight * nHeight + iWidth;
					KernelData[Index] = KernelTemplate[iHeight][iWidth];
				}
			}
		}
	}
	
	FMLInstance Instance = CreateInstance(Context, Input, Kernel, Convolution, Output, Algorithm, KernelData);

	//Use(Context, Alpha, Beta, Input, InputData, Kernel, Convolution, Output, Algorithm, InputData, OutputData, KernelData);
	//Use(Instance);

	Destroy(Kernel);
	Destroy(Input);
	Destroy(Output);
	Destroy(Convolution);
	Destroy(Algorithm);
	Destroy(Result);
	// Destroy(Workspace);
	
	// Destroy(Instance);
	
	Destroy(Context);
	
	UE_LOG(LogTemp, Display, TEXT("VActML Test 1 Success"));
}

#endif;

void FVActML::_Unsafe_Create(FMLDevice& Into, int32 Index)
{
	int32 MaxCount = 0;
	if (Index < 0) { _CHECK_CUDA(cudaGetDeviceCount(&MaxCount)); Index = MaxCount - 1; }
	Into.Index = Index;
	_CHECK_CUDA(cudaGetDevice(&Into.Id));
	_CHECK_CUDA(cudaGetDeviceProperties(&Into.Specs, Into.Id));
}

FMLDevice FVActML::CreateDevice(int32 Index)
{
	FMLDevice Into;
	_Unsafe_Create(Into, Index);
	return Into;
}

void FVActML::_Unsafe_Create(TArray<FMLDevice>& Into)
{
	int32 Count = 0;
	_CHECK_CUDA(cudaGetDeviceCount(&Count));
	Into.SetNum(Count);
	for (int32 Index = 0; Index < Count; ++Index)
	{
		_Unsafe_Create(Into[Index], Index);
	}
}

TArray<FMLDevice> FVActML::CreateDevice()
{
	TArray<FMLDevice> Into;
	_Unsafe_Create(Into);
	return Into;
}

void FVActML::_Unsafe_Create(FMLContext& Into)
{
	_CHECK_CUDNN(cudnnCreate(&Into.Handle));
}

FMLContext FVActML::CreateContext()
{
	FMLContext Into;
	_Unsafe_Create(Into);
	return Into;
}

//template<cudnnDataType_t InType = CUDNN_CROSS_CORRELATION, cudnnTensorFormat_t InFormat = CUDNN_TENSOR_NHWC>
void FVActML::_Unsafe_Create(FMLTensor& Into, int32 Width, int32 Height, int32 Size, int32 Count)
{
	Into.Format = CUDNN_TENSOR_NCHW;//CUDNN_TENSOR_NHWC;
	Into.Type = CUDNN_DATA_FLOAT;
	_CHECK_CUDNN(cudnnCreateTensorDescriptor(&Into.Desc));
	_CHECK_CUDNN(cudnnSetTensor4dDescriptor(Into.Desc, Into.Format, Into.Type, Count, Size, Height, Width));
	Into.SizeOf = (Count * Size * Width * Height) * sizeof(float);
	_CHECK_CUDA(cudaMallocManaged(&Into.Bytes, Into.SizeOf));
}

FMLTensor FVActML::CreateTensor(int32 Width, int32 Height, int32 Size, int32 Count)
{
	FMLTensor Into;
	_Unsafe_Create(Into, Width, Height, Size, Count);
	return Into;
}

//template<cudnnDataType_t InType = CUDNN_CROSS_CORRELATION, cudnnTensorFormat_t InFormat = CUDNN_TENSOR_NHWC>
// void FVActML::_Unsafe_Create(FMLTensor& Into, int32 N, const TArray<int32> Dims, const TArray<int32> Strids)
// {
// 	_CHECK_CUDNN(cudnnCreateTensorDescriptor(Into.Desc));
// 	_CHECK_CUDNN(cudnnSetTensorNdDescriptor(Into.Desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, N, *Dims, *Strids));
// 	Into.SizeOf = sizeof(float);
// 	for(int32 Index = 0; Index < N; ++Index) { Into.SizeOf *= Dims[Index]; }
// }

//template<cudnnDataType_t InType = CUDNN_CROSS_CORRELATION, cudnnTensorFormat_t InFormat = CUDNN_TENSOR_NHWC>
void FVActML::_Unsafe_Create(FMLKernel& Into, int32 Width, int32 Height, int32 SizeOut, int32 SizeIn)
{
	Into.Format = CUDNN_TENSOR_NCHW;//CUDNN_TENSOR_NHWC;
	Into.Type = CUDNN_DATA_FLOAT;
	_CHECK_CUDNN(cudnnCreateFilterDescriptor(&Into.Desc));
	_CHECK_CUDNN(cudnnSetFilter4dDescriptor(Into.Desc, Into.Type, Into.Format, SizeOut, SizeIn, Height, Width));
	Into.SizeOf = (SizeIn * SizeOut * Width * Height) * sizeof(float);
	_CHECK_CUDA(cudaMallocManaged(&Into.Bytes, Into.SizeOf));
}

FMLKernel FVActML::CreateKernel(int32 Width, int32 Height, int32 SizeOut, int32 SizeIn)
{
	FMLKernel Into;
	_Unsafe_Create(Into, Width, Height, SizeOut, SizeIn);
	return Into;
}

//template<cudnnDataType_t InType = CUDNN_CROSS_CORRELATION, cudnnTensorFormat_t InFormat = CUDNN_TENSOR_NHWC>
// void FVActML::_Unsafe_Create(FMLKernel& Into, int32 N, const TArray<int32> Dims)
// {
// 	_CHECK_CUDNN(cudnnCreateFilterDescriptor(Into.Desc));
// 	_CHECK_CUDNN(cudnnSetFilterNdDescriptor(Into.Desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, N, *Dims));
// 	Into.SizeOf = sizeof(float);
// 	for(int32 Index = 2; Index < N; ++Index) { Into.SizeOf *= Dims[Index]; }
// }

//template<cudnnDataType_t InType = CUDNN_CROSS_CORRELATION,cu cudnnConvolutionMode_t InMode = CUDNN_DATA_FLOAT>
void FVActML::_Unsafe_Create(FMLConvolution& Into, int32 PadWidth, int32 PadHeight, int32 StrideWidth, int32 StrideHeight, int32 DilationWidth, int32 DilationHeight)
{
	Into.Mode = CUDNN_CROSS_CORRELATION;
	Into.Type = CUDNN_DATA_FLOAT;
	_CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&Into.Desc));
	_CHECK_CUDNN(cudnnSetConvolution2dDescriptor(Into.Desc, PadHeight, PadWidth, StrideHeight, StrideWidth, DilationHeight, DilationWidth, Into.Mode, Into.Type));
}

FMLConvolution FVActML::CreateConvolution(int32 PadWidth, int32 PadHeight, int32 StrideWidth, int32 StrideHeight, int32 DilationWidth, int32 DilationHeight)
{
	FMLConvolution Into;
	_Unsafe_Create(Into, PadWidth, PadHeight, StrideWidth, StrideHeight, DilationHeight, DilationWidth);
	return Into;
}

//template<cudnnDataType_t InType = CUDNN_CROSS_CORRELATION, cudnnConvolutionMode_t InMode = CUDNN_DATA_FLOAT>
// void FVActML::_Unsafe_Create(FMLConvolution& Into, int32 N, const TArray<int32> Pads, const TArray<int32> Strids, const TArray<int32> Dilations)
// {
// 	_CHECK_CUDNN(cudnnCreateConvolotion(Into.Desc));
// 	_CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(Into.Desc, N, *Pads, *Strids, *Dilations, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
// }

void FVActML::_Unsafe_Create(FMLAlgorithm& Into, FMLContext& Context, FMLTensor& Input, FMLKernel& Kernel, FMLConvolution& Convolution, FMLTensor& Output, FMLResult& Result, int32 LimitCount, bool bAllowShringking)
{
	Into.Type = EMLAlgorithm::ConvFwd;
	Result.Type = EMLResult::ConvFwd;
	_CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(Context.Handle, &Result.MaxCount));
	_CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(Context.Handle, Input.Desc, Kernel.Desc, Convolution.Desc, Output.Desc, LimitCount, &Result.Count, &Result.ConvFwd));
}

FMLAlgorithm FVActML::CreateAlgorithm(FMLContext& Context, FMLTensor& Input, FMLKernel& Kernel, FMLConvolution& Convolution, FMLTensor& Output, FMLResult& Result, int32 Count, bool bAllowShringking)
{
	FMLAlgorithm Into;
	_Unsafe_Create(Into, Context, Input, Kernel, Convolution, Output, Result, Count, bAllowShringking);
	return Into;
}

void FVActML::_Unsafe_Create(FMLActivation& Into, FMLTensor& Input, FMLTensor& Output, float Alpha, float Beta, double Gamma)
{
	Into.Mode = CUDNN_ACTIVATION_SIGMOID;
	Into.Propagation = CUDNN_NOT_PROPAGATE_NAN;
	Into.Alpha[0] = Alpha;
	Into.Beta[0] = Beta;
	
	_CHECK_CUDNN(cudnnCreateActivationDescriptor(&Into.Desc));
	_CHECK_CUDNN(cudnnSetActivationDescriptor(Into.Desc, Into.Mode, Into.Propagation, Gamma));
}
	
FMLActivation FVActML::CreateActivation(FMLTensor& Input, FMLTensor& Output, float Alpha, float Beta, double Gamma)
{
	FMLActivation Into;
	_Unsafe_Create(Into, Input, Output, Alpha, Beta, Gamma);
	return Into;
}

void FVActML::_Unsafe_Create(FMLInstance& Into, FMLContext& Context, FMLTensor& Input, FMLKernel& Kernel, FMLConvolution& Convolution, FMLTensor& Output, FMLAlgorithm& Algorithm, TArray<float> KernelData, int32 DefaultByte)
{
	_CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(Context.Handle, Input.Desc, Kernel.Desc, Convolution.Desc, Output.Desc, Algorithm.ConvFwd, &Into.SizeOf));
	_CHECK_CUDA(cudaMalloc(&Into.Bytes, Into.SizeOf));
	_CHECK_CUDA(cudaMalloc(&Input.Bytes, Input.SizeOf));
	//_CHECK_CUDA(cudaMemcpy(Input.Bytes, InputData.GetData(), Input.SizeOf, cudaMemcpyHostToDevice))
	_CHECK_CUDA(cudaMalloc(&Output.Bytes, Output.SizeOf));
	_CHECK_CUDA(cudaMemset(Output.Bytes, DefaultByte, Output.SizeOf));
	_CHECK_CUDA(cudaMalloc(&Kernel.Bytes, Kernel.SizeOf));
	_CHECK_CUDA(cudaMemcpy(Kernel.Bytes, KernelData.GetData(), Kernel.SizeOf, cudaMemcpyHostToDevice));
}

FMLInstance FVActML::CreateInstance(FMLContext& Context, FMLTensor& Input, FMLKernel& Kernel, FMLConvolution& Convolution, FMLTensor& Output, FMLAlgorithm& Algorithm, TArray<float> KernelData, int32 DefaultByte)
{
	FMLInstance Into;
	_Unsafe_Create(Into, Context, Input, Kernel, Convolution, Output, Algorithm, KernelData, DefaultByte);
	return Into;
}

void FVActML::_Unsafe_Use(const FMLDevice& Device)
{
	_CHECK_CUDA(cudaSetDevice(Device.Index));
}

void FVActML::Use(const FMLDevice& Device)
{
	_Unsafe_Use(Device);
}

void FVActML::_Unsafe_Forward(const FMLContext& Context, FMLActivation& Activation, FMLTensor& Input, FMLTensor& Output)
{
	_CHECK_CUDNN(cudnnActivationForward(Context.Handle, Activation.Desc, Activation.Alpha, Input.Desc, Input.Bytes, Activation.Beta, Output.Desc, Output.Bytes));
}

void FVActML::Forward(const FMLContext& Context, FMLActivation& Activation, FMLTensor& Input, FMLTensor& Output)
{
	_Unsafe_Forward(Context, Activation, Input, Output);
}

// void FVActML::_Unsafe_Use(FMLInstance& Instance)
// {
// 	_CHECK_CUDNN(cudnnConvolutionForward(Instance.Context, Alpha, Instance.Input, Instance.InputData, Instance.Kernel, Instance.Kernel, Instance.KernelData, Instance.Convolution, Instance.Algorithm, Instance.Bytes, Instance.SizeOf, Beta, Instance.Output, Instance.OutputData));
// }

// FMLInstance& FVActML::Use(FMLInstance& Instance)
// {
// 	_Unsafe_Use(Instance);
// 	return Instance;
// }

// void FVActML::_Unsafe_Create(FMLInstance& Into, FMLContext& Context, TArray<float> InputData, TArray<float> OutputData, FMLConvolution& Convolution, FMLAlgorithm& Algorithm, int32 Size, int32 Count = 1)
// {
// 	FMLTensor Input, Output;
// 	_Unsafe_Create(Input, )
// }

void FVActML::_Unsafe_Destroy(FMLKernel& Kernel, bool bData)
{
	if (bData) { _CHECK_CUDA(cudaFree(Kernel.Bytes)); Kernel.Bytes = nullptr; }
	_CHECK_CUDNN(cudnnDestroyFilterDescriptor(Kernel.Desc));
}

void FVActML::Destroy(FMLKernel& Kernel, bool bData)
{
	_Unsafe_Destroy(Kernel, bData);
}

void FVActML::_Unsafe_Destroy(FMLTensor& Tensor, bool bData)
{
	if (bData) { _CHECK_CUDA(cudaFree(Tensor.Bytes)); Tensor.Bytes = nullptr; }
	_CHECK_CUDNN(cudnnDestroyTensorDescriptor(Tensor.Desc));
}

void FVActML::Destroy(FMLTensor& Tensor, bool bData)
{
	_Unsafe_Destroy(Tensor, bData);
}

void FVActML::_Unsafe_Destroy(FMLConvolution& Convolution, bool bData)
{
	_CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(Convolution.Desc));
}

void FVActML::Destroy(FMLConvolution& Convolution, bool bData)
{
	_Unsafe_Destroy(Convolution, bData);
}

void FVActML::_Unsafe_Destroy(FMLAlgorithm& Algorithm, bool bData)
{
	_CHECK_CUDNN(cudnnDestroyAlgorithmDescriptor(Algorithm.Desc));
}

void FVActML::Destroy(FMLAlgorithm& Algorithm, bool bData)
{
	_Unsafe_Destroy(Algorithm, bData);
}

// void FVActML::_Unsafe_Destroy(FMLWorkspace& Workspace, bool bData)
// {
// 	if (bData) { _CHECK_CUDA(cudaFree(Workspace.Bytes)); Workspace.Bytes = nullptr; }
// }

// void FVActML::Destroy(FMLWorkspace& Workspace, bool bData)
// {
// 	_Unsafe_Destroy(Workspace, bData);
// }

void FVActML::_Unsafe_Destroy(FMLContext& Context, bool bData)
{
	_CHECK_CUDNN(cudnnDestroy(Context.Handle));
}

void FVActML::Destroy(FMLContext& Context, bool bData)
{
	_Unsafe_Destroy(Context, bData);
}

// static void FVActML::_Unsafe_Destroy(FMLInstance& Instance, bool bData, bool bContext)
// {
//	_Unsafe_Destory(Instance.Input, bData, false);
//	_Unsafe_Destory(Instance.Output, bData, false);
//	_Unsafe_Destory(Instance.Kernel, bData, false);
//	_Unsafe_Destory(Instance.Convolution, bData);
//	_Unsafe_Destroy(Instance.Algorithm, bData);
//	_Unsafe_Destory(Instance.Workspace, bData, false);
//	//_Unsafe_Destory();
//	if(bContext) { _Unsafe_Destory(Instance.Context); }
// }

// static void FVActML::_Unsafe_Destroy(FMLInstance& Instance, bool bData, bool bContext)
// {
// 	_Unsafe_Destory(Instance, bData, bContext);
// }

void FVActML::_Unsafe_Destroy(FMLDevice& Device, bool bData)
{
}

void FVActML::Destroy(FMLDevice& Device, bool bData)
{
	_Unsafe_Destroy(Device, bData);
}


void FVActML::_Unsafe_Destroy(FMLResult& Result, bool bData)
{
	//if (bData){ _CHECK_CUDNN(cudnnDestroyAlgorithmPerformance(Result.ConvFwd, Result.Count)); }
}

void FVActML::Destroy(FMLResult& Result, bool bData)
{
	_Unsafe_Destroy(Result, bData);
}

void FVActML::_Unsafe_Destroy(FMLActivation& Activation, bool bData)
{
	_CHECK_CUDNN(cudnnDestroyActivationDescriptor(Activation.Desc));
}

void FVActML::Destroy(FMLActivation& Activation, bool bData)
{
	_Unsafe_Destroy(Activation, bData);
}


//void FVActML::_Unsafe_Save(const FMLFormatSafetensorHeader& Model, EMLFormatSafetensor Format)
//{
//	switch (Format)
//	{
//	case EMLFormatSafetensor::JSon:
//		/* code */
//		break;
//	case EMLFormatSafetensor::Binary:
//		/* code */
//		break;
//	case EMLFormatSafetensor::VActBinary:
//		/* code */
//		break;
//	default:
//		break;
//	}
//}


//void FVActML::Save(const FMLFormatSafetensorHeader& Model, EMLFormatSafetensor Format)
//{
//	_Unsafe_Save(Activation, bData);
//}
//
//void FVActML::_Unsafe_Load(FMLFormatSafetensorHeader& Into, EMLFormatSafetensor Format)
//{
//	switch (Format)
//	{
//	case EMLFormatSafetensor::JSon:
//		/* code */
//		break;
//	case EMLFormatSafetensor::Binary:
//		/* code */
//		break;
//	case EMLFormatSafetensor::VActBinary:
//		/* code */
//		break;
//	default:
//		break
//}
//
//void FVActML::Load(FMLFormatSafetensorHeader& Into, EMLFormatSafetensor Format)
//{
//	_Unsafe_Load(Activation, bData);
//}