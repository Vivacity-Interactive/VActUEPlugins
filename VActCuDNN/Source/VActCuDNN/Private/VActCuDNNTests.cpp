#if WITH_EDITOR
#include "VActCuDNNTypes.h"
#include "VActCuDNN.h"

void FVActCuDNNTests::_Debug_Cuda_Test()
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

	UE_LOG(LogTemp, Display, TEXT("VActCuDNN Cuda Test Success"));
}

void FVActCuDNNTests::_Debug_CuDNN_Test_0()
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

	UE_LOG(LogTemp, Display, TEXT("VActCuDNN CuDNN Test 0 Success"));
}

void FVActCuDNNTests::_Debug_CuDNN_Test_1()
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

	UE_LOG(LogTemp, Display, TEXT("VActCuDNN Test 1 Convolution"));
	_CHECK_CUDNN(cudnnConvolutionForward(Handle, &Alpha, InputDesc, Input, KernelDesc, Kernel, ConvolutionDesc, Algorithm, Workspace, WorkspaceSizeOf, &Beta, OutputDesc, Output));	
	_CHECK_CUDA(cudaMemcpy(OutputX, Output, ImageSizeOf, CopyFrom));

	UE_LOG(LogTemp, Display, TEXT("VActCuDNN Test 1 Activation"));
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

	UE_LOG(LogTemp, Display, TEXT("VActCuDNN CuDNN Test 1 Success"));
}

void FVActCuDNNTests::_Debug_VActCuDNN_Test_0()
{
	const int32 Width = 4, Height = 4, Size = 1;
	TArray<float> InputData, OutputData;

	FMLDevice Device = CreateDevice(-1);
	UE_LOG(LogTemp, Display, TEXT("VActCuDNN Test 0 Using Device %s(%d, %d)"), ANSI_TO_TCHAR(Device.Specs.name), Device.Index, Device.Id);
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
	
	UE_LOG(LogTemp, Display, TEXT("VActCuDNN Test 0 Success"));
}

void FVActCuDNNTests::_Debug_VActCuDNN_Test_1()
{
	//UTexture* InputData, * OutputData, KernelData;
	const int32 Width = 3, Height = 3, Size = 1;
	TArray<float> InputData, OutputData, KernelData;

	FMLResult Result;

	FMLDevice Device = CreateDevice(-1);
	UE_LOG(LogTemp, Display, TEXT("VActCuDNN Test 1 Using Device %s(%d, %d)"), ANSI_TO_TCHAR(Device.Specs.name), Device.Index, Device.Id);
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
	
	UE_LOG(LogTemp, Display, TEXT("VActCuDNN Test 1 Success"));
}

#endif;