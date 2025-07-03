#include "VActCuDNN.h"

#include "HAL/UnrealMemory.h"

void FVActCuDNN::_Unsafe_Create(FCuDevice& Into, int32 Index)
{
	int32 MaxCount = 0;
	if (Index < 0) { _CHECK_CUDA(cudaGetDeviceCount(&MaxCount)); Index = MaxCount - 1; }
	Into.Index = Index;
	_CHECK_CUDA(cudaGetDevice(&Into.Id));
	_CHECK_CUDA(cudaGetDeviceProperties(&Into.Specs, Into.Id));
}

FCuDevice FVActCuDNN::CreateDevice(int32 Index)
{
	FCuDevice Into;
	_Unsafe_Create(Into, Index);
	return Into;
}

void FVActCuDNN::_Unsafe_Create(TArray<FCuDevice>& Into)
{
	int32 Count = 0;
	_CHECK_CUDA(cudaGetDeviceCount(&Count));
	Into.SetNum(Count);
	for (int32 Index = 0; Index < Count; ++Index)
	{
		_Unsafe_Create(Into[Index], Index);
	}
}

TArray<FCuDevice> FVActCuDNN::CreateDevice()
{
	TArray<FCuDevice> Into;
	_Unsafe_Create(Into);
	return Into;
}

void FVActCuDNN::_Unsafe_Create(FCuContext& Into)
{
	_CHECK_CUDNN(cudnnCreate(&Into.Handle));
}

FCuContext FVActCuDNN::CreateContext()
{
	FCuContext Into;
	_Unsafe_Create(Into);
	return Into;
}

//template<cudnnDataType_t InType = CUDNN_CROSS_CORRELATION, cudnnTensorFormat_t InFormat = CUDNN_TENSOR_NHWC>
void FVActCuDNN::_Unsafe_Create(FCuTensor& Into, int32 Width, int32 Height, int32 Size, int32 Count)
{
	Into.Format = CUDNN_TENSOR_NCHW;//CUDNN_TENSOR_NHWC;
	Into.Type = CUDNN_DATA_FLOAT;
	_CHECK_CUDNN(cudnnCreateTensorDescriptor(&Into.Desc));
	_CHECK_CUDNN(cudnnSetTensor4dDescriptor(Into.Desc, Into.Format, Into.Type, Count, Size, Height, Width));
	Into.SizeOf = (Count * Size * Width * Height) * sizeof(float);
	_CHECK_CUDA(cudaMallocManaged(&Into.Bytes, Into.SizeOf));
}

FCuTensor FVActCuDNN::CreateTensor(int32 Width, int32 Height, int32 Size, int32 Count)
{
	FCuTensor Into;
	_Unsafe_Create(Into, Width, Height, Size, Count);
	return Into;
}

//template<cudnnDataType_t InType = CUDNN_CROSS_CORRELATION, cudnnTensorFormat_t InFormat = CUDNN_TENSOR_NHWC>
// void FVActCuDNN::_Unsafe_Create(FCuTensor& Into, int32 N, const TArray<int32> Dims, const TArray<int32> Strids)
// {
// 	_CHECK_CUDNN(cudnnCreateTensorDescriptor(Into.Desc));
// 	_CHECK_CUDNN(cudnnSetTensorNdDescriptor(Into.Desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, N, *Dims, *Strids));
// 	Into.SizeOf = sizeof(float);
// 	for(int32 Index = 0; Index < N; ++Index) { Into.SizeOf *= Dims[Index]; }
// }

//template<cudnnDataType_t InType = CUDNN_CROSS_CORRELATION, cudnnTensorFormat_t InFormat = CUDNN_TENSOR_NHWC>
void FVActCuDNN::_Unsafe_Create(FCuKernel& Into, int32 Width, int32 Height, int32 SizeOut, int32 SizeIn)
{
	Into.Format = CUDNN_TENSOR_NCHW;//CUDNN_TENSOR_NHWC;
	Into.Type = CUDNN_DATA_FLOAT;
	_CHECK_CUDNN(cudnnCreateFilterDescriptor(&Into.Desc));
	_CHECK_CUDNN(cudnnSetFilter4dDescriptor(Into.Desc, Into.Type, Into.Format, SizeOut, SizeIn, Height, Width));
	Into.SizeOf = (SizeIn * SizeOut * Width * Height) * sizeof(float);
	_CHECK_CUDA(cudaMallocManaged(&Into.Bytes, Into.SizeOf));
}

FCuKernel FVActCuDNN::CreateKernel(int32 Width, int32 Height, int32 SizeOut, int32 SizeIn)
{
	FCuKernel Into;
	_Unsafe_Create(Into, Width, Height, SizeOut, SizeIn);
	return Into;
}

//template<cudnnDataType_t InType = CUDNN_CROSS_CORRELATION, cudnnTensorFormat_t InFormat = CUDNN_TENSOR_NHWC>
// void FVActCuDNN::_Unsafe_Create(FCuKernel& Into, int32 N, const TArray<int32> Dims)
// {
// 	_CHECK_CUDNN(cudnnCreateFilterDescriptor(Into.Desc));
// 	_CHECK_CUDNN(cudnnSetFilterNdDescriptor(Into.Desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, N, *Dims));
// 	Into.SizeOf = sizeof(float);
// 	for(int32 Index = 2; Index < N; ++Index) { Into.SizeOf *= Dims[Index]; }
// }

//template<cudnnDataType_t InType = CUDNN_CROSS_CORRELATION,cu cudnnConvolutionMode_t InMode = CUDNN_DATA_FLOAT>
void FVActCuDNN::_Unsafe_Create(FCuConvolution& Into, int32 PadWidth, int32 PadHeight, int32 StrideWidth, int32 StrideHeight, int32 DilationWidth, int32 DilationHeight)
{
	Into.Mode = CUDNN_CROSS_CORRELATION;
	Into.Type = CUDNN_DATA_FLOAT;
	_CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&Into.Desc));
	_CHECK_CUDNN(cudnnSetConvolution2dDescriptor(Into.Desc, PadHeight, PadWidth, StrideHeight, StrideWidth, DilationHeight, DilationWidth, Into.Mode, Into.Type));
}

FCuConvolution FVActCuDNN::CreateConvolution(int32 PadWidth, int32 PadHeight, int32 StrideWidth, int32 StrideHeight, int32 DilationWidth, int32 DilationHeight)
{
	FCuConvolution Into;
	_Unsafe_Create(Into, PadWidth, PadHeight, StrideWidth, StrideHeight, DilationHeight, DilationWidth);
	return Into;
}

//template<cudnnDataType_t InType = CUDNN_CROSS_CORRELATION, cudnnConvolutionMode_t InMode = CUDNN_DATA_FLOAT>
// void FVActCuDNN::_Unsafe_Create(FCuConvolution& Into, int32 N, const TArray<int32> Pads, const TArray<int32> Strids, const TArray<int32> Dilations)
// {
// 	_CHECK_CUDNN(cudnnCreateConvolotion(Into.Desc));
// 	_CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(Into.Desc, N, *Pads, *Strids, *Dilations, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
// }

void FVActCuDNN::_Unsafe_Create(FCuAlgorithm& Into, FCuContext& Context, FCuTensor& Input, FCuKernel& Kernel, FCuConvolution& Convolution, FCuTensor& Output, FCuResult& Result, int32 LimitCount, bool bAllowShringking)
{
	Into.Type = ECuAlgorithm::ConvFwd;
	Result.Type = ECuResult::ConvFwd;
	_CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(Context.Handle, &Result.MaxCount));
	_CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(Context.Handle, Input.Desc, Kernel.Desc, Convolution.Desc, Output.Desc, LimitCount, &Result.Count, &Result.ConvFwd));
}

FCuAlgorithm FVActCuDNN::CreateAlgorithm(FCuContext& Context, FCuTensor& Input, FCuKernel& Kernel, FCuConvolution& Convolution, FCuTensor& Output, FCuResult& Result, int32 Count, bool bAllowShringking)
{
	FCuAlgorithm Into;
	_Unsafe_Create(Into, Context, Input, Kernel, Convolution, Output, Result, Count, bAllowShringking);
	return Into;
}

void FVActCuDNN::_Unsafe_Create(FCuActivation& Into, FCuTensor& Input, FCuTensor& Output, float Alpha, float Beta, double Gamma)
{
	Into.Mode = CUDNN_ACTIVATION_SIGMOID;
	Into.Propagation = CUDNN_NOT_PROPAGATE_NAN;
	Into.Alpha[0] = Alpha;
	Into.Beta[0] = Beta;
	
	_CHECK_CUDNN(cudnnCreateActivationDescriptor(&Into.Desc));
	_CHECK_CUDNN(cudnnSetActivationDescriptor(Into.Desc, Into.Mode, Into.Propagation, Gamma));
}
	
FCuActivation FVActCuDNN::CreateActivation(FCuTensor& Input, FCuTensor& Output, float Alpha, float Beta, double Gamma)
{
	FCuActivation Into;
	_Unsafe_Create(Into, Input, Output, Alpha, Beta, Gamma);
	return Into;
}

void FVActCuDNN::_Unsafe_Create(FCuInstance& Into, FCuContext& Context, FCuTensor& Input, FCuKernel& Kernel, FCuConvolution& Convolution, FCuTensor& Output, FCuAlgorithm& Algorithm, TArray<float> KernelData, int32 DefaultByte)
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

FCuInstance FVActCuDNN::CreateInstance(FCuContext& Context, FCuTensor& Input, FCuKernel& Kernel, FCuConvolution& Convolution, FCuTensor& Output, FCuAlgorithm& Algorithm, TArray<float> KernelData, int32 DefaultByte)
{
	FCuInstance Into;
	_Unsafe_Create(Into, Context, Input, Kernel, Convolution, Output, Algorithm, KernelData, DefaultByte);
	return Into;
}

void FVActCuDNN::_Unsafe_Use(const FCuDevice& Device)
{
	_CHECK_CUDA(cudaSetDevice(Device.Index));
}

void FVActCuDNN::Use(const FCuDevice& Device)
{
	_Unsafe_Use(Device);
}

void FVActCuDNN::_Unsafe_Forward(const FCuContext& Context, FCuActivation& Activation, FCuTensor& Input, FCuTensor& Output)
{
	_CHECK_CUDNN(cudnnActivationForward(Context.Handle, Activation.Desc, Activation.Alpha, Input.Desc, Input.Bytes, Activation.Beta, Output.Desc, Output.Bytes));
}

void FVActCuDNN::Forward(const FCuContext& Context, FCuActivation& Activation, FCuTensor& Input, FCuTensor& Output)
{
	_Unsafe_Forward(Context, Activation, Input, Output);
}

// void FVActCuDNN::_Unsafe_Use(FCuInstance& Instance)
// {
// 	_CHECK_CUDNN(cudnnConvolutionForward(Instance.Context, Alpha, Instance.Input, Instance.InputData, Instance.Kernel, Instance.Kernel, Instance.KernelData, Instance.Convolution, Instance.Algorithm, Instance.Bytes, Instance.SizeOf, Beta, Instance.Output, Instance.OutputData));
// }

// FCuInstance& FVActCuDNN::Use(FCuInstance& Instance)
// {
// 	_Unsafe_Use(Instance);
// 	return Instance;
// }

// void FVActCuDNN::_Unsafe_Create(FCuInstance& Into, FCuContext& Context, TArray<float> InputData, TArray<float> OutputData, FCuConvolution& Convolution, FCuAlgorithm& Algorithm, int32 Size, int32 Count = 1)
// {
// 	FCuTensor Input, Output;
// 	_Unsafe_Create(Input, )
// }

void FVActCuDNN::_Unsafe_Destroy(FCuKernel& Kernel, bool bData)
{
	if (bData) { _CHECK_CUDA(cudaFree(Kernel.Bytes)); Kernel.Bytes = nullptr; }
	_CHECK_CUDNN(cudnnDestroyFilterDescriptor(Kernel.Desc));
}

void FVActCuDNN::Destroy(FCuKernel& Kernel, bool bData)
{
	_Unsafe_Destroy(Kernel, bData);
}

void FVActCuDNN::_Unsafe_Destroy(FCuTensor& Tensor, bool bData)
{
	if (bData) { _CHECK_CUDA(cudaFree(Tensor.Bytes)); Tensor.Bytes = nullptr; }
	_CHECK_CUDNN(cudnnDestroyTensorDescriptor(Tensor.Desc));
}

void FVActCuDNN::Destroy(FCuTensor& Tensor, bool bData)
{
	_Unsafe_Destroy(Tensor, bData);
}

void FVActCuDNN::_Unsafe_Destroy(FCuConvolution& Convolution, bool bData)
{
	_CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(Convolution.Desc));
}

void FVActCuDNN::Destroy(FCuConvolution& Convolution, bool bData)
{
	_Unsafe_Destroy(Convolution, bData);
}

void FVActCuDNN::_Unsafe_Destroy(FCuAlgorithm& Algorithm, bool bData)
{
	_CHECK_CUDNN(cudnnDestroyAlgorithmDescriptor(Algorithm.Desc));
}

void FVActCuDNN::Destroy(FCuAlgorithm& Algorithm, bool bData)
{
	_Unsafe_Destroy(Algorithm, bData);
}

// void FVActCuDNN::_Unsafe_Destroy(FCuWorkspace& Workspace, bool bData)
// {
// 	if (bData) { _CHECK_CUDA(cudaFree(Workspace.Bytes)); Workspace.Bytes = nullptr; }
// }

// void FVActCuDNN::Destroy(FCuWorkspace& Workspace, bool bData)
// {
// 	_Unsafe_Destroy(Workspace, bData);
// }

void FVActCuDNN::_Unsafe_Destroy(FCuContext& Context, bool bData)
{
	_CHECK_CUDNN(cudnnDestroy(Context.Handle));
}

void FVActCuDNN::Destroy(FCuContext& Context, bool bData)
{
	_Unsafe_Destroy(Context, bData);
}

// static void FVActCuDNN::_Unsafe_Destroy(FCuInstance& Instance, bool bData, bool bContext)
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

// static void FVActCuDNN::_Unsafe_Destroy(FCuInstance& Instance, bool bData, bool bContext)
// {
// 	_Unsafe_Destory(Instance, bData, bContext);
// }

void FVActCuDNN::_Unsafe_Destroy(FCuDevice& Device, bool bData)
{
}

void FVActCuDNN::Destroy(FCuDevice& Device, bool bData)
{
	_Unsafe_Destroy(Device, bData);
}


void FVActCuDNN::_Unsafe_Destroy(FCuResult& Result, bool bData)
{
	//if (bData){ _CHECK_CUDNN(cudnnDestroyAlgorithmPerformance(Result.ConvFwd, Result.Count)); }
}

void FVActCuDNN::Destroy(FCuResult& Result, bool bData)
{
	_Unsafe_Destroy(Result, bData);
}

void FVActCuDNN::_Unsafe_Destroy(FCuActivation& Activation, bool bData)
{
	_CHECK_CUDNN(cudnnDestroyActivationDescriptor(Activation.Desc));
}

void FVActCuDNN::Destroy(FCuActivation& Activation, bool bData)
{
	_Unsafe_Destroy(Activation, bData);
}


//void FVActCuDNN::_Unsafe_Save(const FCuFormatSafetensorHeader& Model, ECuFormatSafetensor Format)
//{
//	switch (Format)
//	{
//	case ECuFormatSafetensor::JSon:
//		/* code */
//		break;
//	case ECuFormatSafetensor::Binary:
//		/* code */
//		break;
//	case ECuFormatSafetensor::VActBinary:
//		/* code */
//		break;
//	default:
//		break;
//	}
//}


//void FVActCuDNN::Save(const FCuFormatSafetensorHeader& Model, ECuFormatSafetensor Format)
//{
//	_Unsafe_Save(Activation, bData);
//}
//
//void FVActCuDNN::_Unsafe_Load(FCuFormatSafetensorHeader& Into, ECuFormatSafetensor Format)
//{
//	switch (Format)
//	{
//	case ECuFormatSafetensor::JSon:
//		/* code */
//		break;
//	case ECuFormatSafetensor::Binary:
//		/* code */
//		break;
//	case ECuFormatSafetensor::VActBinary:
//		/* code */
//		break;
//	default:
//		break
//}
//
//void FVActCuDNN::Load(FCuFormatSafetensorHeader& Into, ECuFormatSafetensor Format)
//{
//	_Unsafe_Load(Activation, bData);
//}