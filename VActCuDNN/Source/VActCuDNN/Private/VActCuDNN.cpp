#include "VActCuDNN.h"

#include "HAL/UnrealMemory.h"

void FVActCuDNN::_Unsafe_Create(FMLDevice& Into, int32 Index)
{
	int32 MaxCount = 0;
	if (Index < 0) { _CHECK_CUDA(cudaGetDeviceCount(&MaxCount)); Index = MaxCount - 1; }
	Into.Index = Index;
	_CHECK_CUDA(cudaGetDevice(&Into.Id));
	_CHECK_CUDA(cudaGetDeviceProperties(&Into.Specs, Into.Id));
}

FMLDevice FVActCuDNN::CreateDevice(int32 Index)
{
	FMLDevice Into;
	_Unsafe_Create(Into, Index);
	return Into;
}

void FVActCuDNN::_Unsafe_Create(TArray<FMLDevice>& Into)
{
	int32 Count = 0;
	_CHECK_CUDA(cudaGetDeviceCount(&Count));
	Into.SetNum(Count);
	for (int32 Index = 0; Index < Count; ++Index)
	{
		_Unsafe_Create(Into[Index], Index);
	}
}

TArray<FMLDevice> FVActCuDNN::CreateDevice()
{
	TArray<FMLDevice> Into;
	_Unsafe_Create(Into);
	return Into;
}

void FVActCuDNN::_Unsafe_Create(FMLContext& Into)
{
	_CHECK_CUDNN(cudnnCreate(&Into.Handle));
}

FMLContext FVActCuDNN::CreateContext()
{
	FMLContext Into;
	_Unsafe_Create(Into);
	return Into;
}

//template<cudnnDataType_t InType = CUDNN_CROSS_CORRELATION, cudnnTensorFormat_t InFormat = CUDNN_TENSOR_NHWC>
void FVActCuDNN::_Unsafe_Create(FMLTensor& Into, int32 Width, int32 Height, int32 Size, int32 Count)
{
	Into.Format = CUDNN_TENSOR_NCHW;//CUDNN_TENSOR_NHWC;
	Into.Type = CUDNN_DATA_FLOAT;
	_CHECK_CUDNN(cudnnCreateTensorDescriptor(&Into.Desc));
	_CHECK_CUDNN(cudnnSetTensor4dDescriptor(Into.Desc, Into.Format, Into.Type, Count, Size, Height, Width));
	Into.SizeOf = (Count * Size * Width * Height) * sizeof(float);
	_CHECK_CUDA(cudaMallocManaged(&Into.Bytes, Into.SizeOf));
}

FMLTensor FVActCuDNN::CreateTensor(int32 Width, int32 Height, int32 Size, int32 Count)
{
	FMLTensor Into;
	_Unsafe_Create(Into, Width, Height, Size, Count);
	return Into;
}

//template<cudnnDataType_t InType = CUDNN_CROSS_CORRELATION, cudnnTensorFormat_t InFormat = CUDNN_TENSOR_NHWC>
// void FVActCuDNN::_Unsafe_Create(FMLTensor& Into, int32 N, const TArray<int32> Dims, const TArray<int32> Strids)
// {
// 	_CHECK_CUDNN(cudnnCreateTensorDescriptor(Into.Desc));
// 	_CHECK_CUDNN(cudnnSetTensorNdDescriptor(Into.Desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, N, *Dims, *Strids));
// 	Into.SizeOf = sizeof(float);
// 	for(int32 Index = 0; Index < N; ++Index) { Into.SizeOf *= Dims[Index]; }
// }

//template<cudnnDataType_t InType = CUDNN_CROSS_CORRELATION, cudnnTensorFormat_t InFormat = CUDNN_TENSOR_NHWC>
void FVActCuDNN::_Unsafe_Create(FMLKernel& Into, int32 Width, int32 Height, int32 SizeOut, int32 SizeIn)
{
	Into.Format = CUDNN_TENSOR_NCHW;//CUDNN_TENSOR_NHWC;
	Into.Type = CUDNN_DATA_FLOAT;
	_CHECK_CUDNN(cudnnCreateFilterDescriptor(&Into.Desc));
	_CHECK_CUDNN(cudnnSetFilter4dDescriptor(Into.Desc, Into.Type, Into.Format, SizeOut, SizeIn, Height, Width));
	Into.SizeOf = (SizeIn * SizeOut * Width * Height) * sizeof(float);
	_CHECK_CUDA(cudaMallocManaged(&Into.Bytes, Into.SizeOf));
}

FMLKernel FVActCuDNN::CreateKernel(int32 Width, int32 Height, int32 SizeOut, int32 SizeIn)
{
	FMLKernel Into;
	_Unsafe_Create(Into, Width, Height, SizeOut, SizeIn);
	return Into;
}

//template<cudnnDataType_t InType = CUDNN_CROSS_CORRELATION, cudnnTensorFormat_t InFormat = CUDNN_TENSOR_NHWC>
// void FVActCuDNN::_Unsafe_Create(FMLKernel& Into, int32 N, const TArray<int32> Dims)
// {
// 	_CHECK_CUDNN(cudnnCreateFilterDescriptor(Into.Desc));
// 	_CHECK_CUDNN(cudnnSetFilterNdDescriptor(Into.Desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, N, *Dims));
// 	Into.SizeOf = sizeof(float);
// 	for(int32 Index = 2; Index < N; ++Index) { Into.SizeOf *= Dims[Index]; }
// }

//template<cudnnDataType_t InType = CUDNN_CROSS_CORRELATION,cu cudnnConvolutionMode_t InMode = CUDNN_DATA_FLOAT>
void FVActCuDNN::_Unsafe_Create(FMLConvolution& Into, int32 PadWidth, int32 PadHeight, int32 StrideWidth, int32 StrideHeight, int32 DilationWidth, int32 DilationHeight)
{
	Into.Mode = CUDNN_CROSS_CORRELATION;
	Into.Type = CUDNN_DATA_FLOAT;
	_CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&Into.Desc));
	_CHECK_CUDNN(cudnnSetConvolution2dDescriptor(Into.Desc, PadHeight, PadWidth, StrideHeight, StrideWidth, DilationHeight, DilationWidth, Into.Mode, Into.Type));
}

FMLConvolution FVActCuDNN::CreateConvolution(int32 PadWidth, int32 PadHeight, int32 StrideWidth, int32 StrideHeight, int32 DilationWidth, int32 DilationHeight)
{
	FMLConvolution Into;
	_Unsafe_Create(Into, PadWidth, PadHeight, StrideWidth, StrideHeight, DilationHeight, DilationWidth);
	return Into;
}

//template<cudnnDataType_t InType = CUDNN_CROSS_CORRELATION, cudnnConvolutionMode_t InMode = CUDNN_DATA_FLOAT>
// void FVActCuDNN::_Unsafe_Create(FMLConvolution& Into, int32 N, const TArray<int32> Pads, const TArray<int32> Strids, const TArray<int32> Dilations)
// {
// 	_CHECK_CUDNN(cudnnCreateConvolotion(Into.Desc));
// 	_CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(Into.Desc, N, *Pads, *Strids, *Dilations, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
// }

void FVActCuDNN::_Unsafe_Create(FMLAlgorithm& Into, FMLContext& Context, FMLTensor& Input, FMLKernel& Kernel, FMLConvolution& Convolution, FMLTensor& Output, FMLResult& Result, int32 LimitCount, bool bAllowShringking)
{
	Into.Type = EMLAlgorithm::ConvFwd;
	Result.Type = EMLResult::ConvFwd;
	_CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(Context.Handle, &Result.MaxCount));
	_CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(Context.Handle, Input.Desc, Kernel.Desc, Convolution.Desc, Output.Desc, LimitCount, &Result.Count, &Result.ConvFwd));
}

FMLAlgorithm FVActCuDNN::CreateAlgorithm(FMLContext& Context, FMLTensor& Input, FMLKernel& Kernel, FMLConvolution& Convolution, FMLTensor& Output, FMLResult& Result, int32 Count, bool bAllowShringking)
{
	FMLAlgorithm Into;
	_Unsafe_Create(Into, Context, Input, Kernel, Convolution, Output, Result, Count, bAllowShringking);
	return Into;
}

void FVActCuDNN::_Unsafe_Create(FMLActivation& Into, FMLTensor& Input, FMLTensor& Output, float Alpha, float Beta, double Gamma)
{
	Into.Mode = CUDNN_ACTIVATION_SIGMOID;
	Into.Propagation = CUDNN_NOT_PROPAGATE_NAN;
	Into.Alpha[0] = Alpha;
	Into.Beta[0] = Beta;
	
	_CHECK_CUDNN(cudnnCreateActivationDescriptor(&Into.Desc));
	_CHECK_CUDNN(cudnnSetActivationDescriptor(Into.Desc, Into.Mode, Into.Propagation, Gamma));
}
	
FMLActivation FVActCuDNN::CreateActivation(FMLTensor& Input, FMLTensor& Output, float Alpha, float Beta, double Gamma)
{
	FMLActivation Into;
	_Unsafe_Create(Into, Input, Output, Alpha, Beta, Gamma);
	return Into;
}

void FVActCuDNN::_Unsafe_Create(FMLInstance& Into, FMLContext& Context, FMLTensor& Input, FMLKernel& Kernel, FMLConvolution& Convolution, FMLTensor& Output, FMLAlgorithm& Algorithm, TArray<float> KernelData, int32 DefaultByte)
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

FMLInstance FVActCuDNN::CreateInstance(FMLContext& Context, FMLTensor& Input, FMLKernel& Kernel, FMLConvolution& Convolution, FMLTensor& Output, FMLAlgorithm& Algorithm, TArray<float> KernelData, int32 DefaultByte)
{
	FMLInstance Into;
	_Unsafe_Create(Into, Context, Input, Kernel, Convolution, Output, Algorithm, KernelData, DefaultByte);
	return Into;
}

void FVActCuDNN::_Unsafe_Use(const FMLDevice& Device)
{
	_CHECK_CUDA(cudaSetDevice(Device.Index));
}

void FVActCuDNN::Use(const FMLDevice& Device)
{
	_Unsafe_Use(Device);
}

void FVActCuDNN::_Unsafe_Forward(const FMLContext& Context, FMLActivation& Activation, FMLTensor& Input, FMLTensor& Output)
{
	_CHECK_CUDNN(cudnnActivationForward(Context.Handle, Activation.Desc, Activation.Alpha, Input.Desc, Input.Bytes, Activation.Beta, Output.Desc, Output.Bytes));
}

void FVActCuDNN::Forward(const FMLContext& Context, FMLActivation& Activation, FMLTensor& Input, FMLTensor& Output)
{
	_Unsafe_Forward(Context, Activation, Input, Output);
}

// void FVActCuDNN::_Unsafe_Use(FMLInstance& Instance)
// {
// 	_CHECK_CUDNN(cudnnConvolutionForward(Instance.Context, Alpha, Instance.Input, Instance.InputData, Instance.Kernel, Instance.Kernel, Instance.KernelData, Instance.Convolution, Instance.Algorithm, Instance.Bytes, Instance.SizeOf, Beta, Instance.Output, Instance.OutputData));
// }

// FMLInstance& FVActCuDNN::Use(FMLInstance& Instance)
// {
// 	_Unsafe_Use(Instance);
// 	return Instance;
// }

// void FVActCuDNN::_Unsafe_Create(FMLInstance& Into, FMLContext& Context, TArray<float> InputData, TArray<float> OutputData, FMLConvolution& Convolution, FMLAlgorithm& Algorithm, int32 Size, int32 Count = 1)
// {
// 	FMLTensor Input, Output;
// 	_Unsafe_Create(Input, )
// }

void FVActCuDNN::_Unsafe_Destroy(FMLKernel& Kernel, bool bData)
{
	if (bData) { _CHECK_CUDA(cudaFree(Kernel.Bytes)); Kernel.Bytes = nullptr; }
	_CHECK_CUDNN(cudnnDestroyFilterDescriptor(Kernel.Desc));
}

void FVActCuDNN::Destroy(FMLKernel& Kernel, bool bData)
{
	_Unsafe_Destroy(Kernel, bData);
}

void FVActCuDNN::_Unsafe_Destroy(FMLTensor& Tensor, bool bData)
{
	if (bData) { _CHECK_CUDA(cudaFree(Tensor.Bytes)); Tensor.Bytes = nullptr; }
	_CHECK_CUDNN(cudnnDestroyTensorDescriptor(Tensor.Desc));
}

void FVActCuDNN::Destroy(FMLTensor& Tensor, bool bData)
{
	_Unsafe_Destroy(Tensor, bData);
}

void FVActCuDNN::_Unsafe_Destroy(FMLConvolution& Convolution, bool bData)
{
	_CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(Convolution.Desc));
}

void FVActCuDNN::Destroy(FMLConvolution& Convolution, bool bData)
{
	_Unsafe_Destroy(Convolution, bData);
}

void FVActCuDNN::_Unsafe_Destroy(FMLAlgorithm& Algorithm, bool bData)
{
	_CHECK_CUDNN(cudnnDestroyAlgorithmDescriptor(Algorithm.Desc));
}

void FVActCuDNN::Destroy(FMLAlgorithm& Algorithm, bool bData)
{
	_Unsafe_Destroy(Algorithm, bData);
}

// void FVActCuDNN::_Unsafe_Destroy(FMLWorkspace& Workspace, bool bData)
// {
// 	if (bData) { _CHECK_CUDA(cudaFree(Workspace.Bytes)); Workspace.Bytes = nullptr; }
// }

// void FVActCuDNN::Destroy(FMLWorkspace& Workspace, bool bData)
// {
// 	_Unsafe_Destroy(Workspace, bData);
// }

void FVActCuDNN::_Unsafe_Destroy(FMLContext& Context, bool bData)
{
	_CHECK_CUDNN(cudnnDestroy(Context.Handle));
}

void FVActCuDNN::Destroy(FMLContext& Context, bool bData)
{
	_Unsafe_Destroy(Context, bData);
}

// static void FVActCuDNN::_Unsafe_Destroy(FMLInstance& Instance, bool bData, bool bContext)
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

// static void FVActCuDNN::_Unsafe_Destroy(FMLInstance& Instance, bool bData, bool bContext)
// {
// 	_Unsafe_Destory(Instance, bData, bContext);
// }

void FVActCuDNN::_Unsafe_Destroy(FMLDevice& Device, bool bData)
{
}

void FVActCuDNN::Destroy(FMLDevice& Device, bool bData)
{
	_Unsafe_Destroy(Device, bData);
}


void FVActCuDNN::_Unsafe_Destroy(FMLResult& Result, bool bData)
{
	//if (bData){ _CHECK_CUDNN(cudnnDestroyAlgorithmPerformance(Result.ConvFwd, Result.Count)); }
}

void FVActCuDNN::Destroy(FMLResult& Result, bool bData)
{
	_Unsafe_Destroy(Result, bData);
}

void FVActCuDNN::_Unsafe_Destroy(FMLActivation& Activation, bool bData)
{
	_CHECK_CUDNN(cudnnDestroyActivationDescriptor(Activation.Desc));
}

void FVActCuDNN::Destroy(FMLActivation& Activation, bool bData)
{
	_Unsafe_Destroy(Activation, bData);
}


//void FVActCuDNN::_Unsafe_Save(const FMLFormatSafetensorHeader& Model, EMLFormatSafetensor Format)
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


//void FVActCuDNN::Save(const FMLFormatSafetensorHeader& Model, EMLFormatSafetensor Format)
//{
//	_Unsafe_Save(Activation, bData);
//}
//
//void FVActCuDNN::_Unsafe_Load(FMLFormatSafetensorHeader& Into, EMLFormatSafetensor Format)
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
//void FVActCuDNN::Load(FMLFormatSafetensorHeader& Into, EMLFormatSafetensor Format)
//{
//	_Unsafe_Load(Activation, bData);
//}