#include "VActML.h"

void FVActML::_Unsafe_Create(FMLContext& Into)
{
	
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

void FVActML::_Unsafe_Create(FMLActivation& Into, FMLTensor& Input, FMLTensor& Output, float Alpha, float Beta, double Gamma)
{

}
	
FMLActivation FVActML::CreateActivation(FMLTensor& Input, FMLTensor& Output, float Alpha, float Beta, double Gamma)
{
	FMLActivation Into;
	_Unsafe_Create(Into, Input, Output, Alpha, Beta, Gamma);
	return Into;
}

void FVActML::_Unsafe_Forward(const FMLContext& Context, FMLActivation& Activation, FMLTensor& Input, FMLTensor& Output)
{

}

void FVActML::Forward(const FMLContext& Context, FMLActivation& Activation, FMLTensor& Input, FMLTensor& Output)
{
	_Unsafe_Forward(Context, Activation, Input, Output);
}

void FVActML::_Unsafe_Destroy(FMLTensor& Tensor, bool bData)
{

}

void FVActML::Destroy(FMLTensor& Tensor, bool bData)
{
	_Unsafe_Destroy(Tensor, bData);
}

void FVActML::_Unsafe_Destroy(FMLContext& Context, bool bData)
{
	_CHECK_CUDNN(cudnnDestroy(Context.Handle));
}

void FVActML::Destroy(FMLContext& Context, bool bData)
{
	_Unsafe_Destroy(Context, bData);
}

void FVActML::_Unsafe_Destroy(FMLActivation& Activation, bool bData)
{
	_CHECK_CUDNN(cudnnDestroyActivationDescriptor(Activation.Desc));
}

void FVActML::Destroy(FMLActivation& Activation, bool bData)
{
	_Unsafe_Destroy(Activation, bData);
}