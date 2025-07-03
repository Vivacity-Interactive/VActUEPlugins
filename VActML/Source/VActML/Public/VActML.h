#pragma once

#define _VACT_SIZE_T SIZE_T

#include "VActMLTypes.h"

#include "CoreMinimal.h"
#include "VActML.generated.h"

struct FMLTensor;
//struct FMLWorkspace;
//struct FMLPooling;
struct FMLActivation;
//struct FMLTransformer;
//struct FMLReduce;
//struct FMLLoss;
//struct FMLTransform;

USTRUCT()
struct FVActML
{
	GENERATED_BODY()

	static void _Unsafe_Create(FMLContext& Into);

	static FMLContext CreateContext();
	
	//static void _Unsafe_Create(FMLTensor& Into, int32 Width, int32 Size, int32 Count = 1);

	static void _Unsafe_Create(FMLTensor& Into, int32 Width, int32 Height, int32 Size, int32 Count = 1);
	
	static FMLTensor CreateTensor(int32 Width, int32 Height, int32 Size, int32 Count = 1);

	//static void _Unsafe_Create(FMLTensor& Into, int32 N, const TArray<int32> Dims, const TArray<int32> Strids);

	//static void _Unsafe_Create(FMLKernel& Into, int32 Width, int32 SizeOut, int32 SizeIn);

	static void _Unsafe_Forward(const FMLContext& Context, FMLActivation& Activation, FMLTensor& Input, FMLTensor& Output);

	static void Forward(const FMLContext& Context, FMLActivation& Activation, FMLTensor& Input, FMLTensor& Output);

	static void _Unsafe_Destroy(FMLContext& Context, bool bData = true);

	static void Destroy(FMLContext& Context, bool bData = true);

};