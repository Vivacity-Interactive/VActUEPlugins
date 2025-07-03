#pragma once

#include "CoreMinimal.h"
#include "VActMLTypes.generated.h"

USTRUCT()
struct VACTML_API FMLContext
{
    GENERATED_BODY()
    
};

USTRUCT()
struct VACTML_API FMLShape
{
    uint32
}

USTRUCT()
struct VACTML_API FMLEmbed
{

}

USTRUCT()
struct VACTML_API FMLTensor
{
    GENERATED_BODY()

};

USTRUCT()
struct VACTML_API FMLActivation
{
    GENERATED_BODY()

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
