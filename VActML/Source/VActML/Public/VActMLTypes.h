#pragma once

#include "CoreMinimal.h"
#include "VActMLTypes.generated.h"

UENUM()
enum class EMLType
{
    Float = 0, //F32
    Double = 1, //F64
    Half = 2, //F16
    Int8 = 3,
    Int32 = 4,
    Int8x4 = 5,
    UInt8 = 6,
    UInt8x4 = 7,
    Int8x32 = 8,
    BFloat16 = 9,
    Int64 = 10,
    Boolean = 11,
    FP8_E4M3 = 12,
    FP8_E5M2 = 13,
    Fast_Float_For_FP8 = 14,
    Int16 = 15,
    UInt16 = 16,
    UInt64 = 17,
    Bits_BE = 18,
    Bits_LE = 19,
    Bits = Bits_LE,
    Unknown = 20
};

USTRUCT()
struct VACTML_API FMLContext
{
    GENERATED_BODY()
    
};

USTRUCT()
struct VACTML_API FMLShape
{
    GENERATED_BODY()
    
};

USTRUCT()
struct VACTML_API FMLEmbed
{
    GENERATED_BODY()

};

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
    F64 = EMLType::Double,
    F32 = EMLType::Float,
    F16 = EMLType::Half,
    BF16 = EMLType::BFloat16,
    I64 = EMLType::Int64,
    I32 = EMLType::Int32,
    I16 = EMLType::Int16,
    I8 = EMLType::Int8,
    U8 = EMLType::UInt8,
    BOOL = EMLType::Boolean
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
