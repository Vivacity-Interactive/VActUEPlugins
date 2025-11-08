#pragma once

#include "CoreMinimal.h"
#include "VACTSTTTypes.generated.h"

struct FSTTModelUseSettings;

#define VACT_STT_UNIT_HZ 1e-3f

UENUM(BlueprintType)
enum class ESTTModelMode : uint8
{
	None UMETA(Hidden),
	Greedy,
	BeamSearch,
};

UENUM(BlueprintType)
enum class ESTTAudioFormat : uint8
{
	UNKNOWN,
	PCM_8,
	PCM_16,
	PCM_24,
	PCM_24_IN_32,
	PCM_32,
	FLOATING_POINT_32,
	FLOATING_POINT_64
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTSTT_API FSTTCaptureDevice
{
	GENERATED_BODY()

	FSTTCaptureDevice();

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FString DeviceName;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	FString DeviceId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 Index;
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTSTT_API FSTTModelUseSettings
{
	GENERATED_BODY()

	static const FSTTModelUseSettings DefaultUseSettings;

	FSTTModelUseSettings();

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	uint8 bTranslate : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	uint8 bSpecialTokens : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	uint8 bNoFallback : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	uint8 bNoContext : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	uint8 bNoTimeStamp : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	uint8 bSingleSegment : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	uint8 bTinyDiarize : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	uint8 bUseGPU : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	uint8 bFlashAttention : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 MaxTokens;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 BeamSize;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 ThreadCount;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 AudioContext;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FString Language;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	ESTTModelMode Mode;
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTSTT_API FSTTModel
{
	GENERATED_BODY()

	FSTTModel();

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	uint8 bLoaded : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	uint8 bGPU : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	uint8 bFlashAttention : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 SampleRate;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 ChannelCount;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 UnitDuration;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 BatchSampleCount;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	double UnitTimeScale;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	ESTTAudioFormat AudioFormat;

	void* Context;
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTSTT_API FSTTToken
{
	GENERATED_BODY()

	FSTTToken();

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int64 CustomId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 ContextId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 SegmentId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 Id;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32 Index;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Probability;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FString Text;
};