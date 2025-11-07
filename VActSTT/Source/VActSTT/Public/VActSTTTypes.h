#pragma once

#include "CoreMinimal.h"
#include "VACTSTTTypes.generated.h"

//template<typename ElementType, template<typename> class HandlerType>
//struct FIteratorHandler;

template<typename ElementType>
struct FBufferHandler;

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

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
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
	int64  TimeStamp;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32  ContextId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32  SegmentId;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32  Id;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int32  Index;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Duration;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Probability;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FString Text;
};

template<typename ElementType>
struct VACTSTT_API FBufferHandler
{
	int32 Head;

	int32 Size;

	TArray<ElementType>* Array;

	FBufferHandler()
		: Head(0)
		, Size(0)
		, Array(nullptr)
	{
	}

	FBufferHandler(TArray<ElementType>& InArray)
		: Head(0)
		, Size(InArray.Num())
		, Array(&InArray)
	{
	}

	void Add(ElementType&& Item)
	{
		(*Array)[Head] = MoveTemp(Item);
		Head = ++Head % (*Array).Num();
		Size = FMath::Min(++Size, (*Array).Num());
	}

	void Add(ElementType& Item)
	{
		(*Array)[Head] = Item;
		Head = ++Head % (*Array).Num();
		Size = FMath::Min(++Size, (*Array).Num());
	}

	void Reset(bool bResetArray = false)
	{
		Head = 0;
		Size = 0;
		if (bResetArray) { (*Array).Reset(); }
	}

	template<typename Func>
	void ForEach(Func&& Callback) const
	{
		for (int32 Index = 0; Index < Size; ++Index)
		{
			const int32 _Index = BufferIndex(Index);
			Callback((*Array)[_Index]);
		}
	}

	constexpr FORCEINLINE int32 BufferIndex(int32 Index) const { return (Head + (*Array).Num() - Size + Index) % (*Array).Num(); }

	constexpr FORCEINLINE int32 Num() const { return Size; }

	FORCEINLINE ElementType& operator[](int32 Index)
	{
		const int32 _Index = BufferIndex(Index);
		return (*Array)[_Index];
	}

	FORCEINLINE const ElementType& operator[](int32 Index) const
	{
		const int32 _Index = BufferIndex(Index);
		return (*Array)[_Index];
	}

	/*FBufferHandler<ElementType> begin()
	{
		return FIteratorHandler(*this, 0);
	}

	FBufferHandler<ElementType> end()
	{
		return FIteratorHandler(*this, Size);
	}*/

};


//template<typename ElementType, template<typename> class HandlerType>
//struct FIteratorHandler
//{
//	int32 Index;
//
//	HandlerType<ElementType>& Handler;
//
//	FIteratorHandler(HandlerType<ElementType>& InHandler)
//		: Index(0)
//		, Handler(InHandler)
//	{
//	}
//
//	operator bool() const { return Index < Handler.Num(); }
//
//	bool operator !=(const FIteratorHandler& Other) const
//	{
//		return Index != Other.Index;
//	}
//
//	FIteratorHandler& operator++()
//	{
//		++Index;
//		return *this;
//	}
//
//	FIteratorHandler& operator--()
//	{
//		--Index;
//		return *this;
//	}
//
//	ElementType& operator*()
//	{
//		int32 _Index = Handler.BufferIndex(Index);
//		return (*Handler.Array)[_Index];
//	}
//
//	ElementType* operator->()
//	{
//		int32 _Index = Handler.BufferIndex(Index);
//		return &(*Handler.Array)[_Index];
//	}
//};
//
//template<typename ElementType, template<typename> class HandlerType>
//FIteratorHandler(HandlerType<ElementType>&) -> FIteratorHandler<ElementType, HandlerType>;