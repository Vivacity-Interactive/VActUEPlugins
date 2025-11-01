#pragma once

#define _VACT_SIZE_T SIZE_T

#include "VActSTTTypes.h"
#include "CoreMinimal.h"

#include "VActSTT.generated.h"

struct FSTTModel;

USTRUCT()
struct FVActSTT
{
	GENERATED_BODY()

	const static TMap<ESTTModelMode, uint32> ModeMap;

	FORCEINLINE static void LoadModel(FSTTModel& Into, const FString& Path, bool bForceLoad = false)
	{
		const bool bLoad = !Into.bLoaded || Into.Context == nullptr || bForceLoad;
		if (bLoad) { _Unsafe_LoadModel(Into, Path); }
	}

	FORCEINLINE static void LoadModel(FSTTModel& Into, const FFilePath& FilePath)
	{
		_Unsafe_LoadModel(Into, FilePath);
	}

	FORCEINLINE static void UnloadModel(FSTTModel& Model, bool bFree = true)
	{
		if (Model.Context == nullptr) { return; }
		_Unsafe_UnloadModel(Model, bFree);
	}

	FORCEINLINE static void UseModel(bool& Into, int32& SegmentCount, const FSTTModel& Model, const TArray<float>& Buffer, const FSTTModelUseSettings& UseSettings = FSTTModelUseSettings::DefaultUseSettings)
	{
		if (Model.Context == nullptr) { Into = false; SegmentCount = 0; return; }
		_Unsafe_UseModel(Into, SegmentCount, Model, Buffer, UseSettings);
	}

	FORCEINLINE static void PopulateToken(FSTTToken& Into, const FSTTModel& Model, int32 SegmentId, int32 TokenIndex)
	{
		if (Model.Context == nullptr) { Into.Id = -1; Into.Index = -1; return; }
		_Unsafe_PopulateToken(Into, Model, SegmentId, TokenIndex);
	}

	FORCEINLINE static void PopulateToken(FSTTToken& Into, int32& TokenCount, const FSTTModel& Model, int32 SegmentId)
	{
		if (Model.Context == nullptr) { TokenCount = 0; Into.Id = -1; Into.Index = -1; return; }
		_Unsafe_PopulateToken(Into, TokenCount, Model, SegmentId);
	}

	FORCEINLINE static bool IsSpecialToken(bool& Into, const FSTTModel& Model, FSTTToken& TokenId)
	{
		if (Model.Context == nullptr) { return false; }
		_Unsafe_IsSpecialToken(Into, Model, TokenId.Id);
		return Into;
	}

	FORCEINLINE static bool IsSpecialToken(const FSTTModel& Model, FSTTToken& TokenId)
	{
		if (Model.Context == nullptr) { return false; }
		bool Into = false;
		_Unsafe_IsSpecialToken(Into, Model, TokenId.Id);
		return Into;
	}

	FORCEINLINE static int32 TokenId(int32& Into, const FSTTModel& Model, int32 SegmentId, int32 TokenIndex)
	{
		if (Model.Context == nullptr) { return -1; }
		_Unsafe_TokenId(Into, Model, SegmentId, TokenIndex);
		return Into;
	}

	FORCEINLINE static int32 TokenId(const FSTTModel& Model, int32 SegmentId, int32 TokenIndex)
	{
		if (Model.Context == nullptr) { return  -1;; }
		int32 Into = -1;
		_Unsafe_TokenId(Into, Model, SegmentId, TokenIndex);
		return Into;
	}

	FORCEINLINE static void _Unsafe_LoadModel(FSTTModel& Into, const FFilePath& FilePath)
	{
		_Unsafe_LoadModel(Into, FPaths::ProjectContentDir() / FilePath.FilePath);
	}

	FORCEINLINE static void _Unsafe_LoadModel(FSTTModel& Into, TArray<uint8>& Raw)
	{
		_Unsafe_LoadModel(Into, Raw.GetData(), Raw.Num());
	}

	FORCEINLINE static void _Unsafe_IsSpecialToken(bool& Into, const FSTTModel& Model, FSTTToken& TokenId)
	{
		_Unsafe_IsSpecialToken(Into, Model, TokenId.Id);
	}

	FORCEINLINE static int32 _Unsafe_Resample(const float* InAudio, int32 FromRate, float* OutAudio, int32 ToRate, int32 NumSamples)
	{
		const double Ratio = static_cast<double>(FromRate) / static_cast<double>(ToRate);
		return _Unsafe_Resample(InAudio, OutAudio, Ratio, NumSamples);

	}

	FORCEINLINE static int32 _Unsafe_Upsample(const float* InAudio, int32 FromRate, float* OutAudio, int32 ToRate, int32 NumSamples)
	{
		const double Ratio = static_cast<double>(FromRate) / static_cast<double>(ToRate);
		return _Unsafe_Upsample(InAudio, OutAudio, Ratio, NumSamples);
	}

	FORCEINLINE static int32 _Unsafe_Downsample(const float* InAudio, int32 FromRate, float* OutAudio, int32 ToRate, int32 NumSamples)
	{
		const double Ratio = static_cast<double>(FromRate) / static_cast<double>(ToRate);
		return _Unsafe_Downsample(InAudio, OutAudio, Ratio, NumSamples);
	}

	static void _Unsafe_LoadModel(FSTTModel& Into, const FString& Path);

	static void _Unsafe_LoadModel(FSTTModel& Into, uint8* Raw, SIZE_T Count);

	static void _Unsafe_UnloadModel(FSTTModel& Model, bool bFree = true);

	static void _Unsafe_UseModel(bool& Into, int32& SegmentCount, const FSTTModel& Model, const TArray<float>& Buffer, const FSTTModelUseSettings& UseSettings = FSTTModelUseSettings::DefaultUseSettings);

	static void _Unsafe_PopulateToken(FSTTToken& Into, const FSTTModel& Model, int32 SegmentId, int32 TokenIndex);

	static void _Unsafe_PopulateToken(FSTTToken& Into, int32& TokenCount, const FSTTModel& Model, int32 SegmentId);

	static void _Unsafe_IsSpecialToken(bool& Into, const FSTTModel& Model, int32 TokenId);

	static void _Unsafe_TokenId(int32& Into, const FSTTModel& Model, int32 SegmentId, int32 TokenIndex);

	static int32 _Unsafe_Resample(const float* InAudio, float* OutAudio, double Ratio, int32 NumSamples);

	static int32 _Unsafe_Upsample(const float* InAudio, float* OutAudio, double Ratio, int32 NumSamples);

	static int32 _Unsafe_Downsample(const float* InAudio, float* OutAudio, double Ratio, int32 NumSamples);

	static int32 _Unsafe_ToMonoAvrage(const float* InAudio, float* OutAudio, int32 NumChannels, int32 NumSamples);

	static int32 _Unsafe_ToMonoCopy(const float* InAudio, int32 ChannelId, float* OutAudio, int32 NumChannels, int32 NumSamples);

	static int32 _Unsafe_ToMultiCopy(const float* InAudio, float* OutAudio, int32 NumChannels, int32 NumSamples);

};
