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

	const static TMap<ESTTAudioFormat, uint32> FormatMap;

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

	FORCEINLINE static void UseModel(bool& Into, int32& SegmentCount, const FSTTModel& Model, const TArray<float>& Buffer, int32 Count, TArray<int32>& PromptBuffer, int32 PromptCount, const FSTTModelUseSettings& UseSettings = FSTTModelUseSettings::DefaultUseSettings)
	{
		const bool bUse = Model.Context != nullptr && Count > 0 && Count < Buffer.Num() && PromptCount >= 0 && PromptCount <= PromptBuffer.Num();
		if (!bUse) { Into = false; SegmentCount = 0; return; }
		_Unsafe_UseModel(Into, SegmentCount, Model, Buffer.GetData(), Count, UseSettings, PromptBuffer.GetData(), PromptCount);
	}

	FORCEINLINE static void UseModel(bool& Into, int32& SegmentCount, const FSTTModel& Model, const TArray<float>& Buffer, TArray<int32>& PromptBuffer, int32 PromptCount, const FSTTModelUseSettings& UseSettings = FSTTModelUseSettings::DefaultUseSettings)
	{
		const bool bUse = Model.Context == nullptr  && PromptCount >= 0 && PromptCount <= PromptBuffer.Num();
		if (bUse) { Into = false; SegmentCount = 0; return; }
		_Unsafe_UseModel(Into, SegmentCount, Model, Buffer.GetData(), Buffer.Num(), UseSettings, PromptBuffer.GetData(), PromptCount);
	}

	FORCEINLINE static void UseModel(bool& Into, int32& SegmentCount, const FSTTModel& Model, const TArray<float>& Buffer, int32 Count, TArray<int32>& PromptBuffer, const FSTTModelUseSettings& UseSettings = FSTTModelUseSettings::DefaultUseSettings)
	{
		const bool bUse = Model.Context != nullptr && Count > 0 && Count < Buffer.Num();
		if (!bUse) { Into = false; SegmentCount = 0; return; }
		_Unsafe_UseModel(Into, SegmentCount, Model, Buffer.GetData(), Count, UseSettings, PromptBuffer.GetData(), PromptBuffer.Num());
	}

	FORCEINLINE static void UseModel(bool& Into, int32& SegmentCount, const FSTTModel& Model, const TArray<float>& Buffer, TArray<int32>& PromptBuffer, const FSTTModelUseSettings& UseSettings = FSTTModelUseSettings::DefaultUseSettings)
	{
		if (Model.Context == nullptr) { Into = false; SegmentCount = 0; return; }
		_Unsafe_UseModel(Into, SegmentCount, Model, Buffer.GetData(), Buffer.Num(), UseSettings, PromptBuffer.GetData(), PromptBuffer.Num());
	}

	FORCEINLINE static void UseModel(bool& Into, int32& SegmentCount, const FSTTModel& Model, const TArray<float>& Buffer, int32 Count, const FSTTModelUseSettings& UseSettings = FSTTModelUseSettings::DefaultUseSettings)
	{
		const bool bUse = Model.Context != nullptr && Count > 0 && Count < Buffer.Num();
		if (!bUse) { Into = false; SegmentCount = 0; return; }
		_Unsafe_UseModel(Into, SegmentCount, Model, Buffer.GetData(), Count, UseSettings);
	}

	FORCEINLINE static void UseModel(bool& Into, int32& SegmentCount, const FSTTModel& Model, const TArray<float>& Buffer, const FSTTModelUseSettings& UseSettings = FSTTModelUseSettings::DefaultUseSettings)
	{
		if (Model.Context == nullptr) { Into = false; SegmentCount = 0; return; }
		_Unsafe_UseModel(Into, SegmentCount, Model, Buffer.GetData(), Buffer.Num(), UseSettings);
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

	FORCEINLINE static void TokenTime(int64& T0, int64& T1, const FSTTModel& Model, int32 SegmentId, int32 TokenIndex)
	{
		if (Model.Context == nullptr) { T0 = T1 = 0u; return; }
		_Unsafe_TokenTime(T0, T1, Model, SegmentId, TokenIndex);
	}

	FORCEINLINE static void SegmentTime(int64& T0, int64& T1, const FSTTModel& Model, int32 SegmentId)
	{
		if (Model.Context == nullptr) { T0 = T1 = 0; return; }
		_Unsafe_SegmentTime(T0, T1, Model, SegmentId);
	}

	FORCEINLINE static void SegmentText(FString& Into, const FSTTModel& Model, int32 SegmentId)
	{
		if (Model.Context == nullptr) { Into = ""; return; }
		_Unsafe_SegmentText(Into, Model, SegmentId);
	}

	FORCEINLINE static void SegmentText(FString& Into, const FSTTModel& Model, int32 SegmentId, int32 SegmentTextOffset)
	{
		if (Model.Context == nullptr) { Into = ""; return; }
		_Unsafe_SegmentText(Into, Model, SegmentId);
		Into = Into.Mid(SegmentTextOffset);
	}
	
	FORCEINLINE static void SegmentTextAppend(FString& Into, const FSTTModel& Model, int32 SegmentId)
	{
		if (Model.Context == nullptr) { Into = ""; return; }
		_Unsafe_SegmentTextAppend(Into, Model, SegmentId);
	}

	FORCEINLINE static void SegmentTextAppend(FString& Into, const FSTTModel& Model, int32 SegmentId, int32 SegmentTextOffset)
	{
		if (Model.Context == nullptr) { Into = ""; return; }
		_Unsafe_SegmentTextAppend(Into, Model, SegmentId);
		Into = Into.Mid(SegmentTextOffset);
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

	FORCEINLINE static void HighPassFilter(TArray<float>& InAudio, TArray<float>& OutAudio, float Treshold, int32 SampleRate)
	{
		bool bFilter = OutAudio.Num() <= InAudio.Num();
		if (bFilter) { _Unsafe_HighPassFilter(InAudio.GetData(), OutAudio.GetData(), OutAudio.Num(), Treshold, SampleRate); }
	}

	FORCEINLINE static void HighPassFilter(TArray<float>& InAudio, TArray<float>& OutAudio, int32 NumSamples, float Treshold, int32 SampleRate)
	{
		bool bFilter = OutAudio.Num() >= NumSamples && InAudio.Num() >= NumSamples;
		if (bFilter) { _Unsafe_HighPassFilter(InAudio.GetData(), OutAudio.GetData(), NumSamples, Treshold, SampleRate); }
	}

	FORCEINLINE static bool VAD(const TArray<float>& InAudio, TArray<float>& OutAudio, int32 SampleRate, int32 Last, float Treshold, float FequencyTreshold)
	{
		const bool bVAD = OutAudio.Num() <= InAudio.Num();
		return bVAD && _Unsafe_VAD(InAudio.GetData(), OutAudio.GetData(), OutAudio.Num(), SampleRate, Last, Treshold, FequencyTreshold);;
	}

	FORCEINLINE static bool VAD(const TArray<float>& InAudio, TArray<float>& OutAudio, int32 NumSamples, int32 SampleRate, int32 Last, float Treshold, float FequencyTreshold)
	{
		const bool bVAD = OutAudio.Num() >= NumSamples && InAudio.Num() >= NumSamples;
		return bVAD && _Unsafe_VAD(InAudio.GetData(), OutAudio.GetData(), NumSamples, SampleRate, Last, Treshold, FequencyTreshold);
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

	static void _Unsafe_LoadModel(FSTTModel& Into, const FString& Path);

	static void _Unsafe_LoadModel(FSTTModel& Into, uint8* Raw, SIZE_T Count);

	static void _Unsafe_UnloadModel(FSTTModel& Model, bool bFree = true);

	static void _Unsafe_UseModel(bool& Into, int32& SegmentCount, const FSTTModel& Model, const float* Buffer, int32 Count, const FSTTModelUseSettings& UseSettings = FSTTModelUseSettings::DefaultUseSettings, int32* PromptBuffer = nullptr, int32 PromptCount = 0);

	static void _Unsafe_PopulateToken(FSTTToken& Into, const FSTTModel& Model, int32 SegmentId, int32 TokenIndex);

	static void _Unsafe_PopulateToken(FSTTToken& Into, int32& TokenCount, const FSTTModel& Model, int32 SegmentId);

	static void _Unsafe_SegmentTime(int64& T0, int64& T1, const FSTTModel& Model, int32 SegmentId);

	static void _Unsafe_SegmentTextAppend(FString& Into, const FSTTModel& Model, int32 SegmentId);

	static void _Unsafe_SegmentTextAppend(FString& Into, const FSTTModel& Model, int32 SegmentId, int32 SegmentTextOffset);

	static void _Unsafe_SegmentText(FString& Into, const FSTTModel& Model, int32 SegmentId);

	static void _Unsafe_SegmentText(FString& Into, const FSTTModel& Model, int32 SegmentId, int32 SegmentTextOffset);

	static void _Unsafe_IsSpecialToken(bool& Into, const FSTTModel& Model, int32 TokenId);

	static void _Unsafe_TokenId(int32& Into, const FSTTModel& Model, int32 SegmentId, int32 TokenIndex);

	static void _Unsafe_TokenTime(int64& T0, int64& T1, const FSTTModel& Model, int32 SegmentId, int32 TokenIndex);

	static void _Unsafe_HighPassFilter(const float* InAudio, float* OutAudio, int32 NumSamples, float Treshold, int32 SampleRate);

	static bool _Unsafe_VAD(const float* InAudio, float* OutAudio, int32 NumSamples, int32 SampleRate, int32 Last, float Treshold, float FequencyTreshold);

};
