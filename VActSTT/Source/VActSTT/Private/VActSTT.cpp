#include "VActSTT.h"

#include "whisper.h"

const TMap<ESTTModelMode, uint32> FVActSTT::ModeMap = {
	{ ESTTModelMode::Greedy, WHISPER_SAMPLING_GREEDY},
	{ ESTTModelMode::BeamSearch, WHISPER_SAMPLING_BEAM_SEARCH },
};

void FVActSTT::_Unsafe_LoadModel(FSTTModel& Into, const FString& Path)
{
	struct whisper_context_params cparams = whisper_context_default_params();
	cparams.use_gpu = Into.bGPU;
	cparams.flash_attn = Into.bFlashAttention;

	Into.Context = whisper_init_from_file_with_params(TCHAR_TO_ANSI(*Path), cparams);
	Into.bLoaded = Into.Context != nullptr;
}

void FVActSTT::_Unsafe_LoadModel(FSTTModel& Into, uint8* Raw, SIZE_T Count)
{
	struct whisper_context_params cparams = whisper_context_default_params();
	cparams.use_gpu = Into.bGPU;
	cparams.flash_attn = Into.bFlashAttention;

	Into.Context = whisper_init_from_buffer_with_params((void*)Raw, sizeof(uint8) * Count, cparams);
	Into.bLoaded = Into.Context != nullptr;
}

void FVActSTT::_Unsafe_UnloadModel(FSTTModel& Model, bool bFree)
{
	if (bFree) { whisper_free(reinterpret_cast<whisper_context*>(Model.Context)); }
	
	Model.bLoaded = false;
	Model.Context = nullptr;
}

void FVActSTT::_Unsafe_UseModel(bool& Into, int32& SegmentCount, const FSTTModel& Model, const TArray<float>& Buffer, const FSTTModelUseSettings& UseSettings)
{
	whisper_context* Context = reinterpret_cast<whisper_context*>(Model.Context);

	whisper_full_params params = whisper_full_default_params((enum whisper_sampling_strategy)ModeMap[UseSettings.Mode]);
	params.print_progress = false;
	params.print_special = false;
	params.print_timestamps = false;
	params.print_realtime = false;
	params.translate = UseSettings.bTranslate;
	params.language = TCHAR_TO_ANSI(*UseSettings.Language);

	Into = whisper_full(Context, params, Buffer.GetData(), Buffer.Num()) == 0;

	if (Into) { SegmentCount = whisper_full_n_segments(Context); }
}

void FVActSTT::_Unsafe_PopulateToken(FSTTToken& Into, const FSTTModel& Model, int32 SegmentId, int32 TokenIndex)
{
	whisper_context* Context = reinterpret_cast<whisper_context*>(Model.Context);

	Into.Id = whisper_full_get_token_id(Context, SegmentId, TokenIndex);
	Into.Index = TokenIndex;
	Into.Probability = whisper_full_get_token_p(Context, SegmentId, TokenIndex);
	const char* text = whisper_full_get_token_text(Context, SegmentId, TokenIndex);
	Into.Text = ANSI_TO_TCHAR(text);
}

void FVActSTT::_Unsafe_PopulateToken(FSTTToken& Into, int32& TokenCount, const FSTTModel& Model, int32 SegmentId)
{
	whisper_context* Context = reinterpret_cast<whisper_context*>(Model.Context);

	Into.SegmentId = SegmentId;
	//Into.TimeStamp = whisper_full_get_segment_t0(Context, SegmentId);
	//Into.Duration = whisper_full_get_segment_t1(Context, SegmentId) - Into.TimeStamp;

	TokenCount = whisper_full_n_tokens(Context, SegmentId);
}

void FVActSTT::_Unsafe_IsSpecialToken(bool& Into, const FSTTModel& Model, int32 TokenId)
{
	whisper_context* Context = reinterpret_cast<whisper_context*>(Model.Context);
	
	Into = TokenId >= whisper_token_eot(Context);
}

void FVActSTT::_Unsafe_TokenId(int32& Into, const FSTTModel& Model, int32 SegmentId, int32 TokenIndex)
{
	whisper_context* Context = reinterpret_cast<whisper_context*>(Model.Context);
	
	Into = whisper_full_get_token_id(Context, SegmentId, TokenIndex);
}


int32 FVActSTT::_Unsafe_Resample(const float* InAudio, float* OutAudio, double Ratio, int32 NumSamples)
{
	int32 _NumSamples = 0;
	const bool bDownsample = Ratio < 1.0;
	if (bDownsample) { _NumSamples = _Unsafe_Downsample(InAudio, OutAudio, Ratio, NumSamples); }
	else { _NumSamples = _Unsafe_Upsample(InAudio, OutAudio, Ratio, NumSamples); }
	return _NumSamples;
}

int32 FVActSTT::_Unsafe_Upsample(const float* InAudio, float* OutAudio, double Ratio, int32 NumSamples)
{
	const int32 _NumSamples = FMath::CeilToInt(NumSamples / Ratio);
	for (int32 SampleId = _NumSamples - 1; SampleId >= 0; --SampleId)
	{
		double T = SampleId * Ratio;
		int32 IndexA = FMath::Clamp(int32(FMath::FloorToInt(T)), 0, NumSamples - 1);
		int32 IndexB = FMath::Clamp(IndexA + 1, 0, NumSamples - 1);
		float Alpha = static_cast<float>(T - IndexA);
		OutAudio[SampleId] = FMath::Lerp(InAudio[IndexA], InAudio[IndexB], Alpha);
	}
	return _NumSamples;
}

int32 FVActSTT::_Unsafe_Downsample(const float* InAudio, float* OutAudio, double Ratio, int32 NumSamples)
{
	const int32 _NumSamples = FMath::FloorToInt(NumSamples / Ratio);
	for (int32 SampleId = 0; SampleId < _NumSamples; ++SampleId)
	{
		double T = SampleId * Ratio;
		int32 IndexA = FMath::Clamp(int32(FMath::FloorToInt(T)), 0, NumSamples - 1);
		int32 IndexB = FMath::Clamp(IndexA + 1, 0, NumSamples - 1);
		float Alpha = static_cast<float>(T - IndexA);
		OutAudio[SampleId] = FMath::Lerp(InAudio[IndexA], InAudio[IndexB], Alpha);
	}
	return _NumSamples;
}

int32 FVActSTT::_Unsafe_ToMonoAvrage(const float* InAudio, float* OutAudio, int32 NumChannels, int32 NumSamples)
{
	const int32 _NumSamples = NumSamples / NumChannels;
	for (int32 SampleId = 0; SampleId < _NumSamples; ++SampleId)
	{
		float Sum = 0.f;
		for (int32 ChannelId = 0; ChannelId < NumSamples; ++ChannelId)
		{
			Sum += InAudio[SampleId * NumSamples + ChannelId];
		}
		OutAudio[SampleId] = Sum / NumSamples;
	}
	return _NumSamples;
}

int32 FVActSTT::_Unsafe_ToMonoCopy(const float* InAudio, int32 ChannelId, float* OutAudio, int32 NumChannels, int32 NumSamples)
{
	const int32 _NumSamples = NumSamples / NumChannels;
	for (int32 SampleId = 0; SampleId < _NumSamples; ++SampleId)
	{
		OutAudio[SampleId] = InAudio[SampleId * NumChannels + ChannelId];
	}
	return _NumSamples;
}

int32 FVActSTT::_Unsafe_ToMultiCopy(const float* InAudio, float* OutAudio, int32 NumChannels, int32 NumSamples)
{
	const int32 _NumSamples = NumSamples * NumChannels;
	// loop invariant SampleId * NumChannels + ChannelId >= SampleId
	for (int32 SampleId = NumSamples - 1; SampleId >= 0; --SampleId)
	{
		const float Sample = InAudio[SampleId];
		for (int32 ChannelId = NumChannels - 1; ChannelId >= 0; --ChannelId)
		{
			OutAudio[SampleId * NumChannels + ChannelId] = Sample;
		}
	}
	return  _NumSamples;
}