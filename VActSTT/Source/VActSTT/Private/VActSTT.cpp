#include "VActSTT.h"

#include "whisper.h"
#include "AudioCaptureCore.h"

const TMap<ESTTModelMode, uint32> FVActSTT::ModeMap = {
	{ ESTTModelMode::Greedy, WHISPER_SAMPLING_GREEDY},
	{ ESTTModelMode::BeamSearch, WHISPER_SAMPLING_BEAM_SEARCH }
};

const TMap<ESTTAudioFormat, uint32> FVActSTT::FormatMap = {
	{ ESTTAudioFormat::UNKNOWN, static_cast<uint32>(Audio::EPCMAudioEncoding::UNKNOWN)},
	{ ESTTAudioFormat::PCM_8, static_cast<uint32>(Audio::EPCMAudioEncoding::PCM_8) },
	{ ESTTAudioFormat::PCM_16,static_cast<uint32>( Audio::EPCMAudioEncoding::PCM_16) },
	{ ESTTAudioFormat::PCM_24, static_cast<uint32>(Audio::EPCMAudioEncoding::PCM_24) },
	{ ESTTAudioFormat::PCM_24_IN_32, static_cast<uint32>(Audio::EPCMAudioEncoding::PCM_24_IN_32) },
	{ ESTTAudioFormat::PCM_32, static_cast<uint32>(Audio::EPCMAudioEncoding::PCM_32) },
	{ ESTTAudioFormat::FLOATING_POINT_32, static_cast<uint32>(Audio::EPCMAudioEncoding::FLOATING_POINT_32) },
	{ ESTTAudioFormat::FLOATING_POINT_64, static_cast<uint32>(Audio::EPCMAudioEncoding::FLOATING_POINT_64) }
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

void FVActSTT::_Unsafe_UseModel(bool& Into, int32& SegmentCount, const FSTTModel& Model, const float* Buffer, int32 Count, const FSTTModelUseSettings& UseSettings, int32* PromptBuffer, int32 PromptCount)
{
	whisper_context* Context = reinterpret_cast<whisper_context*>(Model.Context);

	whisper_full_params Params = whisper_full_default_params((enum whisper_sampling_strategy)ModeMap[UseSettings.Mode]);
	Params.print_progress = false;
	Params.print_special = false;
	Params.print_timestamps = false;
	Params.print_realtime = false;
	Params.single_segment = UseSettings.bSingleSegment;
	Params.max_tokens = UseSettings.MaxTokens;
	Params.n_threads = UseSettings.ThreadCount;
	Params.translate = UseSettings.bTranslate;
	Params.language = TCHAR_TO_ANSI(*UseSettings.Language);
	Params.beam_search.beam_size = UseSettings.BeamSize;
	Params.audio_ctx = UseSettings.AudioContext;
	Params.tdrz_enable = UseSettings.bTinyDiarize;
	Params.temperature_inc = UseSettings.bNoFallback ? 0.0f : Params.temperature_inc;
	Params.prompt_tokens = UseSettings.bNoContext ? nullptr : PromptBuffer;
	Params.prompt_n_tokens = UseSettings.bNoContext ? 0 : PromptCount;

	Into = whisper_full(Context, Params, Buffer, Count) == 0;

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

	TokenCount = whisper_full_n_tokens(Context, SegmentId);
}

void FVActSTT::_Unsafe_SegmentTime(int64& T0, int64& T1, const FSTTModel& Model, int32 SegmentId)
{
	whisper_context* Context = reinterpret_cast<whisper_context*>(Model.Context);
	T0 = whisper_full_get_segment_t0(Context, SegmentId);
	T1 = whisper_full_get_segment_t1(Context, SegmentId);
}

void FVActSTT::_Unsafe_SegmentTextAppend(FString& Into, const FSTTModel& Model, int32 SegmentId)
{
	whisper_context* Context = reinterpret_cast<whisper_context*>(Model.Context);
	const char* text = whisper_full_get_segment_text(Context, SegmentId);
	Into += ANSI_TO_TCHAR(text);
}

void FVActSTT::_Unsafe_SegmentTextAppend(FString& Into, const FSTTModel& Model, int32 SegmentId, int32 SegmentTextOffset)
{
	whisper_context* Context = reinterpret_cast<whisper_context*>(Model.Context);
	const char* text = whisper_full_get_segment_text(Context, SegmentId);
	Into += ANSI_TO_TCHAR(text + SegmentTextOffset);
}

void FVActSTT::_Unsafe_SegmentText(FString& Into, const FSTTModel& Model, int32 SegmentId)
{
	whisper_context* Context = reinterpret_cast<whisper_context*>(Model.Context);
	const char* text = whisper_full_get_segment_text(Context, SegmentId);
	Into = ANSI_TO_TCHAR(text); 
}

void FVActSTT::_Unsafe_SegmentText(FString& Into, const FSTTModel& Model, int32 SegmentId, int32 SegmentTextOffset)
{
	whisper_context* Context = reinterpret_cast<whisper_context*>(Model.Context);
	const char* text = whisper_full_get_segment_text(Context, SegmentId);
	Into = ANSI_TO_TCHAR(text + SegmentTextOffset);
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

void FVActSTT::_Unsafe_TokenTime(int64& T0, int64& T1, const FSTTModel& Model, int32 SegmentId, int32 TokenIndex)
{
	whisper_context* Context = reinterpret_cast<whisper_context*>(Model.Context);
	T0 = T1 = static_cast<int64>(FPlatformTime::Seconds() * 1000);
}



void FVActSTT::_Unsafe_HighPassFilter(const float* InAudio, float* OutAudio, int32 NumSamples, float Treshold, int32 SampleRate)
{
	const float Tao = 1.0f / (2.0f * PI * Treshold);
	const float Delta = 1.0f / static_cast<float>(SampleRate);
	const float Alpha = Delta / (Tao + Delta);

	float Sample = InAudio[0];

	for (int32 SampleId = 1; SampleId < NumSamples; ++SampleId)
	{
		OutAudio[SampleId] = Alpha * (Sample + InAudio[SampleId] - InAudio[SampleId - 1]);
	}
}

bool FVActSTT::_Unsafe_VAD(const float* InAudio, float* OutAudio, int32 NumSamples, int32 SampleRate, int32 Last, float Treshold, float FequencyTreshold)
{
	const int32 NumSamplesLast = SampleRate * Last / 1000;

	bool bVAD = NumSamplesLast < NumSamples;
	if (bVAD)
	{
		if (Treshold > 0.0f) { _Unsafe_HighPassFilter(InAudio, OutAudio, NumSamples, FequencyTreshold, SampleRate); }

		float Energy = 0.0f;
		float EnergyLast = 0.0f;

		for (int32 SampleId = 0; SampleId < NumSamples; ++SampleId)
		{
			Energy += FMath::Abs(InAudio[SampleId]);

			if (SampleId >= NumSamples - NumSamplesLast) { EnergyLast += InAudio[SampleId]; }
		}

		Energy /= NumSamples;
		EnergyLast /= NumSamplesLast;

		bVAD = EnergyLast > Treshold * Energy;
	}
	return bVAD;
}
