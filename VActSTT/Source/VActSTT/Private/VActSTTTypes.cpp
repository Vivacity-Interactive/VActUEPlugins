#include "VActSTTTypes.h"

const FSTTModelUseSettings FSTTModelUseSettings::DefaultUseSettings = FSTTModelUseSettings();

FSTTCaptureDevice::FSTTCaptureDevice()
	: DeviceName(TEXT("Default"))
	, DeviceId(TEXT("Default"))
	, Index(INDEX_NONE)
{

}

FSTTModelUseSettings::FSTTModelUseSettings()
	: bTranslate(false)
	, bSpecialTokens(false)
	, bNoFallback(true)
	, bNoContext(true)
	, bNoTimeStamp(false)
	, bSingleSegment(false)
	, bTinyDiarize(false)
	, bUseGPU(false)
	, bFlashAttention(true)
	, MaxTokens(32)
	, BeamSize(-1)
	, ThreadCount(8)
	, AudioContext(0)
	, Language(TEXT("en"))
	, Mode(ESTTModelMode::Greedy)
{
}

FSTTModel::FSTTModel()
	: bLoaded(false)
	, bGPU(false)
	, bFlashAttention(true)
	, SampleRate(16000)
	, ChannelCount(1)
	, UnitDuration(30000)
	, BatchSampleCount(1024)
	, UnitTimeScale(10.0)
	, AudioFormat(ESTTAudioFormat::FLOATING_POINT_32)
	, Context(nullptr)
{
}

FSTTToken::FSTTToken()
	: TimeStamp(-1)
	, ContextId(-1)
	, SegmentId(-1)
	, Id(-1)
	, Index(-1)
	, Duration(0.0f)
	, Probability(0.0f)
	, Text(TEXT(""))
{
}