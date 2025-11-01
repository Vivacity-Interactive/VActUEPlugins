#include "VActSTTTypes.h"

const FSTTModelUseSettings FSTTModelUseSettings::DefaultUseSettings = FSTTModelUseSettings();

FSTTModelUseSettings::FSTTModelUseSettings()
	: bTranslate(false)
	, bSkipSpecial(true)
	, Language(TEXT("en"))
	, Mode(ESTTModelMode::Greedy)
{
}

FSTTModel::FSTTModel()
	: bLoaded(false)
	, bGPU(false)
	, bFlashAttention(false)
	, SampleRate(16000)
	, ChannelCount(1)
	, Context(nullptr)
{
}

FSTTToken::FSTTToken()
	: TimeStamp(-1)
	, BatchId(-1)
	, SegmentId(-1)
	, Id(-1)
	, Index(-1)
	, Duration(0.0f)
	, Probability(0.0f)
	, Text(TEXT(""))
{
}