#include "STTComponent.h"
#include "VActSTT.h"
#include "AudioResampler.h"
#include "Async/Async.h"
#include "Misc/Paths.h"

USTTComponent::USTTComponent()
	: bEnabled(false)
	, TimeListIteration(0)
	, bReady(false)
	, bRunning(false)
	, bNewLine(false)
	, bVAD(false)
	, bSegmentText(false)
	, bTokens(true)
	, bLazyAudioInit(true)
	, bEchoCancellation(false)
	, VADWindowDuration(2000)
	, VADDuration(1000)
	, NewLineCount(9)
	, LineCount(0)
	, IterationCount(0)
	, TokenCapacity(32)
	, StepDuration(500)
	, LengthDuration(5000)
	, KeepDuration(200)
	, ProcessInterval(0.01f)
	, VADInterval(0.1f)
	, VADTreshold(0.6f)
	, VADFrequencyTreshold(100.0f)
	, UseSettings(FSTTModelUseSettings::DefaultUseSettings)
	, LineText("")
{
	PrimaryComponentTick.bCanEverTick = true;

#if WITH_EDITOR
	_DEBUG_SampleActivity = 0;
#endif
}

void USTTComponent::StartProcessing()
{
	const bool bStart = !bEnabled && bReady && Model != nullptr && Model->Model.Context != nullptr;
	if (bStart)
	{
		bEnabled = true;
		AudioBuffer.SetNum(0, false);
		TimeListIteration = FPlatformTime::Seconds() * 1000u;
		IterationCount = 0;
		TokenPromptCount = 0;
		SamplesNewCount = 0;
		const bool bStartCapture = !AudioCapture.IsCapturing() && AudioCapture.IsStreamOpen();
		if (bStartCapture) { AudioCapture.StartStream(); }

		ProcessTask = Async(EAsyncExecution::Thread, [WeakThis = TWeakObjectPtr<USTTComponent>(this)]()
		{
#if WITH_EDITOR
			UE_LOG(LogTemp, Warning, TEXT("Started Process '%s'"), *GetNameSafe(WeakThis.Get()));
#endif
			while (WeakThis.IsValid() && WeakThis->bEnabled)
			{
				WeakThis->bRunning = true;
				WeakThis->OnProcessSample();
				FPlatformProcess::Sleep(WeakThis->ProcessInterval);
			}
			WeakThis->bRunning = false;
#if WITH_EDITOR
			UE_LOG(LogTemp, Warning, TEXT("Stopped Process '%s'"), *GetNameSafe(WeakThis.Get()));
#endif
		});
	}
}

void USTTComponent::StopProcessing()
{
	bEnabled = false;
	const bool bStop = AudioCapture.IsCapturing();
	if (bStop) { AudioCapture.StopStream(); }
}

void USTTComponent::TickAudioSTT(float DeltaTime)
{
	
}

bool USTTComponent::InitForNewSettings()
{
	bool bSuccess = false;
	const bool bStopCapture =  AudioCapture.IsCapturing();
	if (bStopCapture) { AudioCapture.StopStream(); }

	if (Model)
	{
		IterationCount = 0;
		TokenPromptCount = 0;
		SamplesNewCount = 0;
		TimeListIteration = FPlatformTime::Seconds();
		
		SamplesStepCount = VACT_STT_UNIT_HZ * StepDuration * Model->Model.SampleRate;
		SamplesLengthCount = VACT_STT_UNIT_HZ * LengthDuration * Model->Model.SampleRate;
		SamplesKeepCount = VACT_STT_UNIT_HZ * KeepDuration * Model->Model.SampleRate;
		SamplesUnitCount = VACT_STT_UNIT_HZ * Model->Model.UnitDuration * Model->Model.SampleRate;
		bVAD = SamplesStepCount <= 0;

		NewLineCount = !bVAD ? FMath::Max(1, LengthDuration / StepDuration - 1) : 1;

		UseSettings.bNoTimeStamp = !bVAD;
		UseSettings.bNoContext |= bVAD;
		UseSettings.MaxTokens = 0;
		UseSettings.bSingleSegment = !bVAD;

		TokenBuffer.Reset();
		TokenBuffer.SetCapacity(TokenCapacity);

		AudioBuffer.Reset();
		AudioBuffer.SetCapacity(SamplesUnitCount);

		NewAudioBuffer.SetNumZeroed(SamplesUnitCount);
		bSuccess = true;
	}
	return bSuccess;
}

bool USTTComponent::InitForNewAudioSource(uint32 AudioIndex)
{
	bool bSuccess = false;

	if (Model)
	{
		const bool bAbortCapture = AudioCapture.IsStreamOpen() || AudioCapture.IsCapturing();
		if (bAbortCapture) { AudioCapture.AbortStream(); }

		IterationCount = 0;
		TokenPromptCount = 0;
		TimeListIteration = FPlatformTime::Seconds();

		AudioCaptureParms = Audio::FAudioCaptureDeviceParams();
		AudioCaptureParms.PCMAudioEncoding = static_cast<Audio::EPCMAudioEncoding>(FVActSTT::FormatMap[Model->Model.AudioFormat]);
		AudioCaptureParms.NumInputChannels = Model->Model.ChannelCount;
		AudioCaptureParms.SampleRate = Model->Model.SampleRate;
		AudioCaptureParms.bUseHardwareAEC = bEchoCancellation;

		bSuccess = AudioCapture.OpenAudioCaptureStream(
			AudioCaptureParms,
			[WeakThis = TWeakObjectPtr<USTTComponent>(this)]
			(const void* InAudio, int32 NumFrames, int32 NumChannels, int32 SampleRate, double StreamTime, bool bOverFlow)
			{
				if (WeakThis.IsValid())
				{ 
					WeakThis->OnAudioSamples(InAudio, NumFrames, NumChannels, SampleRate, StreamTime, bOverFlow);
				}
			},
			Model->Model.BatchSampleCount
		);

//#if WITH_EDITOR
//		if (bSuccess)
//		{
//			UE_LOG(LogTemp, Warning, TEXT("Init Source '%s', S:%d, C:%d, (%d,%d,%d) %s"), *GetNameSafe(this), 
//				AudioCaptureParms.SampleRate, AudioCaptureParms.NumInputChannel,
//				NewAudioBuffer.Num(), AudioBuffer.Num(), AudioBuffer.GetCapacity(),
//				bSuccess ? TEXT("True") : TEXT("False"));
//		}
//#endif
	}
	return bSuccess;
}

bool USTTComponent::InitForNewModel(USTTModelAsset* InModel)
{
	bool bSuccess = false;
	if (InModel)
	{
		if (IsRunning()) { StopProcessing(); }
		Model = InModel;
	}

	const bool bTryLoadModel = Model != nullptr && Model->FilePath.FilePath.IsEmpty() == false;
	if (bTryLoadModel)
	{
		IterationCount = 0;
		TokenPromptCount = 0;
		SamplesNewCount = 0;
		TimeListIteration = FPlatformTime::Seconds();

		FString FullPath = FPaths::ProjectContentDir() / Model->FilePath.FilePath;

		// todo mayb eneeds to be loaded Async
		FVActSTT::LoadModel(Model->Model, Model->FilePath);

		bSuccess = Model->Model.bLoaded && Model->Model.Context != nullptr;

#if WITH_EDITOR
		UE_LOG(LogTemp, Warning, TEXT("Init Model '%s', %s"), *GetNameSafe(this), bSuccess ? TEXT("True") : TEXT("False"));
#endif
	}
	return bSuccess;
}
//typedef TFunction<void(const void* InAudio, int32 NumFrames, int32 NumChannels, int32 SampleRate, double StreamTime, bool bOverFlow)> FOnAudioCaptureFunction;
void USTTComponent::OnAudioSamples(const void* InAudio, int32 NumFrames, int32 NumChannels, int32 SampleRate, double StreamTime, bool bOverFlow)
{

	const bool bValid = InAudio != nullptr && NumFrames > 0
		&& Model != nullptr
		&& Model->Model.SampleRate > 0;

	const float* _InFloatAudio = static_cast<const float*>(InAudio);

	if (bValid)
	{
#if WITH_EDITOR
		_DEBUG_SampleActivity = _InFloatAudio[0];
#endif
		AudioBuffer.Push(_InFloatAudio, NumFrames);
	}
#if WITH_EDITOR
	else { UE_LOG(LogTemp, Log, TEXT("   -Sampling '%s' Failed, %d:%d"), *GetNameSafe(this), SampleRate, Model ? Model->Model.SampleRate : -1); }
#endif
}

void USTTComponent::OnProcessSample()
{
	/*if (bVAD)
	{
		double TimeNow = FPlatformTime::Seconds();
		double Duration = TimeNow - TimeListIteration;

		const bool bVADWait = Duration < VADWindowDuration;
		if (bVADWait) { FPlatformProcess::Sleep(VADInterval); return; }
	}*/

	const bool bProcess = Model != nullptr && AudioBuffer.Num() >= static_cast<uint32>(SamplesStepCount);

	// maybe drop if 
	if (bProcess)
	{

		const uint32 PopSamplesCount = FMath::Min(AudioBuffer.Num(), static_cast<uint32>(NewAudioBuffer.Num()));
		const uint32 PopOverflowCount = AudioBuffer.Num() - PopSamplesCount;
		const uint32 SamplesTakeCount = static_cast<uint32>(NewAudioBuffer.Num()) - PopSamplesCount;
		AudioBuffer.Pop(PopOverflowCount);

		/*if (bVAD)
		{
			// pop VADDuration check and if valid continue
			const bool bVADWait = FVActSTT::_Unsafe_VAD(NewAudioBuffer.GetData(), NewAudioBuffer.GetData(), VADDuration,
				Model->Model.SampleRate, 0, VADTreshold, VADFrequencyTreshold);

			if (bVADWait) { FPlatformProcess::Sleep(VADInterval); return; }
		}*/
		
		FMemory::Memmove(NewAudioBuffer.GetData(), NewAudioBuffer.GetData() + PopSamplesCount, SamplesTakeCount * sizeof(float));
		const int32 SamplesPopedCount = AudioBuffer.Pop(NewAudioBuffer.GetData() + SamplesTakeCount, PopSamplesCount);
		SamplesNewCount = SamplesTakeCount + PopSamplesCount;
		
		// maybe drop if remaining

		int32 SegmentsCount = 0;
		bool bSuccess = false;
		FVActSTT::_Unsafe_UseModel(bSuccess, SegmentsCount, Model->Model, 
			NewAudioBuffer.GetData(), SamplesNewCount, 
			UseSettings, 
			TokenPromptBuffer.GetData(), TokenPromptCount);
		
		const bool bTokenPrompt = !UseSettings.bNoContext;

		if (bSuccess)
		{
			TokenPromptCount = 0;

			const int64 RealTimeOffset = (FPlatformTime::Seconds() - (SamplesTakeCount / Model->Model.SampleRate)) * 1000ll;

			++IterationCount;
			bNewLine = (IterationCount % NewLineCount) == 0;
			if (bNewLine)
			{
				++LineCount;
				LineText = "";
				TimeStartNewLine = FPlatformTime::Seconds() * 1000ll;
			}
			
			
			if (UseSettings.bNoContext) { TokenPromptBuffer.SetNumUninitialized(SegmentsCount * UseSettings.MaxTokens, EAllowShrinking::No); }
			for (int32 SegmentId = 0; SegmentId < SegmentsCount; ++SegmentId)
			{
				int32 TokenCount = 0;
				FSTTToken Token;
				FVActSTT::PopulateToken(Token, TokenCount, Model->Model, SegmentId);

				//uint32 FrameStart = SegmentId * (Model->Model.UnitDuration)
				//FVActSTT::SegmentFrameStart(FrameStart, Model->Model, SegmentId);

				int64 T0, T1;
				FVActSTT::SegmentTime(T0, T1, Model->Model, SegmentId);
				const int64 _T1 = RealTimeOffset + T1 * Model->Model.UnitTimeScale;
#if WITH_EDITOR
				UE_LOG(LogTemp, Warning, TEXT("Init Model '%lld', %lld"), T0, T1);
#endif

				const bool bAppendSegment = _T1 >= TimeStartNewLine;// FrameStart >= SamplesTakeCount;
				if (bAppendSegment) 
				{
					FVActSTT::SegmentText(LineText, Model->Model, SegmentId);
				}

				const bool bIterateTokens = bTokens || bTokenPrompt;
				if (bIterateTokens)
				{
					for (int32 TokenIndex = 0; TokenIndex < TokenCount; ++TokenIndex)
					{
						FVActSTT::PopulateToken(Token, Model->Model, SegmentId, TokenIndex);

						if (bTokenPrompt) { TokenPromptBuffer[TokenPromptCount] = Token.Id; ++TokenPromptCount; }

						const bool bSkip = !bTokens || Token.Text.IsEmpty() || (!UseSettings.bSpecialTokens && FVActSTT::IsSpecialToken(Model->Model, Token));
						if (bSkip) { continue; }

						TokenBuffer.Push(Token);
					}
				}
			}
			
		}
		else { LineText = ""; }
	}
	else { LineText = ""; }
}

void USTTComponent::BeginPlay()
{
	Super::BeginPlay();

	TokenBuffer.Reset();
	TokenBuffer.SetCapacity(TokenCapacity);

	bool bSuccess = true;
	bSuccess &= InitForNewModel();
	bSuccess &= InitForNewSettings();
	bSuccess &= InitForNewAudioSource();

	bReady = bSuccess;
}

void USTTComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Super::EndPlay(EndPlayReason);

	StopProcessing();
	if (ProcessTask.IsValid()) { ProcessTask.Wait(); }
}

void USTTComponent::BeginDestroy()
{
	Super::BeginDestroy();

	StopProcessing();
	
	//Async(EAsyncExecution::ThreadPool, [
	//	Buffer = MoveTemp(SamplingBuffer),
	//	Task = MoveTemp(ProcessTask),
	//	Model = MoveTemp(Model)
	//]() { if (Task.IsValid()) { Task.Wait(); } });
}

void USTTComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	if (bLazyAudioInit && !bReady)
	{
		bReady = Model != nullptr && Model->Model.bLoaded && Model->Model.Context != nullptr && InitForNewAudioSource();
	}

	if (bEnabled) { TickAudioSTT(DeltaTime); }
}

bool USTTComponent::IsEnabled()
{
	return bEnabled;
}

bool USTTComponent::IsRunning()
{
	return bRunning;
}

int32 USTTComponent::NumTokens()
{
	return TokenBuffer.Num();
}

bool USTTComponent::PopToken(UPARAM(ref) FSTTToken& Into)
{
	const bool bValid = TokenBuffer.Num() > 0;
	if (bValid) { Into = TokenBuffer.Pop(); }
	return bValid;
}

int32 USTTComponent::PopTokens(UPARAM(ref) TArray<FSTTToken>& Into, int32 Count)
{
	return TokenBuffer.Pop(Into.GetData(), FMath::Min(Count, Into.Num()));
}

int32 USTTComponent::PopTokensStart(UPARAM(ref) TArray<FSTTToken>& Into, int32 Count, int32 Start)
{
	const bool bValid = Start < Into.Num();
	return bValid ? TokenBuffer.Pop(&Into[Start], FMath::Min(Count, Into.Num())) : 0;
}