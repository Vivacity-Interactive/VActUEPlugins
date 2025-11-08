#include "STTComponent.h"
#include "VActSTT.h"
#include "AudioResampler.h"
#include "Async/Async.h"
#include "Misc/Paths.h"

USTTComponent::USTTComponent()
	: bEnabled(false)
	, ActiveUseSettings(FSTTModelUseSettings::DefaultUseSettings)
	, TimeListIteration(0)
	, bReady(false)
	, bRunning(false)
	, bNewLine(false)
	, bVAD(false)
	, bForceVAD(false)
	, bCustomLineCount(false)
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
	, SegmentText("")
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
	if (IsRunning()) { StopProcessing(); }

	if (Model)
	{
		IterationCount = 0;
		TokenPromptCount = 0;
		SamplesNewCount = 0;
		TimeListIteration = FPlatformTime::Seconds();
		
		SamplesStepCount = VACT_STT_UNIT_HZ * StepDuration * Model->Model.SampleRate;
		SamplesLengthCount = VACT_STT_UNIT_HZ * LengthDuration * Model->Model.SampleRate;
		SamplesKeepCount = VACT_STT_UNIT_HZ * KeepDuration * Model->Model.SampleRate;
		VADSamplesCount = VACT_STT_UNIT_HZ * VADDuration * Model->Model.SampleRate;
		bVAD = bForceVAD || SamplesStepCount <= 0;

		SamplesUnitCount = FMath::Min(
			SamplesKeepCount + FMath::Max(bVAD ? VADSamplesCount : SamplesLengthCount, SamplesStepCount),
			static_cast<int32>(VACT_STT_UNIT_HZ * Model->Model.UnitDuration * Model->Model.SampleRate)
		);

		NewLineCount = !bVAD ? FMath::Max(1, bCustomLineCount ? NewLineCount : (LengthDuration / StepDuration - 1)) : 1;

		UseSettings.bNoTimeStamp = !bVAD;
		UseSettings.bNoContext |= bVAD;
		UseSettings.MaxTokens = 0;
		UseSettings.bSingleSegment = !bVAD;

		TokenBuffer.Reset();
		TokenBuffer.SetCapacity(TokenCapacity);

		AudioBuffer.Reset();
		AudioBuffer.SetCapacity(SamplesUnitCount);

		NewAudioBuffer.SetNumZeroed(SamplesUnitCount);

		ActiveUseSettings = UseSettings;

		bSuccess = true;
	}
	return bSuccess;
}

bool USTTComponent::InitForNewAudioSource()
{
	bool bSuccess = false;
	if (IsRunning()) { StopProcessing(); }

	if (Model)
	{
		const bool bAbortCapture = AudioCapture.IsStreamOpen() || AudioCapture.IsCapturing();
		if (bAbortCapture) { AudioCapture.AbortStream(); }

		TArray<Audio::FCaptureDeviceInfo> Devices;
		AudioCapture.GetCaptureDevicesAvailable(Devices);

		IterationCount = 0;
		TokenPromptCount = 0;
		TimeListIteration = FPlatformTime::Seconds();

		AudioCaptureParms = Audio::FAudioCaptureDeviceParams();
		AudioCaptureParms.PCMAudioEncoding = static_cast<Audio::EPCMAudioEncoding>(FVActSTT::FormatMap[Model->Model.AudioFormat]);
		AudioCaptureParms.NumInputChannels = Model->Model.ChannelCount;
		AudioCaptureParms.SampleRate = Model->Model.SampleRate;
		AudioCaptureParms.bUseHardwareAEC = bEchoCancellation;
		AudioCaptureParms.DeviceIndex = Devices.IsValidIndex(CaptureDevice.Index) ? CaptureDevice.Index : INDEX_NONE;

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

#if WITH_EDITOR
		if (bSuccess)
		{
			UE_LOG(LogTemp, Warning, TEXT("Init Source '%s', S:%d, C:%d, %s"), *GetNameSafe(this), 
				AudioCaptureParms.SampleRate, AudioCaptureParms.NumInputChannels,
				bSuccess ? TEXT("True") : TEXT("False"));
		}
#endif
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
	
	if (bVAD)
	{
		double TimeNow = FPlatformTime::Seconds();
		double Duration = TimeNow - TimeListIteration;

		const bool bVADWait = Duration < VADWindowDuration;
		if (bVADWait) { FPlatformProcess::Sleep(VADInterval); return; }
	}

	const bool bProcess = Model != nullptr && ((bVAD && AudioBuffer.Num() > 0) || AudioBuffer.Num() >= static_cast<uint32>(SamplesStepCount));

	// maybe drop if 
	if (bProcess)
	{
		const uint32 PopSamplesCount = FMath::Min(AudioBuffer.Num(), static_cast<uint32>(NewAudioBuffer.Num()));
		const uint32 PopOverflowCount = AudioBuffer.Num() - PopSamplesCount;
		const uint32 SamplesTakeCount = static_cast<uint32>(NewAudioBuffer.Num()) - PopSamplesCount;
		AudioBuffer.Pop(PopOverflowCount);

		FMemory::Memmove(NewAudioBuffer.GetData(), NewAudioBuffer.GetData() + PopSamplesCount, SamplesTakeCount * sizeof(float));
		
		uint32 PrePopOffset = 0;
		if (bVAD)
		{
			// pop VADDuration check and if valid continue
			const int32 _VADSamplesCount = FMath::Min(VADSamplesCount, NewAudioBuffer.Num());
			PrePopOffset = AudioBuffer.Pop(NewAudioBuffer.GetData(), _VADSamplesCount);
			const bool bVADWait = FVActSTT::_Unsafe_VAD(NewAudioBuffer.GetData(), NewAudioBuffer.GetData(), PrePopOffset,
				Model->Model.SampleRate, 0, VADTreshold, VADFrequencyTreshold);

			if (bVADWait) { FPlatformProcess::Sleep(VADInterval); return; }
		}
		
		const int32 SamplesPopedCount = AudioBuffer.Pop(NewAudioBuffer.GetData() + SamplesTakeCount + PrePopOffset, PopSamplesCount - PrePopOffset);
		SamplesNewCount = SamplesTakeCount + PopSamplesCount;
		
		// maybe drop if remaining

		int32 SegmentsCount = 0;
		bool bSuccess = false;
		FVActSTT::_Unsafe_UseModel(bSuccess, SegmentsCount, Model->Model, 
			NewAudioBuffer.GetData(), SamplesNewCount, 
			ActiveUseSettings,
			TokenPromptBuffer.GetData(), TokenPromptCount);
		
		const bool bTokenPrompt = !ActiveUseSettings.bNoContext;

		if (bSuccess)
		{
			TokenPromptCount = 0;

			const int64 RealTimeOffset = (FPlatformTime::Seconds() - (SamplesTakeCount / Model->Model.SampleRate)) * 1000ll;

			++IterationCount;
			bNewLine = (IterationCount % NewLineCount) == 0;
			if (bNewLine)
			{
				++LineCount;
				SegmentText = "";
				TimeStartNewLine = FPlatformTime::Seconds() * 1000ll;
			}
			
			
			if (ActiveUseSettings.bNoContext) { TokenPromptBuffer.SetNumUninitialized(SegmentsCount * ActiveUseSettings.MaxTokens, EAllowShrinking::No); }
			for (int32 SegmentId = 0; SegmentId < SegmentsCount; ++SegmentId)
			{
				int32 TokenCount = 0;
				FSTTToken Token;
				FVActSTT::PopulateToken(Token, TokenCount, Model->Model, SegmentId);

				const bool bSegment = OnSegmentPredicate(SegmentId, RealTimeOffset, PopSamplesCount, SamplesTakeCount);
				if (bSegment)
				{
					FVActSTT::SegmentText(SegmentText, Model->Model, SegmentId);
				}

				const bool bIterateTokens = bTokens || bTokenPrompt;
				if (bIterateTokens)
				{
					for (int32 TokenIndex = 0; TokenIndex < TokenCount; ++TokenIndex)
					{
						FVActSTT::PopulateToken(Token, Model->Model, SegmentId, TokenIndex);

						if (bTokenPrompt) { TokenPromptBuffer[TokenPromptCount] = Token.Id; ++TokenPromptCount; }

						const bool bToken = OnTokenPredicate(Token, RealTimeOffset, PopSamplesCount, SamplesTakeCount);
						
						if (bToken)
						{
							TokenBuffer.Push(Token);
						}
					}
				}
			}
			
		}
		else { SegmentText = ""; }
	}
	else { SegmentText = ""; }
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

bool USTTComponent::OnTokenPredicate_Implementation(FSTTToken& InToken, int64 RealTimeOffset, int32 PopSamplesCount, int32 SamplesTakeCount)
{
	const bool bSkip = !bTokens || InToken.Text.IsEmpty() || (!ActiveUseSettings.bSpecialTokens && FVActSTT::IsSpecialToken(Model->Model, InToken));
	return !bSkip;
}

bool USTTComponent::OnSegmentPredicate_Implementation(int32 SegmentId, int64 RealTimeOffset, int32 PopSamplesCount, int32 SamplesTakeCount)
{
	int64 T0, T1;
	FVActSTT::SegmentTime(T0, T1, Model->Model, SegmentId);
	const int64 _T1 = RealTimeOffset + T1 * Model->Model.UnitTimeScale;
	return _T1 >= TimeStartNewLine;
}
#if WITH_EDITOR
void USTTComponent::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
	Super::PostEditChangeProperty(PropertyChangedEvent);

	if (PropertyChangedEvent.Property &&
		PropertyChangedEvent.Property->GetFName() == GET_MEMBER_NAME_CHECKED(USTTComponent, SelectDevice))
	{
		TArray<Audio::FCaptureDeviceInfo> Devices;
		Audio::FAudioCapture TempDevice;
		TempDevice.GetCaptureDevicesAvailable(Devices);

		for (int32 Index = 0; Index < Devices.Num(); ++Index)
		{
			if (Devices[Index].DeviceName == SelectDevice)
			{
				CaptureDevice.DeviceName = Devices[Index].DeviceName;
				CaptureDevice.DeviceId = Devices[Index].DeviceId;
				CaptureDevice.Index = Index;
				break;
			}
		}
	}
}

TArray<FString> USTTComponent::GetAvailableDeviceNames()
{
	TArray<FString> DeviceNames;
	TArray<Audio::FCaptureDeviceInfo> Devices;
	Audio::FAudioCapture TempDevice;
	TempDevice.GetCaptureDevicesAvailable(Devices);

	UE_LOG(LogTemp, Warning, TEXT("Available Devices %d"), Devices.Num());

	for (const Audio::FCaptureDeviceInfo& Info : Devices)
	{
		UE_LOG(LogTemp, Warning, TEXT(" - %s"), *Info.DeviceName);
		DeviceNames.Add(Info.DeviceName);
	}

	return DeviceNames;
}

void USTTComponent::InitNewSettings()
{
	InitForNewSettings();
}

void USTTComponent::InitNewAudioSource()
{
	InitForNewAudioSource();
}

void USTTComponent::InitNewModel()
{
	InitForNewModel();
}
#endif