#include "STTComponent.h"
#include "VActSTT.h"
#include "AudioResampler.h"
#include "Async/Async.h"
#include "Misc/Paths.h"

USTTComponent::USTTComponent()
	: bEnabled(false)
	, TranscriptTokenHandler(TranscriptTokens)
	, bReady(false)
	, bRunning(false)
	, bLazyAudioInit(true)
	, TokenCapacity(19)
	, SourceSampleRate(16000)
	, SourceNumChannels(1)
	, SampleFromChannel(0)
	, BufferDurationScale(2.0f)
	, ChunkDuration(2.0f)
	, ProcessInterval(0.03f)
	, UseSettings(FSTTModelUseSettings::DefaultUseSettings)
{
	PrimaryComponentTick.bCanEverTick = true;

#if WITH_EDITOR
	_DEBUG_SampleIn = _DEBUG_SampleOut = 0;
#endif
}

void USTTComponent::StartProcessing()
{
	const bool bStart = !bEnabled && bReady && Model != nullptr && Model->Model.Context != nullptr;
	if (bStart)
	{
		bEnabled = true;
		AudioBuffer.SetNum(0, false);
		if (!AudioCapture->IsCapturingAudio()) { AudioCapture->StartCapturingAudio(); }

		//TWeakObjectPtr<USTTComponent> WeakThis = this;
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
	const bool bStop = AudioCapture != nullptr && AudioCapture->IsCapturingAudio();
	if (bStop) { AudioCapture->StopCapturingAudio(); }
}

void USTTComponent::TickAudioSTT(float DeltaTime)
{

}

bool USTTComponent::InitForNewAudioSource(bool bForceNewSource)
{
	bool bSuccess = false;
	bool bNewSource = bForceNewSource || AudioCapture == nullptr;
	if (bNewSource) { AudioCapture = NewObject<UAudioCapture>(); }

	if (Model)
	{
		const int32 Sampling = Model->Model.SampleRate;
		const int32 SamplingBufferSize = FMath::CeilToInt(ChunkDuration * Sampling);
		const int32 AudioBufferSize = FMath::CeilToInt(SamplingBufferSize * BufferDurationScale);
		SamplingBuffer.SetNumUninitialized(SamplingBufferSize);
		AudioBuffer.Reset();
		AudioBuffer.SetCapacity(AudioBufferSize);
		if (AudioCapture != nullptr)
		{
			bSuccess = AudioCapture->OpenDefaultAudioStream();
			SourceSampleRate = AudioCapture->GetSampleRate();
			SourceNumChannels = AudioCapture->GetNumChannels();
			/*Resampler.Init(
				Audio::EResamplingMethod::Linear,
				static_cast<float>(Model->Model.SampleRate) / SourceSampleRate,
				Model->Model.ChannelCount);*/

			AudioCapture->AddGeneratorDelegate([WeakThis = TWeakObjectPtr<USTTComponent>(this)]
				(const float* InAudio, int32 NumSamples)
				{
					if (WeakThis.IsValid()) { WeakThis->OnAudioSample(InAudio, NumSamples); }
				}
			);
			
#if WITH_EDITOR
			UE_LOG(LogTemp, Warning, TEXT("Init Source '%s', S:%d, C:%d, (%d,%d,%d) %s"), *GetNameSafe(this), 
				SourceSampleRate, SourceNumChannels,
				SamplingBuffer.Num(), AudioBuffer.Num(), AudioBuffer.GetCapacity(),
				bSuccess ? TEXT("True") : TEXT("False"));
#endif
		}
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
		FString FullPath = FPaths::ProjectContentDir() / Model->FilePath.FilePath;

		// todo mayb eneeds to be loaded Async
		FVActSTT::LoadModel(Model->Model, Model->FilePath);
		BatchId = 0;

		/*const bool bInitResampler = Model->Model.bLoaded && Model->Model.Context != nullptr && AudioCapture != nullptr;
		if (bInitResampler)
		{
			Resampler.Init(
				Audio::EResamplingMethod::Linear,
				static_cast<float>(Model->Model.SampleRate) / static_cast<float>(SourceSampleRate),
				Model->Model.ChannelCount);
		}*/
		bSuccess = Model->Model.bLoaded && Model->Model.Context != nullptr;
#if WITH_EDITOR
		UE_LOG(LogTemp, Warning, TEXT("Init Model '%s', %s"), *GetNameSafe(this), bSuccess ? TEXT("True") : TEXT("False"));
#endif
	}
	return bSuccess;
}

void USTTComponent::OnAudioSample(const float* InAudio, int32 NumSamples)
{
	const bool bValid = InAudio != nullptr && NumSamples > 0 
		&& Model != nullptr && SourceSampleRate > 0 
		&& Model->Model.SampleRate > 0 
		&& SampleFromChannel < SourceNumChannels;

	if (bValid)
	{
		// warning resize without shring may cause memory problems
		// warning cast away const may be dangerous here
		
		BatchId = 0;
		float* _InAudio = const_cast<float*>(InAudio);
#if WITH_EDITOR
		_DEBUG_SampleIn = _InAudio[0];
#endif
		const bool bRechannel = SourceNumChannels != Model->Model.ChannelCount;
		if (bRechannel)
		{
			const int32 _NumSamples = NumSamples / SourceNumChannels;
			ResampleBuffer.SetNumUninitialized(_NumSamples, EAllowShrinking::No);
			FVActSTT::_Unsafe_ToMonoCopy(_InAudio, SampleFromChannel, ResampleBuffer.GetData(), SourceNumChannels, NumSamples);
			NumSamples = _NumSamples;
			_InAudio = ResampleBuffer.GetData();
		}

		const bool bResample = SourceSampleRate != Model->Model.SampleRate;
		if (bResample)
		{
			const double Ratio = static_cast<double>(Model->Model.SampleRate)/ static_cast<double>(SourceSampleRate);
			const int32 _NumSamples = FMath::CeilToInt(static_cast<double>(NumSamples) * Ratio);
			ResampleBuffer.SetNumUninitialized(_NumSamples, EAllowShrinking::No);
			FVActSTT::_Unsafe_Resample(_InAudio, ResampleBuffer.GetData(), Ratio, NumSamples);
			NumSamples = _NumSamples;
			_InAudio = ResampleBuffer.GetData();
		}

		const bool bMultichannel = Model->Model.ChannelCount > 1;
		if (bMultichannel)
		{
			const int32 _NumSamples = NumSamples * Model->Model.ChannelCount;
			ResampleBuffer.SetNumUninitialized(_NumSamples, EAllowShrinking::No);
			FVActSTT::_Unsafe_ToMultiCopy(_InAudio, ResampleBuffer.GetData(), Model->Model.ChannelCount, NumSamples);
			NumSamples = _NumSamples;
			_InAudio = ResampleBuffer.GetData();
		}

#if WITH_EDITOR
		_DEBUG_SampleOut = _InAudio[0];
#endif
		AudioBuffer.Push(_InAudio, NumSamples);
	}
#if WITH_EDITOR
	else { UE_LOG(LogTemp, Log, TEXT("   -Sampling '%s' Failed, %d:%d"), *GetNameSafe(this), SourceSampleRate, Model ? Model->Model.SampleRate : -1); }	
#endif
}

void USTTComponent::OnProcessSample()
{
	const bool bProcess = Model != nullptr && AudioBuffer.Num() >= static_cast<uint32>(SamplingBuffer.Num());
	if (bProcess)
	{
		AudioBuffer.Pop(SamplingBuffer.GetData(), SamplingBuffer.Num());

		int32 SegmentsCount = 0;
		bool bSuccess = false;
		FVActSTT::UseModel(bSuccess, SegmentsCount, Model->Model, SamplingBuffer, UseSettings);

		if (bSuccess)
		{
			for (int32 SegmentId = 0; SegmentId < SegmentsCount; ++SegmentId)
			{
				int32 TokenCount = 0;
				FSTTToken Token;
				Token.BatchId = BatchId;
				FVActSTT::PopulateToken(Token, TokenCount, Model->Model, SegmentId);

				for (int32 TokenIndex = 0; TokenIndex < TokenCount; ++TokenIndex)
				{
					FVActSTT::PopulateToken(Token, Model->Model, SegmentId, TokenIndex);
					
					const bool bSkip = Token.Text.IsEmpty() || (UseSettings.bSkipSpecial && FVActSTT::IsSpecialToken(Model->Model, Token));
					if (bSkip) { continue; }

					if (TranscriptTokenHandler.Array) { TranscriptTokenHandler.Add(Token); }
				}
			}
			++BatchId;
		}
	}
}

void USTTComponent::BeginPlay()
{
	Super::BeginPlay();

	TranscriptTokens.SetNum(TokenCapacity);

	bool bSuccess = true;
	bSuccess &= InitForNewModel();
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