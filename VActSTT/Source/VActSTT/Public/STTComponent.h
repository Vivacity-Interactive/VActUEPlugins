#pragma once

#include "AudioCaptureComponent.h"
#include "AudioCapture.h"
#include "Sound/SoundSubmix.h"
#include "Sound/SoundWaveProcedural.h"
#include "DSP/Dsp.h"
#include "STTModelAsset.h"
#include "AudioCaptureCore.h"

#include "VActSTTTypes.h"

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "STTComponent.generated.h"

class UAudioCaptureComponent;
class USoundSubmix;
class UAudioCapture;
class USoundWaveProcedural;
class USoundCue;
class USTTModelAsset;

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class VACTSTT_API USTTComponent : public UActorComponent
{
	GENERATED_BODY()

	FThreadSafeBool bEnabled;

	TArray<float> SamplingBuffer;

	TArray<float> ResampleBuffer;

	//TArray<float> RechannelBuffer;

	Audio::TCircularAudioBuffer<float> AudioBuffer;

	//Audio::FResampler Resampler;

	FBufferHandler<FSTTToken> TranscriptTokenHandler;

	TFuture<void> ProcessTask;

	int32 BatchId;

public:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VActSTT")
	TArray<FSTTToken> TranscriptTokens;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VActSTT")
	uint8 bReady : 1;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VActSTT")
	uint8 bRunning : 1;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "VActSTT")
	uint8 bLazyAudioInit : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT|Sampling")
	int32 TokenCapacity;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VActSTT|Sampling")
	int32 SourceSampleRate;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VActSTT|Sampling")
	int32 SourceNumChannels;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT|Sampling")
	int32 SampleFromChannel;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT|Sampling")
	float BufferDurationScale;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT|Sampling")
	float ChunkDuration;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT")
	float ProcessInterval;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT")
	FSTTModelUseSettings UseSettings;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Instanced, Category = "VActSTT")
	TObjectPtr<USTTModelAsset> Model;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT")
	TObjectPtr<UAudioCapture> AudioCapture;

public:
	USTTComponent();

	void TickAudioSTT(float DeltaTime);

	bool InitForNewAudioSource(bool bForceNewSource = false);

	bool InitForNewModel(USTTModelAsset* InModel = nullptr);

	void OnAudioSample(const float* InAudio, int32 NumSamples);

	void OnProcessSample();

	UFUNCTION(BlueprintCallable, CallInEditor)
	void StartProcessing();

	UFUNCTION(BlueprintCallable, CallInEditor)
	void StopProcessing();

	UFUNCTION(BlueprintCallable, CallInEditor)
	bool IsEnabled();

	UFUNCTION(BlueprintCallable, CallInEditor)
	bool IsRunning();

protected:
	virtual void BeginPlay() override;

	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	virtual void BeginDestroy() override;

public:
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

#if WITH_EDITORONLY_DATA
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "_DEBUG", meta = (ClampMin = "-1.0", ClampMax = "1.0", UIMin = "-1.0", UIMax = "1.0"))
	float _DEBUG_SampleIn;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "_DEBUG", meta = (ClampMin = "-1.0", ClampMax = "1.0", UIMin = "-1.0", UIMax = "1.0"))
	float _DEBUG_SampleOut;
#endif
};
