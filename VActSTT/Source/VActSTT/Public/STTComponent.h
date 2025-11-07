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

	TArray<float> NewAudioBuffer;

	TArray<int32> TokenPromptBuffer;

	Audio::TCircularAudioBuffer<float> AudioBuffer;

	Audio::TCircularAudioBuffer<FSTTToken> TokenBuffer;

	TFuture<void> ProcessTask;

	int32 SamplesStepCount;

	int32 SamplesLengthCount;

	int32 SamplesKeepCount;

	int32 SamplesUnitCount;

	uint32 SamplesNewCount;

	uint32 TokenPromptCount;

	int64 TimeStartNewLine;

	Audio::FAudioCapture AudioCapture;

	Audio::FAudioCaptureDeviceParams AudioCaptureParms;

	double TimeListIteration;

public:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VActSTT|State")
	uint8 bReady : 1;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VActSTT|State")
	uint8 bRunning : 1;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VActSTT|State")
	uint8 bNewLine : 1;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VActSTT|State")
	uint8 bVAD : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT")
	uint8 bSegmentText : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT")
	uint8 bTokens : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT")
	uint8 bLazyAudioInit : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT")
	uint8 bEchoCancellation : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT|VAD")
	int32 VADWindowDuration;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT|VAD")
	int32 VADDuration;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VActSTT|State")
	int32 NewLineCount;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VActSTT|State")
	int32 LineCount;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VActSTT|State")
	int32 IterationCount;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT|Buffers")
	int32 TokenCapacity;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT|Buffers")
	int32 StepDuration;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT|Buffers")
	int32 LengthDuration;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT|Buffers")
	int32 KeepDuration;
	
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT")
	float ProcessInterval;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT|VAD")
	float VADInterval;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT|VAD")
	float VADTreshold;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT|VAD")
	float VADFrequencyTreshold;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VActSTT")
	FSTTModelUseSettings UseSettings;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Instanced, Category = "VActSTT")
	TObjectPtr<USTTModelAsset> Model;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VActSTT|State")
	FString LineText;

public:
	USTTComponent();

	void TickAudioSTT(float DeltaTime);

	bool InitForNewSettings();

	bool InitForNewAudioSource(uint32 AudioIndex = INDEX_NONE);

	bool InitForNewModel(USTTModelAsset* InModel = nullptr);

	void OnAudioSamples(const void* InAudio, int32 NumFrames, int32 NumChannels, int32 SampleRate, double StreamTime, bool bOverFlow);

	void OnProcessSample();

	UFUNCTION(BlueprintCallable, CallInEditor)
	void StartProcessing();

	UFUNCTION(BlueprintCallable, CallInEditor)
	void StopProcessing();

	UFUNCTION(BlueprintCallable, CallInEditor)
	bool IsEnabled();

	UFUNCTION(BlueprintPure, CallInEditor)
	bool IsRunning();

	UFUNCTION(BlueprintPure, CallInEditor)
	int32 NumTokens();

	UFUNCTION(BlueprintCallable, CallInEditor)
	bool PopToken(UPARAM(ref) FSTTToken& Into);

	UFUNCTION(BlueprintCallable, CallInEditor)
	int32 PopTokens(UPARAM(ref) TArray<FSTTToken>& Into, int32 Count);

	UFUNCTION(BlueprintCallable, CallInEditor)
	int32 PopTokensStart(UPARAM(ref) TArray<FSTTToken>& Into, int32 Count, int32 Start);

protected:
	virtual void BeginPlay() override;

	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	virtual void BeginDestroy() override;

public:
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

#if WITH_EDITORONLY_DATA
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "_DEBUG", meta = (ClampMin = "-1.0", ClampMax = "1.0", UIMin = "-1.0", UIMax = "1.0"))
	float _DEBUG_SampleActivity;
#endif
};
