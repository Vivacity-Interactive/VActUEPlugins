#include "APIServerAudioUpload.h"

#include "Sound/SoundWaveProcedural.h"

UAPIServerAudioUpload::UAPIServerAudioUpload()
	: bForceAudioFormat(false)
	, AudioFormat(EAPIAudioFormat::Unknown)
	, SampleRate(44100)
	, NumChannels(2)
	, Bits(16)
{

}

bool UAPIServerAudioUpload::OnDataIn(
	const FHttpServerRequest& Request,
	const FAPIEntry& SelfEntry,
	UAPIRoute* Parent,
	UAPIInstance* Instance,
	FGuid& UserId
)
{
	USoundWave* Audio = nullptr;
	TArray<FAPIConstMultipartSegment> FormSegments;
	bool bSuccess = Request.Body.Num() > 0 && FVActAPI::Multipart(Request, FormSegments) && FormSegments.Num() > 0;

#if WITH_EDITOR
	UE_LOG(LogTemp, Warning, TEXT("%s try starting audio request"), *GetNameSafe(this));
#endif

	if (bSuccess)
	{
		const int32 HeaderSize = 44;
		FAPIConstMultipartSegment& Segment = FormSegments[0];		

		int32 _NumChannels = NumChannels;
		int32 _SampleRate = SampleRate;
		int32 _Bits = Bits;
		int32 _BlockAlign = _NumChannels * (_Bits / 8);
		int32 _PCMSize = Segment.Body.Num() - HeaderSize;
		
		// Only takes wave atm

		if (!bForceAudioFormat)
		{
			_NumChannels = *reinterpret_cast<const uint16*>(Segment.Body.GetData() + 22);
			_SampleRate = *reinterpret_cast<const uint32*>(Segment.Body.GetData() + 24);
			_Bits = *reinterpret_cast<const uint16*>(Segment.Body.GetData() + 34);
			_BlockAlign = *reinterpret_cast<const uint16*>(Segment.Body.GetData() + 32);
		}

		float _Duration = static_cast<float>(Segment.Body.Num() - HeaderSize) / (_SampleRate * _BlockAlign);

		USoundWaveProcedural* _Audio;
		_Audio = NewObject<USoundWaveProcedural>();
		_Audio->QueueAudio(Segment.Body.GetData() + HeaderSize, _PCMSize);
		Audio = _Audio;

		Audio->NumChannels = _NumChannels;
		Audio->Duration = _Duration;
		Audio->bLooping = false;
		Audio->bCanProcessAsync = true;
		Audio->SetSampleRate(static_cast<uint32>(_SampleRate));

		bSuccess &= OnAudioIn(Audio);
	}
	
	return bSuccess;
}

bool UAPIServerAudioUpload::OnDataOut(
	TUniquePtr<FHttpServerResponse>& Response,
	const FHttpServerRequest& Request,
	const FAPIEntry& SelfEntry,
	UAPIRoute* Parent,
	UAPIInstance* Instance,
	FGuid& UserId
)
{
	return false;
}

bool UAPIServerAudioUpload::OnAudioIn_Implementation(
	USoundWave* Image
)
{
	
	return true;
}

USoundWave* UAPIServerAudioUpload::OnAudioOut_Implementation(
	const FString& AssetId
)
{
	return nullptr;
}