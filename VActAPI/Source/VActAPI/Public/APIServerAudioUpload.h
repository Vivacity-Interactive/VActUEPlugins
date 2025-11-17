#pragma once

#include "VActAPITypes.h"
#include "APIRoute.h"
#include "APIInstance.h"
#include "Misc/Guid.h"
#include "VActAPI.h"

#include "Sound/SoundWave.h"

#include "HttpServerResponse.h"
#include "HttpServerRequest.h"

#include "CoreMinimal.h"
#include "APICallback.h"
#include "APIServerAudioUpload.generated.h"

class UAPICallback;
class UAPIRoute;
struct FAPIEntry;
class UAPIInstance;
class USoundWave;

USTRUCT(BlueprintType, Blueprintable)
struct VACTAPI_API FAPIAudio
{
    GENERATED_BODY()
};

UCLASS()
class VACTAPI_API UAPIServerAudioUpload : public UAPICallback
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    uint8 bForceAudioFormat : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    EAPIAudioFormat AudioFormat;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 SampleRate;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 NumChannels;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 Bits;

public:
    UAPIServerAudioUpload();

    virtual bool OnDataIn(
        const FHttpServerRequest& Request,
        const FAPIEntry& SelfEntry,
        UAPIRoute* Parent,
        UAPIInstance* Instance,
        FGuid& UserId
    ) override;

    virtual bool OnDataOut(
        TUniquePtr<FHttpServerResponse>& Response,
        const FHttpServerRequest& Request,
        const FAPIEntry& SelfEntry,
        UAPIRoute* Parent,
        UAPIInstance* Instance,
        FGuid& UserId
    ) override;

public:
    UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
    bool OnAudioIn(
        USoundWave* Audio
    );

    UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
    USoundWave* OnAudioOut(
        const FString& AssetId
    );
};
