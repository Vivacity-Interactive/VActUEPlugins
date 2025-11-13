#pragma once

#include "VActAPITypes.h"
#include "APIRoute.h"
#include "APIInstance.h"
#include "Misc/Guid.h"
#include "VActAPI.h"

#include "Engine/Texture2D.h"

//#include "HttpServerModule.h"
#include "HttpServerResponse.h"
#include "HttpServerRequest.h"

#include "Templates/UniquePtr.h"
#include "CoreMinimal.h"
#include "APICallback.h"
#include "APIServerImageUpload.generated.h"

class UAPICallback;
class UAPIRoute;
struct FAPIEntry;
class UAPIInstance;
class UTexture2D;

UCLASS(Blueprintable, BlueprintType, EditInlineNew)
class VACTAPI_API UAPIServerImageUpload : public UAPICallback
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    uint8 bForceImageRawFormat : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    EAPIImageFormat ImageFormat;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    EAPIImageRawFormat RawImageFormat;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    EAPIGammaSpace GammaSpace;

public:
    UAPIServerImageUpload();

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
    bool OnImageIn(
        UTexture2D* Image
    );

    UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
    UTexture2D* OnImageOut(
        const FString& AssetId
    );

};