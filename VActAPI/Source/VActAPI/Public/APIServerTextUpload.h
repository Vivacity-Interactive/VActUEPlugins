#pragma once

#include "VActAPITypes.h"
#include "APIRoute.h"
#include "APIInstance.h"
#include "Misc/Guid.h"
#include "VActAPI.h"

#include "HttpServerResponse.h"
#include "HttpServerRequest.h"

#include "CoreMinimal.h"
#include "APICallback.h"
#include "APIServerTextUpload.generated.h"

UCLASS()
class VACTAPI_API UAPIServerTextUpload : public UAPICallback
{
	GENERATED_BODY()
	
public:
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
    bool OnTextIn(
        const FString& Text
    );

    UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
    FString OnTextOut(
        const FString& AssetId
    );
};
