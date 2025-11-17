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
#include "APISequenceCallback.generated.h"

class UAPICallback;
class UAPIRoute;
struct FAPIEntry;
class UAPIInstance;

UCLASS()
class VACTAPI_API UAPISequenceCallback : public UAPICallback
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    TArray<TObjectPtr<UAPICallback>> Sequence;

public:
    UAPISequenceCallback();

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

    UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
    bool OnPartial(
        const int32& ErrorIndex
    );

};
