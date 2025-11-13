#pragma once

#include "VActAPITypes.h"
#include "APIRoute.h"
#include "APIInstance.h"
#include "Misc/Guid.h"

//#include "HttpServerModule.h"
#include "HttpServerResponse.h"
#include "HttpServerRequest.h"

#include "Templates/UniquePtr.h"
#include "CoreMinimal.h"
#include "APICallback.h"
#include "APIServerPing.generated.h"

class UAPICallback;
class UAPIRoute;
struct FAPIEntry;
class UAPIInstance;

UCLASS(Blueprintable, BlueprintType, EditInlineNew)
class VACTAPI_API UAPIServerPing : public UAPICallback
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    uint8 bPingScopeId : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    uint8 bAsHeader : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString Value;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString Key;

public:
    UAPIServerPing();

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
    bool OnPingIn(
        const FString& InValue,
        int64 Ip,
        int32 Port
    );

    UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
    bool OnPingOut(
        int64 Ip,
        int32 Port
    );

};