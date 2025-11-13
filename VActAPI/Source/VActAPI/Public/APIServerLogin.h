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
#include "APIServerLogin.generated.h"

class UAPICallback;
class UAPIRoute;
struct FAPIEntry;
class UAPIInstance;

UCLASS(Blueprintable, BlueprintType, EditInlineNew)
class VACTAPI_API UAPIServerLogin : public UAPICallback
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    uint8 bFixedToken : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    uint8 bAsHeader : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString Token;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString Key;

public:
    UAPIServerLogin();

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
    bool OnLoginIn(
        const FString& Value,
        int64 Ip,
        int32 Port
    );

    UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
    bool OnLoginOut(
        int64 Ip,
        int32 Port
    );

};