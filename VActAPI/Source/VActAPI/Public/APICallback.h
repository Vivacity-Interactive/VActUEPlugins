#pragma once

#include "VActAPITypes.h"
#include "APIRoute.h"
#include "APIInstance.h"


#include "HttpServerModule.h"
#include "HttpServerResponse.h"
#include "HttpServerRequest.h"
#include "Misc/Guid.h"

#include "Templates/UniquePtr.h"
#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "APICallback.generated.h"

class UAPIRoute;
struct FAPIEntry;
class UAPIInstance;

UCLASS(Blueprintable, BlueprintType, EditInlineNew)
class VACTAPI_API UAPICallback : public UObject
{
    GENERATED_BODY()

public:
    UPROPERTY()
    TWeakObjectPtr<UObject> ContextObject;

public:
    UFUNCTION(BlueprintCallable)
    UObject* GetContextObject() const;

public:

    virtual bool OnDataIn(
        const FHttpServerRequest& Request,
        const FAPIEntry& SelfEntry,
        UAPIRoute* Parent,
        UAPIInstance* Instance,
        FGuid& UserId
    );

    virtual bool OnDataOut(
        TUniquePtr<FHttpServerResponse>& Response,
        const FHttpServerRequest& Request,
        const FAPIEntry& SelfEntry,
        UAPIRoute* Parent,
        UAPIInstance* Instance,
        FGuid& UserId
    );

public:

    FORCEINLINE UWorld* _GetWorld() {
        return ContextObject.IsValid() ? GEngine->GetWorldFromContextObject(ContextObject.Get(), EGetWorldErrorMode::LogAndReturnNull) : nullptr;
    }

    UFUNCTION(BlueprintCallable)
    AActor* GetActorWithTag(FName Tag);

    UFUNCTION(BlueprintCallable)
    AActor* GetActorOfClass(TSubclassOf<AActor> ActorClass);

    UFUNCTION(BlueprintCallable)
    AActor* GetActorOfClassWithTag(TSubclassOf<AActor> ActorClass, FName Tag);

    UFUNCTION(BlueprintCallable)
    void GetAllActorsWithTag(FName Tag, TArray<AActor*>& OutActors);

    UFUNCTION(BlueprintCallable)
    void GetAllActorsOfClass(TSubclassOf<AActor> ActorClass, TArray<AActor*>& OutActors);

    UFUNCTION(BlueprintCallable)
    void GetAllActorsOfClassWithTag(TSubclassOf<AActor> ActorClass, FName Tag, TArray<AActor*>& OutActors);
};