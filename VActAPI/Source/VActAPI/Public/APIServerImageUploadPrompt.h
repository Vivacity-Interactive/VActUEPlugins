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
#include "APIServerImageUploadPrompt.generated.h"

class UAPICallback;
class UAPIRoute;
struct FAPIEntry;
class UAPIInstance;

UCLASS(Blueprintable, BlueprintType, EditInlineNew)
class VACTAPI_API UAPIServerImageUploadPrompt : public UAPICallback
{
    GENERATED_BODY()

    static const FString DefaultPrompt;

public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    uint8 bSecurityIntersect : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    uint8 bUseFile : 1;

    UPROPERTY(EditAnywhere, meta = (MultiLine = true))
    FFilePath File;

    UPROPERTY(EditAnywhere, meta = (MultiLine = true))
    FString Prompt;

    UPROPERTY(EditAnywhere, meta = (MultiLine = true))
    FString Key;

    UPROPERTY(EditAnywhere, meta = (MultiLine = true))
    FAPIEntry ActionEntry;

#if WITH_EDITORONLY_DATA
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    TArray<EAPIEntryContent> SelectedSupportedFormats;
#endif
    UPROPERTY()
    TSet<EAPIEntryContent> SupportedFormats;

public:
    UAPIServerImageUploadPrompt();

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
    bool OnPromptIn(
        const FString& InPrompt
    );

    UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
    bool OnPromptOut(
        FString& InPrompt
    );

#if WITH_EDITOR
    virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

};