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

};