#pragma once

#include "VActAPITypes.h"

#include "HttpServerModule.h"

#include "IHttpRouter.h"

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "APIRoute.generated.h"

UCLASS(Blueprintable, BlueprintType)
class VACTAPI_API UAPIRoute : public UObject
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FName Name;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 Port;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 PortMaxOffset;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString Route;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString Domain;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    TArray<FAPIEntry> Entries;

    TSharedPtr<IHttpRouter> HttpRouter;

    TSet<FHttpRouteHandle> Handles;

public:
    UAPIRoute();

    UAPIRoute(FString InRoute, int32 InPort = 8080, FString InDomain = TEXT("localhost"));

};