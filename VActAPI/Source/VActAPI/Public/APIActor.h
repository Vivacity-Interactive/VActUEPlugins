#pragma once

#include "APIComponent.h"

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "APIActor.generated.h"

UCLASS()
class VACTAPI_API AAPIActor : public AActor
{
	GENERATED_BODY()

	UPROPERTY(VisibleDefaultsOnly, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
	USceneComponent* DefaultSceneRoot;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
	UAPIComponent* APIComponent;

public:
	AAPIActor();

protected:
	virtual void BeginPlay() override;

};
