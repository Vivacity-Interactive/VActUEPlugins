#pragma once

#include "OICComponent.h"

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "OICActor.generated.h"

UCLASS()
class VACTOIC_API AOICActor : public AActor
{
	GENERATED_BODY()

	UPROPERTY(VisibleDefaultsOnly, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
	USceneComponent* DefaultSceneRoot;
	
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
	UOICComponent* OICComponent;

public:	
	AOICActor();

protected:
	virtual void BeginPlay() override;

};
