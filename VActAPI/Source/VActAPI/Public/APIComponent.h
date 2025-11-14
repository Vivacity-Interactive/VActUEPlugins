#pragma once

#include "APIInstance.h"

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "APIComponent.generated.h"


UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class VACTAPI_API UAPIComponent : public UActorComponent
{
	GENERATED_BODY()

	float APITickTime;

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	uint8 bUseTickInterval : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float APITickDuration;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Instanced)
	UAPIInstance* APIInstance;

public:	
	UAPIComponent();

protected:
	virtual void BeginPlay() override;

	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

public:	
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

};
