#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "APIComponent.generated.h"


UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class VACTAPI_API UAPIComponent : public UActorComponent
{
	GENERATED_BODY()

public:	
	UAPIComponent();

protected:
	virtual void BeginPlay() override;

public:	
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

		
};
