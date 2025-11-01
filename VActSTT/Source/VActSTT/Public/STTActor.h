#pragma once

#include "STTComponent.h"

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "STTActor.generated.h"

UCLASS()
class VACTSTT_API ASTTActor : public AActor
{
	GENERATED_BODY()

	UPROPERTY(VisibleDefaultsOnly, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
	USceneComponent* DefaultSceneRoot;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
	USTTComponent* STTComponent;
	
public:	
	ASTTActor();

protected:
	virtual void BeginPlay() override;

};
