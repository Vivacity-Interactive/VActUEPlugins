#pragma once

#include "VActCharacter.h"

#include "GameFramework/Actor.h"
#include "GameFramework/Controller.h"

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "ActionEvent.generated.h"

class AActor;

UCLASS(editinlinenew, BlueprintType, Blueprintable)
class VACTBASE_API UActionEvent : public UObject
{
	GENERATED_BODY()

public:	
	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	uint8 bRequeue: 1;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	uint8 bCompleted: 1;
	
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	uint8 bSuccess: 1;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	uint8 bInterupted: 1;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	float Duration;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float MaxDuration;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Count;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	TWeakObjectPtr<AVActCharacter> Owner;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	TWeakObjectPtr<AController> OwnerController;
	
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	TWeakObjectPtr<AActor> Target;

public:
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
	void Execute(float DeltaTime);
};
