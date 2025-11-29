#pragma once

#include "Components/PrimitiveComponent.h"
#include "Components/SceneComponent.h"
#include "Animation/AnimationAsset.h"

#include "InputActionValue.h"

#include "CoreMinimal.h"
#include "InteractionEventTypes.generated.h"

class UPremitiveComponent;
class USceneComponent;
class UAnimationAsset;

struct FInputActionValue;

UENUM(BlueprintType, meta = (Bitflags, UseEnumValuesAsMaskValuesInEditor = "true"))
enum class EInteractionEvent : uint8
{
	None = 0,
	OnTriggered = 1 << 0,
	OnStarted = 1 << 1,
	OnOngoing = 1 << 2,
	OnCanceled = 1 << 3,
	OnCompleted = 1 << 4
};

USTRUCT(BlueprintType)
struct VACTVR_API FInteractionEvent
{
	GENERATED_BODY()

	static const FInteractionEvent InvalidInteractionEvent;

	UPROPERTY(EditAnywhere, BlueprintReadWrite);
	uint8 bInvalid : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite);
	uint8 bEnabled : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite);
	uint8 bCause : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite);
	uint8 bActive : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite);
	int64 LastTime;

	UPROPERTY(EditAnywhere, BlueprintReadWrite);
	FName Name;

	UPROPERTY(EditAnywhere, BlueprintReadWrite);
	FName Slot;

	UPROPERTY(EditAnywhere, BlueprintReadWrite);
	int32 Id;

	UPROPERTY(EditAnywhere, BlueprintReadWrite);
	float SecondsSince;

	UPROPERTY(EditAnywhere, BlueprintReadWrite);
	FInputActionValue Value;

	UPROPERTY(EditAnywhere, BlueprintReadWrite);
	TObjectPtr<UPrimitiveComponent> Component;

	UPROPERTY(EditAnywhere, BlueprintReadWrite);
	TObjectPtr<UPrimitiveComponent> Context;

	UPROPERTY(EditAnywhere, BlueprintReadWrite);
	TObjectPtr<AActor> Claimer;

	FInteractionEvent();

	//FInteractionEvent(FInteractionEvent& InOther, FInputActionValue& InValue);

};

USTRUCT(BlueprintType)
struct VACTVR_API FInteractionGrip
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Instanced, meta = (AllowComponentRef))
	TObjectPtr<USceneComponent> Socket;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Instanced, meta = (AllowComponentRef))
	TObjectPtr<UPrimitiveComponent> Collider;

	UPROPERTY(EditAnywhere, BlueprintReadWrite);
	TObjectPtr<UAnimationAsset> Animations;

	bool IsValid() const;

	USceneComponent* GetSocket() const;
};