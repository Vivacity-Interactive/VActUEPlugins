#pragma once

#include "EngineUtils.h"

#include "Components/PrimitiveComponent.h"
#include "Components/SceneComponent.h"

#include "InteractionEventTypes.h"

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "InteractionComponent.generated.h"

class UPremitiveComponent;
class USceneComponent;

UCLASS(Blueprintable, ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class VACTVR_API UInteractionComponent : public UActorComponent
{
	GENERATED_BODY()
	
#if WITH_EDITORONLY_DATA
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Interaction|_DEBUG");
	uint8 _DEBUG_Show_Draw_Limits : 1;
	
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Interaction|_DEBUG");
	FColor _DEBUG_Hint_Color;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Interaction|_DEBUG");
	FColor _DEBUG_Grip_Color;
#endif
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Interaction|Settings");
	uint8 bSnapToNearestSurface : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Interaction|Settings");
	uint8 bSnapToGrip : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Interaction|Settings");
	uint8 bAlignToGrip : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Interaction|Settings");
	uint8 bAllowHijack : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Interaction|Settings");
	uint8 bRadiusRelativeToOwner : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Interaction|Settings");
	float HintRadius;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Interaction|Settings");
	float GripRadius;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Interaction")
	TArray<FInteractionGrip> Grips;

public:
	UInteractionComponent();

	//virtual void OnInput(const FInputActionValue& Value)

#if WITH_EDITOR
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;

	virtual void _DEBUG_Tick(float DeltaTime, bool bPresist = false);
#endif

};
