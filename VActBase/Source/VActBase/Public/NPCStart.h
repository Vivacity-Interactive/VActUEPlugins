#pragma once

#include "VActCharacter.h"

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "Engine/NavigationObjectBase.h"
#include "NPCStart.generated.h"

class ACharacter;

UCLASS(Blueprintable, ClassGroup = Common, hidecategories = Collision)
class VACTBASE_API ANPCStart : public ANavigationObjectBase
{
	GENERATED_BODY()

public:

	UPROPERTY(EditAnywhere)
	TSubclassOf<AVActCharacter> CharacterClass;

	UPROPERTY(EditAnywhere)
	TSubclassOf<AController> CharacterControllerClass;

public:

	ANPCStart(const FObjectInitializer& ObjectInitializer);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Object)
	FName NPCStartTag;

	virtual void BeginPlay() override;

#if WITH_EDITORONLY_DATA
private:
	UPROPERTY()
	TObjectPtr<class UArrowComponent> ArrowComponent;
public:
#endif

#if WITH_EDITORONLY_DATA
	class UArrowComponent* GetArrowComponent() const;
#endif
};
