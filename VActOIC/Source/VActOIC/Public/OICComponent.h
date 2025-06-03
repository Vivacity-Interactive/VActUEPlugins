#pragma once

#include "OICProfile.h"
#include "OICManagerActor.h"

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "OICComponent.generated.h"


UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class VACTOIC_API UOICComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "OIC Profile")
	TWeakObjectPtr<AOICManagerActor> OICManager;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OIC Profile")
	FOICProfileEntry Profile;

public:	
	UOICComponent();

	UFUNCTION(CallInEditor, Category = "OIC Profile")
	void UpdateProfile();

protected:
	virtual void OnRegister() override;

	virtual void OnComponentDestroyed(bool bDestroyingHierarchy) override;
		
};
