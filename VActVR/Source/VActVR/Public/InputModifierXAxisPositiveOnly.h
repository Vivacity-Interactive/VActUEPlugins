#pragma once

#include "CoreMinimal.h"
#include "InputModifiers.h"
#include "InputModifierXAxisPositiveOnly.generated.h"

UCLASS(NotBlueprintable, meta = (DisplayName = "XAxis Positive Only"))
class VACTVR_API UInputModifierXAxisPositiveOnly : public UInputModifier
{
	GENERATED_BODY()
	
protected:
	virtual FInputActionValue ModifyRaw_Implementation(const UEnhancedPlayerInput* PlayerInput, FInputActionValue CurrentValue, float DeltaTime) override;
};
