#include "InputModifierXAxisPositiveOnly.h"

FInputActionValue UInputModifierXAxisPositiveOnly::ModifyRaw_Implementation(const UEnhancedPlayerInput* PlayerInput, FInputActionValue CurrentValue, float DeltaTime)
{
	FVector Value = CurrentValue.Get<FVector>();
	Value.X = FMath::Clamp(Value.X, 0.0, 1.0);
	return FInputActionValue(Value);//Super::ModifyRaw(PlayerInput, FInputActionValue(Value), DeltaTime);
}