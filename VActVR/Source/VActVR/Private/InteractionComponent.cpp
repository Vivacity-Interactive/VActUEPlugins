#include "InteractionComponent.h"

#if WITH_EDITOR
#include "Kismet/KismetSystemLibrary.h"
#endif


UInteractionComponent::UInteractionComponent()
	: bSnapToNearestSurface(false)
	, bSnapToGrip(false)
	, bAlignToGrip(false)
	, bAllowHijack(false)
	, bRadiusRelativeToOwner(false)
	, HintRadius(10.0f)
	, GripRadius(5.0f)
{
#if WITH_EDITOR
	PrimaryComponentTick.bCanEverTick = true;
	_DEBUG_Show_Draw_Limits = true;
	_DEBUG_Hint_Color = FColor(127, 127, 127);
	_DEBUG_Grip_Color = FColor(211, 211, 211);
#endif
}

#if WITH_EDITOR
void UInteractionComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	if (_DEBUG_Show_Draw_Limits) { _DEBUG_Tick(DeltaTime); }
}

void UInteractionComponent::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
	Super::PostEditChangeProperty(PropertyChangedEvent);
	UKismetSystemLibrary::FlushPersistentDebugLines(this);
	_DEBUG_Tick(0.0f, true);
}

void UInteractionComponent::_DEBUG_Tick(float DeltaTime, bool bPresist)
{
	if (bRadiusRelativeToOwner)
	{
		AActor* Owner = GetOwner();
		if (Owner)
		{
			
			FVector Owner_Location = Owner->GetActorLocation();
			DrawDebugSphere(GetWorld(), Owner_Location, HintRadius, 16, _DEBUG_Hint_Color, bPresist, 0);
			DrawDebugSphere(GetWorld(), Owner_Location, GripRadius, 16, _DEBUG_Grip_Color, bPresist, 0);
			return;
		}
	}
	else
	{
		for (const FInteractionGrip& Grip : Grips)
		{
			USceneComponent* Socket = Grip.GetSocket();
			if (Socket)
			{
				FVector Socket_Location = Socket->GetComponentLocation();
				DrawDebugSphere(GetWorld(), Socket_Location, HintRadius, 16, _DEBUG_Hint_Color, bPresist, 0);
				DrawDebugSphere(GetWorld(), Socket_Location, GripRadius, 16, _DEBUG_Grip_Color, bPresist, 0);
			}
		}
	}
}
#endif