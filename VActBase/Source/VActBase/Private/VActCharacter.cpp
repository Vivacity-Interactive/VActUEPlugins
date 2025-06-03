#include "VActCharacter.h"
#include "Engine/LocalPlayer.h"
#include "Components/CapsuleComponent.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "GameFramework/Controller.h"
#include "VActBase.h"

#if WITH_EDITORONLY_DATA
#include "DrawDebugHelpers.h"
#include "Kismet/KismetStringLibrary.h"
#include "Kismet/KismetSystemLibrary.h"
#include "Materials/Material.h"
#endif;

DEFINE_LOG_CATEGORY(LogTemplateCharacter);

//////////////////////////////////////////////////////////////////////////
// AVActCharacter

AVActCharacter::AVActCharacter()
{
	GetCapsuleComponent()->InitCapsuleSize(42.f, 96.0f);
		
	bUseControllerRotationPitch = false;
	bUseControllerRotationYaw = false;
	bUseControllerRotationRoll = false;

	GetCharacterMovement()->bOrientRotationToMovement = true;	
	GetCharacterMovement()->RotationRate = FRotator(0.0f, 500.0f, 0.0f);

	GetCharacterMovement()->JumpZVelocity = 700.0f;
	GetCharacterMovement()->AirControl = 0.35f;
	GetCharacterMovement()->MaxWalkSpeed = 200.0f;
	GetCharacterMovement()->MinAnalogWalkSpeed = 20.0f;
	GetCharacterMovement()->BrakingDecelerationWalking = 2000.0f;
	GetCharacterMovement()->BrakingDecelerationFalling = 1500.0f;

	ReachOffset = FVector(30.0f, 0.0f, -10.0f);
	ReachRadius = 120.0f;
	ObserveRadius = 280.0f;
	StareDistance = 40.0f;

	bFaceTarget = false;

#if WITH_EDITORONLY_DATA 
	_DEBUG_Show_Draw_Limits = false;
	_DEBUG_Col_LookAt = FColor(255, 255, 0);
	_DEBUG_Col_FocusAt = FColor(0, 0, 255);
	_DEBUG_Line_Width = 0.8;
#endif;
}

void AVActCharacter::Move(const FVector Movement, const FRotator Rotation)
{
	const FRotator YawRotation(0, Rotation.Yaw, 0);

	const FVector ForwardDirection = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::X);
	const FVector RightDirection = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::Y);
	
	AddMovementInput(ForwardDirection, Movement.Y);
	AddMovementInput(RightDirection, Movement.X);
}

void AVActCharacter::LookMove(const FVector& Movement)
{
	LookAtMove += Movement;
}

void AVActCharacter::Look(const FVector& Location)
{
	LookAt = Location;
}

FVector AVActCharacter::GetLookAtSourceLocation()
{
	USkeletalMeshComponent* MeshComponent = GetMesh();
	const bool bMesh = MeshComponent != nullptr;
	return bMesh ? MeshComponent->GetSocketLocation(LookAtSourceSocketName) : GetRootComponent()->GetSocketLocation(LookAtSourceSocketName);
}

#if WITH_EDITORONLY_DATA 
void AVActCharacter::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	_DEBUG_Draw_Limits();
}

void AVActCharacter::_DEBUG_Draw_Limits()
{
	if (!_DEBUG_Show_Draw_Limits) { return; }
	
	float Line_Length = 30;
	
	FVector Cur_Up = FVector::UpVector * Line_Length;
	FVector Cur_Left = FVector::LeftVector * Line_Length;
	FVector Cur_Forward = FVector::ForwardVector * Line_Length;
	FVector Look_From = GetLookAtSourceLocation();
	FVector Look_Direction = (LookAt - Look_From).GetSafeNormal();
	FVector Focus_Direction = (FocusAt - Look_From).GetSafeNormal();

	DrawDebugSphere(GetWorld(), FVActBase::LocalOffset(ReachOffset, this), ReachRadius, 16, FColor(255, 255, 255), false, 0);

	DrawDebugSphere(GetWorld(), FVActBase::LocalOffset(ReachOffset, this), ObserveRadius, 16, FColor(92, 92, 92), false, 0);

	DrawDebugLine(GetWorld(), LookAt - Cur_Up, LookAt + Cur_Up, _DEBUG_Col_LookAt, false, 0, 0u, _DEBUG_Line_Width);
	DrawDebugLine(GetWorld(), LookAt - Cur_Left, LookAt + Cur_Left, _DEBUG_Col_LookAt, false, 0, 0u, _DEBUG_Line_Width);
	DrawDebugLine(GetWorld(), LookAt - Cur_Forward, LookAt + Cur_Forward, _DEBUG_Col_LookAt, false, 0, 0u, _DEBUG_Line_Width);

	DrawDebugLine(GetWorld(), FocusAt - Cur_Up, FocusAt + Cur_Up, _DEBUG_Col_FocusAt, false, 0, 0u, _DEBUG_Line_Width);
	DrawDebugLine(GetWorld(), FocusAt - Cur_Left, FocusAt + Cur_Left, _DEBUG_Col_FocusAt, false, 0, 0u, _DEBUG_Line_Width);
	DrawDebugLine(GetWorld(), FocusAt - Cur_Forward, FocusAt + Cur_Forward, _DEBUG_Col_FocusAt, false, 0, 0u, _DEBUG_Line_Width);

	DrawDebugLine(GetWorld(), Look_From, Look_Direction * Line_Length + Look_From, _DEBUG_Col_LookAt, false, 0, 0u, _DEBUG_Line_Width);
	DrawDebugLine(GetWorld(), Look_From, Focus_Direction * Line_Length + Look_From, _DEBUG_Col_FocusAt, false, 0, 0u, _DEBUG_Line_Width);
}
#endif;