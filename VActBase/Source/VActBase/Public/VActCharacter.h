#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "Logging/LogMacros.h"
#include "VActCharacter.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogTemplateCharacter, Log, All);

UCLASS(config=Game)
class VACTBASE_API AVActCharacter : public ACharacter
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Character | Attention")
	FName LookAtSourceSocketName;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Character | Attention")
	bool bFaceTarget;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Character | Memory")
	int32 MaxWorkingMemory;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Character | Memory")
	int32 MaxMultitasking;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Character | Memory")
	float ReactionTime;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Character | Attention")
	float ReachRadius;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Character | Attention")
	float ObserveRadius;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Character | Attention")
	float StareDistance;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Character | Attention")
	FVector ReachOffset;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Character | Attention")
	FVector LookAt;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Character | Attention")
	FVector FocusAt;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Character | Attention")
	FVector LookAtMove;

public:
	AVActCharacter();

	void Move(const FVector Movement, const FRotator Rotation);

	void Look(const FVector& Location);

	void LookMove(const FVector& Movement);

	FVector GetLookAtSourceLocation();

#if WITH_EDITORONLY_DATA
protected:
	virtual void Tick(float DeltaTime) override;


public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = _Debug)
	bool _DEBUG_Show_Draw_Limits;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = _Debug)
	float _DEBUG_Line_Width;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = _Debug)
	FColor _DEBUG_Col_LookAt;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = _Debug)
	FColor _DEBUG_Col_FocusAt;

	void _DEBUG_Draw_Limits();
#endif
};
