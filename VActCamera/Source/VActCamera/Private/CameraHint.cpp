#include "CameraHint.h"

#if WITH_EDITORONLY_DATA
#include "Async/TaskGraphInterfaces.h"
#include "UObject/ConstructorHelpers.h"
#include "Components/ArrowComponent.h"
#include "Engine/Texture2D.h"
#include "Components/BillboardComponent.h"
#endif

#include "Components/CapsuleComponent.h"

ACameraHint::ACameraHint(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	GetCapsuleComponent()->InitCapsuleSize(FLT_EPSILON, FLT_EPSILON);
	GetCapsuleComponent()->SetShouldUpdatePhysicsVolume(false);

#if WITH_EDITORONLY_DATA
	ArrowComponent = CreateEditorOnlyDefaultSubobject<UArrowComponent>(TEXT("Arrow"));

	if (!IsRunningCommandlet())
	{
		// Structure to hold one-time initialization
		struct FConstructorStatics
		{
			ConstructorHelpers::FObjectFinderOptional<UTexture2D> CameraHintTextureObject;
			FName ID_CameraHint;
			FText NAME_CameraHint;
			FName ID_Navigation;
			FText NAME_Navigation;
			FConstructorStatics()
				: CameraHintTextureObject(TEXT("/Engine/EditorResources/Waypoint"))
				, ID_CameraHint(TEXT("CameraHint"))
				, NAME_CameraHint(NSLOCTEXT("SpriteCategory", "CameraHint", "CameraHint"))
				, ID_Navigation(TEXT("Navigation"))
				, NAME_Navigation(NSLOCTEXT("SpriteCategory", "Navigation", "Navigation"))
			{
			}
		};
		static FConstructorStatics ConstructorStatics;

		if (GetGoodSprite())
		{
			GetGoodSprite()->Sprite = ConstructorStatics.CameraHintTextureObject.Get();
			GetGoodSprite()->SetRelativeScale3D(FVector(0.5f, 0.5f, 0.5f));
			GetGoodSprite()->SpriteInfo.Category = ConstructorStatics.ID_CameraHint;
			GetGoodSprite()->SpriteInfo.DisplayName = ConstructorStatics.NAME_CameraHint;
		}
		if (GetBadSprite())
		{
			GetBadSprite()->SetVisibility(false);
		}

		if (ArrowComponent)
		{
			ArrowComponent->ArrowColor = FColor(150, 200, 255);

			ArrowComponent->ArrowSize = 1.0f;
			ArrowComponent->bTreatAsASprite = true;
			ArrowComponent->SpriteInfo.Category = ConstructorStatics.ID_Navigation;
			ArrowComponent->SpriteInfo.DisplayName = ConstructorStatics.NAME_Navigation;
			ArrowComponent->SetupAttachment(GetCapsuleComponent());
			ArrowComponent->bIsScreenSizeScaled = true;
		}
	}

	bIsSpatiallyLoaded = false;
#endif // WITH_EDITORONLY_DATA
}




void ACameraHint::BeginPlay()
{
	Vector.Location = GetActorLocation();
	Vector.Rotation = GetActorRotation();
}

#if WITH_EDITORONLY_DATA
void ACameraHint::Validate()
{
	Super::Validate();

	if (!bValidate)
	{
		GetBadSprite()->SetVisibility(false);
		GetGoodSprite()->SetVisibility(true);
	}
}

/** Returns ArrowComponent subobject **/
UArrowComponent* ACameraHint::GetArrowComponent() const { return ArrowComponent; }

void ACameraHint::PostEditMove(bool bFinished)
{
	Vector.Location = GetActorLocation();
	Vector.Rotation = GetActorRotation();
}
#endif