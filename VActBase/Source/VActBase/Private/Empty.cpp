// Fill out your copyright notice in the Description page of Project Settings.


#include "Empty.h"
#include "Async/TaskGraphInterfaces.h"
#include "UObject/ConstructorHelpers.h"
#include "Components/ArrowComponent.h"
#include "Engine/Texture2D.h"
#include "Components/CapsuleComponent.h"
#include "Components/BillboardComponent.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(Empty)

AEmpty::AEmpty(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	//GetCapsuleComponent()->InitCapsuleSize(40.0f, 92.0f);
	GetCapsuleComponent()->InitCapsuleSize(FLT_EPSILON, FLT_EPSILON);
	GetCapsuleComponent()->SetShouldUpdatePhysicsVolume(false);

#if WITH_EDITORONLY_DATA
	ArrowComponent = CreateEditorOnlyDefaultSubobject<UArrowComponent>(TEXT("Arrow"));

	if (!IsRunningCommandlet())
	{
		// Structure to hold one-time initialization
		struct FConstructorStatics
		{
			ConstructorHelpers::FObjectFinderOptional<UTexture2D> EmptyTextureObject;
			FName ID_Empty;
			FText NAME_Empty;
			FName ID_Navigation;
			FText NAME_Navigation;
			FConstructorStatics()
				: EmptyTextureObject(TEXT("/Engine/EditorResources/Waypoint"))
				, ID_Empty(TEXT("Empty"))
				, NAME_Empty(NSLOCTEXT("SpriteCategory", "Empty", "Empty"))
				, ID_Navigation(TEXT("Navigation"))
				, NAME_Navigation(NSLOCTEXT("SpriteCategory", "Navigation", "Navigation"))
			{
			}
		};
		static FConstructorStatics ConstructorStatics;

		if (GetGoodSprite())
		{
			GetGoodSprite()->Sprite = ConstructorStatics.EmptyTextureObject.Get();
			GetGoodSprite()->SetRelativeScale3D(FVector(0.5f, 0.5f, 0.5f));
			GetGoodSprite()->SpriteInfo.Category = ConstructorStatics.ID_Empty;
			GetGoodSprite()->SpriteInfo.DisplayName = ConstructorStatics.NAME_Empty;
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


void AEmpty::Validate()
{
	Super::Validate();

	if (!bValidate)
	{
		GetBadSprite()->SetVisibility(false);
		GetGoodSprite()->SetVisibility(true);
	}
}

#if WITH_EDITORONLY_DATA
/** Returns ArrowComponent subobject **/
UArrowComponent* AEmpty::GetArrowComponent() const { return ArrowComponent; }
#endif
