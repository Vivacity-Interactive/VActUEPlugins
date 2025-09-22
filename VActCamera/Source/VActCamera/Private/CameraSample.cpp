#include "CameraSample.h"

#if WITH_EDITORONLY_DATA
#include "Async/TaskGraphInterfaces.h"
#include "UObject/ConstructorHelpers.h"
#include "Components/ArrowComponent.h"
#include "Engine/Texture2D.h"
#include "Components/BillboardComponent.h"
#endif

#include "Components/CapsuleComponent.h"

ACameraSample::ACameraSample(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
	, Damp(1.0f)
{
	GetCapsuleComponent()->InitCapsuleSize(FLT_EPSILON, FLT_EPSILON);
	GetCapsuleComponent()->SetShouldUpdatePhysicsVolume(false);

#if WITH_EDITORONLY_DATA
	if (!IsRunningCommandlet())
	{
		struct FConstructorStatics
		{
			ConstructorHelpers::FObjectFinderOptional<UTexture2D> CameraSampleTextureObject;
			FName ID_CameraSample;
			FText NAME_CameraSample;
			FName ID_Navigation;
			FText NAME_Navigation;
			FConstructorStatics()
				: CameraSampleTextureObject(TEXT("/Engine/EditorResources/S_Trigger"))
				, ID_CameraSample(TEXT("CameraSample"))
				, NAME_CameraSample(NSLOCTEXT("SpriteCategory", "CameraSample", "CameraSample"))
				, ID_Navigation(TEXT("Navigation"))
				, NAME_Navigation(NSLOCTEXT("SpriteCategory", "Navigation", "Navigation"))
			{
			}
		};
		static FConstructorStatics ConstructorStatics;

		if (GetGoodSprite())
		{
			GetGoodSprite()->Sprite = ConstructorStatics.CameraSampleTextureObject.Get();
			GetGoodSprite()->SetRelativeScale3D(FVector(0.5f, 0.5f, 0.5f));
			GetGoodSprite()->SpriteInfo.Category = ConstructorStatics.ID_CameraSample;
			GetGoodSprite()->SpriteInfo.DisplayName = ConstructorStatics.NAME_CameraSample;
		}
		if (GetBadSprite())
		{
			GetBadSprite()->SetVisibility(false);
		}
	}

	bIsSpatiallyLoaded = false;
#endif // WITH_EDITORONLY_DATA
}


#if WITH_EDITORONLY_DATA
void ACameraSample::Validate()
{
	Super::Validate();

	if (!bValidate)
	{
		GetBadSprite()->SetVisibility(false);
		GetGoodSprite()->SetVisibility(true);
	}
}

void ACameraSample::__DEBUG_Draw(float DeltaTime)
{

}
#endif