#include "NPCStart.h"
#include "Async/TaskGraphInterfaces.h"
#include "UObject/ConstructorHelpers.h"
#include "Components/ArrowComponent.h"
#include "NPCController.h"
#include "Engine/Texture2D.h"
#include "Components/CapsuleComponent.h"
#include "Components/BillboardComponent.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(NPCStart)

ANPCStart::ANPCStart(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	GetCapsuleComponent()->InitCapsuleSize(40.0f, 92.0f);
	GetCapsuleComponent()->SetShouldUpdatePhysicsVolume(false);

#if WITH_EDITORONLY_DATA
	ArrowComponent = CreateEditorOnlyDefaultSubobject<UArrowComponent>(TEXT("Arrow"));

	if (!IsRunningCommandlet())
	{
		// Structure to hold one-time initialization
		struct FConstructorStatics
		{
			ConstructorHelpers::FObjectFinderOptional<UTexture2D> NPCStartTextureObject;
			FName ID_NPCStart;
			FText NAME_NPCStart;
			FName ID_Navigation;
			FText NAME_Navigation;
			FConstructorStatics()
				: NPCStartTextureObject(TEXT("/Engine/EditorResources/S_Pawn"))
				, ID_NPCStart(TEXT("NPCStart"))
				, NAME_NPCStart(NSLOCTEXT("SpriteCategory", "NPCStart", "NPC Start"))
				, ID_Navigation(TEXT("Navigation"))
				, NAME_Navigation(NSLOCTEXT("SpriteCategory", "Navigation", "Navigation"))
			{
			}
		};
		static FConstructorStatics ConstructorStatics;

		if (GetGoodSprite())
		{
			GetGoodSprite()->Sprite = ConstructorStatics.NPCStartTextureObject.Get();
			GetGoodSprite()->SetRelativeScale3D(FVector(0.5f, 0.5f, 0.5f));
			GetGoodSprite()->SpriteInfo.Category = ConstructorStatics.ID_NPCStart;
			GetGoodSprite()->SpriteInfo.DisplayName = ConstructorStatics.NAME_NPCStart;
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

void ANPCStart::BeginPlay()
{
	const bool bSpawn = CharacterClass && CharacterControllerClass;
	if (bSpawn) 
	{  
		const FTransform Transform = this->GetActorTransform();
		
		AVActCharacter* Character = Cast<AVActCharacter>(GetWorld()->SpawnActor(CharacterClass, &Transform));
		ANPCController* Controller = Cast<ANPCController>(GetWorld()->SpawnActor(CharacterControllerClass, &Transform));
		
		bool bPosses = Character && Controller;
		if (bPosses) 
		{ 
			Controller->Possess(Character); 
		} 
	}
}

#if WITH_EDITORONLY_DATA
/** Returns ArrowComponent subobject **/
UArrowComponent* ANPCStart::GetArrowComponent() const { return ArrowComponent; }
#endif
