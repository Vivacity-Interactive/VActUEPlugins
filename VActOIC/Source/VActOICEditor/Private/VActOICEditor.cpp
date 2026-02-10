#include "VActOICEditor.h"

#include "Editor.h"
#include "ContentBrowserModule.h"
#include "AssetRegistry/AssetData.h"
#include "IContentBrowserSingleton.h"
#include "UObject/SavePackage.h"
#include "Misc/Paths.h"
#include "EngineUtils.h"
#include "AssetRegistry/AssetRegistryModule.h"

#include "Components/InstancedStaticMeshComponent.h"
#include "Components/StaticMeshComponent.h"

#define _VACT_OIC_EDITOR_EXPORT_MSG_SUCCESS TEXT("Exported %s to %s")
#define _VACT_OIC_EDITOR_EXPORT_MSG_FAILED TEXT("Failed to export %s")
#define _VACT_OIC_EDITOR_GEDITOR_FAILED TEXT("No GEditor found")
#define _VACT_OIC_EDITOR_CMD_MSG_FAILED TEXT("No export arguments where provided")

const TArray<FString> FVActOICEditor::ExportOptionNames = {
	TEXT("None"),
	TEXT("PerLevel"),
	TEXT("PerWorld"),
	TEXT("AllCombined")
};

const TMap<FString, ECmdVActExportOptions> FVActOICEditor::ExportOptionNameToEnum = {
	{ ExportOptionNames[(int32)ECmdVActExportOptions::None], ECmdVActExportOptions::None},
	{ ExportOptionNames[(int32)ECmdVActExportOptions::PerLevel], ECmdVActExportOptions::PerLevel},
	{ ExportOptionNames[(int32)ECmdVActExportOptions::PerWorld], ECmdVActExportOptions::PerWorld},
	{ ExportOptionNames[(int32)ECmdVActExportOptions::AllCombined], ECmdVActExportOptions::AllCombined}
};

void FVActOICEditor::Cmd_ExportToOICAsset(const TArray<FString>& Args)
{
	TArray<FAssetData> SelectedAssets;
	switch (Args.Num())
	{
	case 1:
	{
		if (GEditor)
		{
			ExportToOICAsset(GEditor->GetEditorWorldContext(true).World(), Args[0], ECmdVActExportOptions::None);
		}
		else { UE_LOG(LogTemp, Error, _VACT_OIC_EDITOR_GEDITOR_FAILED); }
		break;
	}
	case 2:
	{
		if (GEditor) {
			const ECmdVActExportOptions* Option = ExportOptionNameToEnum.Find(Args[1]);
			if (Option) { ExportToOICAsset(GEditor->GetEditorWorldContext(true).World(), Args[0], *Option); }
			
		}
		else { UE_LOG(LogTemp, Error, _VACT_OIC_EDITOR_GEDITOR_FAILED); }
		break;
	}
	case 0:;
	default: { UE_LOG(LogTemp, Error, _VACT_OIC_EDITOR_CMD_MSG_FAILED); }
	}
}

void FVActOICEditor::ExportToOICAsset(const UWorld* World, const FString& Path, ECmdVActExportOptions Options)
{
	bool bNewProfile = true;
	UOICProfile* Profile = nullptr;
	UPackage* Package = nullptr;
	TMap<FName, int32> NameToObjectId;
	TArray<FAssetData> SelectedAssets;
	const bool bNone = Options == ECmdVActExportOptions::None;	
	

	TArray<const UWorld*> SelectedWorlds;// = UEditorLevelLibrary::GetSelectedLevels();
	SelectedWorlds.Add(World);

	for (const UWorld* LevelWorld : SelectedWorlds)
	{
		if (!LevelWorld) continue;

		NameToObjectId.Empty();
		FString AssetName = "OIC_" + LevelWorld->GetName();
		FString FullPackageName = FPaths::Combine(Path, AssetName);

		Package = CreatePackage(*FullPackageName);
		Package->FullyLoad();
		Profile = NewObject<UOICProfile>(
			Package,
			UOICProfile::StaticClass(),
			*AssetName,
			RF_Public | RF_Standalone
		);

		Profile->Type = "OIC";
		Profile->Version = "V1";
		Profile->Axis = "UnrealEngine";
		Profile->Name = FName(*AssetName);

		bNewProfile = false;

		for (TActorIterator<AActor> It(LevelWorld); It; ++It)
		{
			AActor* Actor = *It;
			if (!Actor) continue;



			TArray<USceneComponent*> SceneComponents;
			Actor->GetComponents<USceneComponent>(SceneComponents);
			TInlineComponentArray<USceneComponent*> Components(Actor);

			for (USceneComponent* Component : Components)
			{
				if (Component->IsA<UInstancedStaticMeshComponent>())
				{
					UInstancedStaticMeshComponent* Instance = CastChecked<UInstancedStaticMeshComponent>(Component);
					UStaticMesh* Object = Instance->GetStaticMesh();

					int32* ObjectId = NameToObjectId.Find(Object->GetFName());
					if (!ObjectId)
					{
						FOICObject _Object;
						_Object.Mesh = Object;
						_Object.Type = EOICAsset::Particle;
						_Object.Meta = -1;
						ObjectId = &NameToObjectId.Add(Object->GetFName(), Profile->Objects.Add(_Object));
					}

					const int32 End = Instance->GetInstanceCount();
					UE_LOG(LogTemp, Warning, TEXT("%s Has %s Instance Count %d"), *GetNameSafe(Instance), *GetNameSafe(Object), End);
					for (int32 Index = 0; Index < End; ++Index)
					{
						FOICInstance _Instance;
						_Instance.Object = *ObjectId;
						_Instance.Parent = *ObjectId;
						_Instance.Meta = -1;
						Instance->GetInstanceTransform(Index, _Instance.Transform, true);
						_Instance.Id = Profile->Instances.Add(_Instance);
					}
				}
				else if (Component->IsA<UStaticMeshComponent>())
				{
					UStaticMeshComponent* Instance = CastChecked<UStaticMeshComponent>(Component);
					UStaticMesh* Object = Instance->GetStaticMesh();

					int32* ObjectId = NameToObjectId.Find(Object->GetFName());
					if (!ObjectId)
					{
						FOICObject _Object;
						_Object.Mesh = Object;
						_Object.Type = EOICAsset::Mesh;
						_Object.Meta = -1;
						ObjectId = &NameToObjectId.Add(Object->GetFName(), Profile->Objects.Add(_Object));
					}

					FOICInstance _Instance;
					_Instance.Object = *ObjectId;
					_Instance.Parent = -1; // TODO needs fix
					_Instance.Meta = -1;
					_Instance.Transform = Instance->GetComponentTransform();
					_Instance.Id = Profile->Instances.Add(_Instance);
				}
				/*else if (Component->IsA<USkinnedMeshComponent>())
				{

				}
				/*else if (Component->IsA<UShapeComponent>())
				{

				}
				else if (Component->IsA<UParticleSystemComponent>())
				{

				}
				else if (Component->IsA<UAudioComponent>())
				{

				}
				else if (Component->IsA<ULightComponent>())
				{

				}*/
			}
		}

		Profile->InstancesCount = Profile->Instances.Num();

		_SaveOICProfileAsset(Profile, Package);
	}
}

void FVActOICEditor::_ResolveSelected(TArray<FAssetData>& Assets)
{
	FContentBrowserModule& ContentBrowserModule = FModuleManager::LoadModuleChecked< FContentBrowserModule>("ContentBrowser");
	IContentBrowserSingleton& ContentBrowserSingleton = ContentBrowserModule.Get();
	ContentBrowserSingleton.GetSelectedAssets(Assets);
}

void FVActOICEditor::_SaveOICProfileAsset(UOICProfile* Profile, UPackage* Package)
{
	Profile->MarkPackageDirty();
	Package->MarkPackageDirty();
	FAssetRegistryModule::AssetCreated(Profile);
	FString PackageFileName = FPackageName::LongPackageNameToFilename(Package->GetName(), FPackageName::GetAssetPackageExtension());
	FSavePackageArgs SaveArgs;
	SaveArgs.TopLevelFlags = RF_Public | RF_Standalone;
	SaveArgs.bForceByteSwapping = false;
	SaveArgs.bWarnOfLongFilename = true;
	SaveArgs.bSlowTask = false;
	UPackage::SavePackage(
		Package,
		Profile,
		*PackageFileName,
		SaveArgs
	);
}