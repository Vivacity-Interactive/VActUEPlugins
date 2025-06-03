#include "OICManagerActor.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "Engine/StaticMeshActor.h"
#include "Kismet/GameplayStatics.h"
#include "PhysicsEngine/BodySetup.h"

const TCHAR AOICManagerActor::InstNameFormat[] = TEXT("%s_%d");
const TCHAR AOICManagerActor::InstNameFormatP[] = TEXT("%s_%s_%d");
const TCHAR AOICManagerActor::ObjNameFormat[] = TEXT("%s_%s");
const TCHAR AOICManagerActor::ObjNameFormatP[] = TEXT("%s_%s");
const FName AOICManagerActor::_DefaultRootName = TEXT("DefaultRootComponent");

AOICManagerActor* AOICManagerActor::GetInstance(const UObject* WorldContextObject)
{
	UWorld* World = GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::LogAndReturnNull);
	AOICManagerActor* Manager = nullptr;
	if (World)
	{
		AActor* PickManager = UGameplayStatics::GetActorOfClass(World, AOICManagerActor::StaticClass());
		if (PickManager) { Manager = Cast<AOICManagerActor>(PickManager); }
		else { Manager = World->SpawnActor<AOICManagerActor>(); }
	}

	return Manager;
}

AOICManagerActor::AOICManagerActor()
{
	bIsEditorOnlyActor = true;

}

void AOICManagerActor::ClearProfiles()
{
	bool bCleard = false;

	for (FOICProfileEntry& Profile : Profiles) { ClearProfile(Profile, bCleard); }

	if (bCleard) { CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS); }
}

void AOICManagerActor::ClearProfile(FOICProfileEntry& Profile, bool& bCleard)
{
	const bool bClear = Profile.Object != nullptr && Profile.bClear;
	if (bClear)
	{
		Profile.Object->Modify();
		for (TPair<FName, FOICTracker>& Pair : Profile.Trackers)
		{
			if (Pair.Value.Actor.IsValid())
			{
				AActor* Actor = Pair.Value.Actor.Get();
				Actor->SetActorLabel(FGuid::NewGuid().ToString());
				Actor->Rename(nullptr, nullptr, REN_DontCreateRedirectors);
				Actor->Destroy();
				bCleard = true;
			}
			Pair.Value.Actor = nullptr;
		}
		Profile.Trackers.Empty();
		Profile.bInitialized = false;
	}
}

void AOICManagerActor::UpdateProfiles()
{
	UWorld* World = GetWorld();

	for (FOICProfileEntry& Profile : Profiles) { UpdateProfile(World, Profile); }
}

void AOICManagerActor::UpdateProfile(UWorld* World, FOICProfileEntry& Profile)
{
	if (World && !Profile.bSkip)
	{
		const bool bCheckCache = Profile.Class != nullptr && Profile.Object == nullptr;
		if (bCheckCache)
		{
			TWeakObjectPtr<UOICProfile>* _Profile = ProfileCache.Find({ World, Profile.Class });
			const bool bProfile = _Profile != nullptr && _Profile->IsValid();
			if (bProfile) { Profile.Object = _Profile->Get(); }
		}

		const bool bInstantiate = bCheckCache;
		if (bInstantiate)
		{
			Profile.Object = NewObject<UOICProfile>(World, Profile.Class);
			Profile.Object->AddToRoot();
			ProfileCache.Add({ World, Profile.Class }, Profile.Object);
		}

		const bool bInitTracking = Profile.bTracked && !Profile.bInitialized;
		const bool bUpdate = Profile.Object && (Profile.bTracked || !Profile.bInitialized);
		if (bUpdate) { _UpdateProfile(World, Profile); Profile.bInitialized = true; }
	}
}

void AOICManagerActor::_UpdateProfile(UWorld* World, FOICProfileEntry& Profile)
{	
	Profile.bUpdateMetas = false;
	const bool bObject = Profile.Object != nullptr && World != nullptr;
	if (bObject)
	{
		for (const FOICInstance& Instance : Profile.Object->Instances)
		{
			const bool bValid = Profile.Object->Objects.IsValidIndex(Instance.Object);
			if (bValid)
			{
				const FOICObject& Object = Profile.Object->Objects[Instance.Object];
				switch (Object.Type)
				{
				case EOICAsset::Actor: UpdateOrInstantiateActor(World, Object, Instance, Profile); break;
				case EOICAsset::Mesh: UpdateOrInstantiateMesh(World, Object, Instance, Profile); break;
				case EOICAsset::Particle: UpdateOrInstantiateParticle(World, Object, Instance, Profile); break;
				case EOICAsset::Data: UpdateOrInstantiateData(World, Object, Instance, Profile); break;
				}
			}
		}

		for (TPair<FName, TWeakObjectPtr<UInstancedStaticMeshComponent>>& Pair : Profile.ISMCs)
		{
			// UpdateISMComponent(World, ISMC, Profile);
			UInstancedStaticMeshComponent* ISMC = nullptr;
			AActor* Actor = nullptr;
			const bool bUpdate = (ISMC = Pair.Value.Get()) != nullptr && (Actor = ISMC->GetOwner()) != nullptr;
			if (bUpdate) { _UpdateActorParent(Actor, Profile); }
		}
	}
}

void AOICManagerActor::_UpdateActorParent(AActor* Actor, FOICProfileEntry& Profile)
{
	if (Actor)
	{
		const bool bUpdateParent = Profile.Parent != nullptr && Profile.Parent != Actor->GetParentActor();
		const bool bRemoveParent = !bUpdateParent && Profile.Parent == nullptr && Actor->GetParentActor() != nullptr;
		if (bUpdateParent)
		{
			FTransform Old = Actor->GetTransform();
			Actor->GetRootComponent()->SetMobility(Profile.Parent->GetRootComponent()->Mobility);
			Actor->AttachToActor(Profile.Parent, FAttachmentTransformRules::KeepRelativeTransform);
			Actor->SetActorRelativeTransform(Old);
		}
		else if (bRemoveParent) { Actor->DetachFromActor(FDetachmentTransformRules::KeepRelativeTransform); }
	}
}

void AOICManagerActor::_CreateDefaultRootComponet(AActor* Actor)
{
	if (Actor)
	{
		UClass* ComponentClass = USceneComponent::StaticClass();
		FName NewName = MakeUniqueObjectName(Actor, ComponentClass, _DefaultRootName);

		USceneComponent* _DefaultRoot = NewObject<USceneComponent>(
			Actor, ComponentClass, NewName, EObjectFlags::RF_NoFlags,
			ComponentClass->GetDefaultObject<USceneComponent>(), false, nullptr);

		if (_DefaultRoot)
		{
			Actor->SetRootComponent(_DefaultRoot);
			Actor->AddInstanceComponent(_DefaultRoot);			
		}
	}
}

void AOICManagerActor::UpdateOrInstantiateActor(UWorld* World, const FOICObject& Object, const FOICInstance& Instance, FOICProfileEntry& Profile)
{
	AActor* Actor = nullptr;
	
	const bool bMayNotExist = Profile.bTracked || !Profile.bInitialized;
	const bool bInstanceMate = Profile.Object->Metas.IsValidIndex(Instance.Meta);
	const bool bObjectMeta = Profile.Object->Metas.IsValidIndex(Object.Meta);
	bool bUpdateMeta = (Profile.bUpdateMetas || !Profile.bInitialized);
	
	if (bMayNotExist)
	{
		FString _InstanceName = FString::Printf(InstNameFormat, *Object.Actor->GetName(), Instance.Id);
		FName _InstanceFName = FName(_InstanceName);
		FOICTracker* Tracker = Profile.Trackers.Find(_InstanceFName);

		const bool bTracker = Tracker != nullptr && Tracker->Actor.IsValid();
		if (bTracker) { Actor = Tracker->Actor.Get(); }
		if (!Actor)
		{
			Actor = World->SpawnActor<AActor>(Object.Actor);
			if (Actor)
			{
				if (Profile.bTracked) { Profile.Trackers.Add(_InstanceFName, { Actor, -1 }); }
				Actor->SetFlags(RF_Transactional);
				Actor->Rename(*_InstanceName, nullptr, REN_DontCreateRedirectors);
				Actor->SetActorLabel(_InstanceName);
				// Maybe OICCallback->OnInitActor(Actor, Instance, Profile);
				bUpdateMeta = true;
			}
		}
	}

	UpdateActor(World, Actor, Object, Instance, Profile, bUpdateMeta && bInstanceMate, bUpdateMeta && bObjectMeta);
}

void AOICManagerActor::UpdateOrInstantiateMesh(UWorld* World, const FOICObject& Object, const FOICInstance& Instance, FOICProfileEntry& Profile)
{
	AStaticMeshActor* Actor = nullptr;
	
	const bool bMayNotExist = Profile.bTracked || !Profile.bInitialized;
	const bool bInstanceMate = Profile.Object->Metas.IsValidIndex(Instance.Meta);
	const bool bObjectMeta = Profile.Object->Metas.IsValidIndex(Object.Meta);
	bool bUpdateMeta = (Profile.bUpdateMetas || !Profile.bInitialized);
	
	if (bMayNotExist)
	{
		FString _InstanceName = Profile.Parent 
			? FString::Printf(InstNameFormatP, *Profile.Parent.GetName(), *Object.Mesh->GetName(), Instance.Id)
			: FString::Printf(InstNameFormat, *Object.Mesh->GetName(), Instance.Id);

		FName _InstanceFName = FName(_InstanceName);
		FOICTracker* Tracker = Profile.Trackers.Find(_InstanceFName);

		const bool bTracker = Tracker != nullptr && Tracker->Actor.IsValid();
		if (bTracker) { Actor = Cast<AStaticMeshActor>(Tracker->Actor.Get()); }
		if (!Actor)
		{
			Actor = World->SpawnActor<AStaticMeshActor>();
			if (Actor)
			{
				if (Profile.bTracked) { Profile.Trackers.Add(_InstanceFName, { Actor, -1 }); }
				Actor->SetFlags(RF_Transactional);
				Actor->Rename(*_InstanceName, nullptr, REN_DontCreateRedirectors);
				Actor->SetActorLabel(_InstanceName);
				Actor->GetStaticMeshComponent()->SetStaticMesh(Object.Mesh);
				// Maybe OICCallback->OnInitMesh(Actor, Instance, Profile);
				bUpdateMeta = true;
			}
		}
	}

	UpdateActor(World, Actor, Object, Instance, Profile, bUpdateMeta && bInstanceMate, bUpdateMeta && bObjectMeta);
}

void AOICManagerActor::UpdateOrInstantiateParticle(UWorld* World, const FOICObject& Object, const FOICInstance& Instance, FOICProfileEntry& Profile)
{
	AActor* Actor = nullptr;
	UInstancedStaticMeshComponent* ISMComponent = nullptr;
	int32 Index = -1;
	bool bNew = false;

	const bool bMayNotExist = Profile.bTracked || !Profile.bInitialized;
	const bool bObjectMeta = Profile.Object->Metas.IsValidIndex(Object.Meta);

	if (bMayNotExist)
	{
		FString _InstanceName = Profile.Parent
			? FString::Printf(InstNameFormatP, *Profile.Parent.GetName(), *Object.Mesh->GetName(), Instance.Id)
			: FString::Printf(InstNameFormat, *Object.Mesh->GetName(), Instance.Id);

		FName _InstanceFName = FName(_InstanceName);
		FOICTracker* Tracker = Profile.Trackers.Find(_InstanceFName);

		const bool bTracker = Tracker != nullptr && Tracker->Actor.IsValid();
		if (bTracker)
		{ 
			Actor = Tracker->Actor.Get();
			Index = Tracker->Index;
		}
		
		FName _ISMCFName = Object.Mesh->GetFName();
		TWeakObjectPtr<UInstancedStaticMeshComponent>* ISMComponentPtr = Profile.ISMCs.Find(_ISMCFName);
		
		bool bISMC = ISMComponentPtr != nullptr && ISMComponentPtr->IsValid();
		if (!bISMC)
		{
			if (!Actor)
			{
				Actor = World->SpawnActor<AActor>();
				if (Actor)
				{
					FString _ActorName = Profile.Parent
						? FString::Printf(ObjNameFormatP, *Profile.Parent.GetName(), *Object.Mesh->GetName())
						: FString::Printf(ObjNameFormat, *Object.Mesh->GetName(), Instance.Id);

					Actor->SetFlags(RF_Transactional);
					Actor->Rename(*_ActorName, nullptr, REN_DontCreateRedirectors);
					Actor->SetActorLabel(_ActorName);
					
					_CreateDefaultRootComponet(Actor);

					//if (bObjectMeta) { UpdateOrInstantiateComponents(World, Actor, Profile.Object->Metas[Object.Meta]); }
				}
			}

			if (Actor)
			{
				UClass* ComponentClass = UInstancedStaticMeshComponent::StaticClass();
				FName NewName = MakeUniqueObjectName(Actor, ComponentClass, _ISMCFName);
				ISMComponent = NewObject<UInstancedStaticMeshComponent>(
					Actor, ComponentClass, NewName, EObjectFlags::RF_NoFlags,
					ComponentClass->GetDefaultObject<UInstancedStaticMeshComponent>(), false, nullptr);

				const bool bComponent = ISMComponent != nullptr;
				if (bComponent)
				{
					Profile.ISMCs.Add(_ISMCFName, ISMComponent);
					ISMComponent->SetStaticMesh(Object.Mesh);
					ISMComponent->RegisterComponent();
					ISMComponent->AttachToComponent(Actor->GetRootComponent(), FAttachmentTransformRules::SnapToTargetIncludingScale);

					UBodySetup* BodySetup = Object.Mesh->GetBodySetup();
					if (BodySetup) { ISMComponent->SetCollisionEnabled(BodySetup->DefaultInstance.GetCollisionEnabled()); }
					
					Actor->AddInstanceComponent(ISMComponent);

					bISMC = true;
				}
			}
			
		} else { ISMComponent = ISMComponentPtr->Get(); }

		const bool bAdd = bISMC && !ISMComponent->IsValidInstance(Index);
		if (bAdd)
		{
			Index = ISMComponent->AddInstance(Instance.Transform, true);
			// Maybe OICCallback->OnInitParticle(ISMComponent, Instance, Profile);
			if (Profile.bTracked) { Profile.Trackers.Add(_InstanceFName, { Actor, Index }); }
			bNew = true;
		}
	}

	const bool bParticle = !bNew && ISMComponent != nullptr && ISMComponent->IsValidInstance(Index);
	if (bParticle)
	{ 
		ISMComponent->UpdateInstanceTransform(Index, Instance.Transform);
	}
}

void AOICManagerActor::UpdateOrInstantiateData(UWorld* World, const FOICObject& Object, const FOICInstance& Instance, FOICProfileEntry& Profile)
{
	UObject* Data = nullptr;

	const bool bMayNotExist = Profile.bTracked || !Profile.bInitialized;

	if (bMayNotExist)
	{
		FName _InstanceFName = FName(FString::Printf(InstNameFormat, *Object.Mesh->GetName(), Instance.Id));
		TObjectPtr<UObject>* DataPtr = Profile.Datas.Find(_InstanceFName);
		if (DataPtr) { Data = DataPtr->Get(); }
		if (!Object.Data)
		{
			Data = NewObject<UObject>(World, Object.Data, _InstanceFName);
			if (Data)
			{
				Data->AddToRoot();
				// Maybe OICCallback->OnInitData(Data, Instance, Profile);
				Profile.Datas.Add(_InstanceFName, Data);
			}
		}
	}
}

void AOICManagerActor::UpdateOrInstantiateComponents(UWorld* World, AActor* Actor, FOICMeta& Meta)
{
	for (const FOICMetaEntry& Entry : Meta.Entries)
	{
		TSubclassOf<UActorComponent> ComponentClass = Entry.Asset;
		if (ComponentClass)
		{
			FName NewName = MakeUniqueObjectName(Actor, ComponentClass, ComponentClass->GetFName());
			UActorComponent* Component = NewObject<UActorComponent>(
				Actor, ComponentClass, NewName, EObjectFlags::RF_NoFlags,
				ComponentClass->GetDefaultObject<UActorComponent>(), false, nullptr);

			const bool bComponent = Component != nullptr;
			if (bComponent)
			{
				Component->RegisterComponent();
				if (Component->IsA<USceneComponent>())
				{
					Cast<USceneComponent>(Component)->AttachToComponent(
						Actor->GetRootComponent(), FAttachmentTransformRules::SnapToTargetIncludingScale);
				}
				Actor->AddInstanceComponent(Component);
				// Maybe OICCallback->OnInitComponent(Component, Entry.Properties, Profile);
			}
		}
	}
}

void AOICManagerActor::UpdateActor(UWorld* World, AActor* Actor, const FOICObject& Object, const FOICInstance& Instance, FOICProfileEntry& Profile, bool bInstanceMate, bool bObjectMeta)
{
	if (Actor)
	{
		if (bInstanceMate) { UpdateOrInstantiateComponents(World, Actor, Profile.Object->Metas[Instance.Meta]); }
		if (bObjectMeta) { UpdateOrInstantiateComponents(World, Actor, Profile.Object->Metas[Object.Meta]); }

		Actor->SetActorTransform(Instance.Transform);
		_UpdateActorParent(Actor, Profile);
	}
}

void AOICManagerActor::BeginPlay()
{
	Super::BeginPlay();
	bReady = true;
	
}

void AOICManagerActor::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Super::EndPlay(EndPlayReason);
	bReady = false;
	for (FOICProfileEntry& Profile : Profiles)
	{ 
		if (Profile.Object)
		{
			for (TPair<FName, TObjectPtr<UObject>>& Pair : Profile.Datas)
			{
				if (Pair.Value) { Pair.Value->RemoveFromRoot(); Pair.Value = nullptr; }
			}
			Profile.Object->RemoveFromRoot(); Profile.Object = nullptr;
		}
	}

}