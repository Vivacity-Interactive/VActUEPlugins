#pragma once

#include "OICProfile.h"

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "OICManagerActor.generated.h"

class UStaticMesh;

struct VACTOIC_API _FOICCacheProfileEntry
{
	UWorld* World;
	TSubclassOf<UOICProfile> Class;

	bool operator==(const _FOICCacheProfileEntry& Other) const
	{
		return World == Other.World
			&& Class == Other.Class;
	}
};

FORCEINLINE uint32 GetTypeHash(const _FOICCacheProfileEntry& Key)
{
	return HashCombine(::PointerHash(Key.World),::PointerHash(Key.Class.Get()));
}

UCLASS()
class VACTOIC_API AOICManagerActor : public AActor
{
	GENERATED_BODY()

	//static const TCHAR InstNameFormat[];

	//static const TCHAR InstNameFormatP[];

	//static const TCHAR ObjNameFormat[];

	//static const TCHAR ObjNameFormatP[];


	TMap<_FOICCacheProfileEntry, TWeakObjectPtr<UOICProfile>> ProfileCache;

public:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	uint8 bReady : 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OIC Manager")
	TArray<FOICProfileEntry> Profiles;
	
public:
	static AOICManagerActor* GetInstance(const UObject* WorldContextObject);

public:	
	AOICManagerActor();

	UFUNCTION(CallInEditor, Category = "OIC Manager")
	void UpdateProfiles();

	UFUNCTION(CallInEditor, Category = "OIC Manager")
	void ClearProfiles();

	FORCEINLINE void ClearProfile(FOICProfileEntry& Profile)
	{
		bool bCleard = false;
		ClearProfile(Profile, bCleard);
	}

	void ClearProfile(FOICProfileEntry& Profile, bool& bCleard);

	void UpdateProfile(UWorld* World, FOICProfileEntry& Profile);

	void UpdateOrInstantiateActor(UWorld* World, const FOICObject& Object, const FOICInstance& Instance, FOICProfileEntry& Profile);

	void UpdateOrInstantiateMesh(UWorld* World, const FOICObject& Object, const FOICInstance& Instance, FOICProfileEntry& Profile);

	void UpdateOrInstantiateParticle(UWorld* World, const FOICObject& Object, const FOICInstance& Instance, FOICProfileEntry& Profile);

	void UpdateOrInstantiateData(UWorld* World, const FOICObject& Object, const FOICInstance& Instance, FOICProfileEntry& Profile);

	void UpdateOrInstantiateComponents(UWorld* World, AActor* Actor, FOICMeta& Meta);

	//void UpdateISMComponent(UWorld* World, UInstancedStaticMeshComponent* ISMComponent, FOICProfileEntry& Profile);

	void UpdateActor(UWorld* World, AActor* Actor, const FOICObject& Object, const FOICInstance& Instance, FOICProfileEntry& Profile, bool bInstanceMate, bool bObjectMeta);

protected:
	static const FName _DefaultRootName;

	virtual void BeginPlay() override;

	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	void _UpdateProfile(UWorld* World, FOICProfileEntry& Profile);

	void _UpdateActorParent(AActor* Actor, FOICProfileEntry& Profile);

	void _CreateDefaultRootComponet(AActor* Actor);
};
