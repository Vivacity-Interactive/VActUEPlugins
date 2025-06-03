#include "VActBase.h"

#include "Engine/EngineTypes.h"
#include "Engine/CollisionProfile.h"
#include "PhysicsEngine/PhysicsSettings.h"

const FName FVActBase::_DefaultRootName = TEXT("DefaultRootComponent");

void FVActBase::CreateDefaultRootComponet(AActor* Actor)
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

void FVActBase::_Unsafe_LocalOffset(FVector& Into, const FVector& Offset, AActor* Actor)
{
	Into = Actor->GetActorLocation()
		+ Actor->GetActorForwardVector() * Offset.X
		+ Actor->GetActorRightVector() * Offset.Y
		+ Actor->GetActorUpVector() * Offset.Z;
}

void FVActBase::_Unsafe_Overlap(bool& Into, const UObject* WorldContextObject, const FVector& Origin, FQuat Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, const TArray<AActor*>& Filter, TArray<FOverlapResult>& OutOverlaps)
{
	FCollisionQueryParams Params(SCENE_QUERY_STAT(ShapeOverlapActors), false);
	Params.AddIgnoredActors(Filter);

	FCollisionObjectQueryParams ObjectParams(ObjectTypes);

	UWorld* World = GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::LogAndReturnNull);
	bool const bOverlap = World && World->OverlapMultiByObjectType(OutOverlaps, Origin, Rotation, ObjectParams, Shape, Params);

	Into = bOverlap;
}

void FVActBase::_Unsafe_Overlap(bool& Into, const UObject* WorldContextObject, const FVector& Origin, FQuat Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, TArray<AActor*>& OutActors)
{
	OutActors.Empty();

	TArray<FOverlapResult> Overlaps;
	_Unsafe_Overlap(Into, WorldContextObject, Origin, Rotation, Shape, ObjectTypes, Filter, Overlaps);

	for (int32 Index = 0; Index < Overlaps.Num(); ++Index)
	{
		FOverlapResult const& Overlap = Overlaps[Index];
		AActor* const Owner = Overlap.GetActor();
		if (!ActorClassFilter || Owner->IsA(ActorClassFilter))
		{
			OutActors.AddUnique(Owner);
		}
	}

	Into = (OutActors.Num() > 0);
}

FCollisionQueryParams FVActBase::_ConfigureCollisionParams(const UObject* WorldContextObject, FName InTraceTag, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex)
{
	FCollisionQueryParams Params(InTraceTag, SCENE_QUERY_STAT_ONLY(LineTraceActors), bTraceComplex);
	Params.bReturnPhysicalMaterial = true;
	Params.bReturnFaceIndex = !UPhysicsSettings::Get()->bSuppressFaceRemapTable;
	Params.AddIgnoredActors(Filter);

	if (bIgnoreSelf)
	{
		const AActor* SelfActor = Cast<AActor>(WorldContextObject);
		if (SelfActor)
		{
			Params.AddIgnoredActor(SelfActor);
		}
		else
		{
			const UObject* CurrentObject = WorldContextObject;
			while (CurrentObject)
			{
				CurrentObject = CurrentObject->GetOuter();
				SelfActor = Cast<AActor>(CurrentObject);
				if (SelfActor)
				{
					Params.AddIgnoredActor(SelfActor);
					break;
				}
			}
		}
	}

	return Params;
}

void FVActBase::_Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, FQuat Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, TArray<FHitResult>& OutHits, bool bIgnoreSelf, bool bTraceComplex)
{
	static const FName LineTraceActors(TEXT("ShapeTraceMulti"));
	FCollisionQueryParams Params = _ConfigureCollisionParams(WorldContextObject, LineTraceActors, Filter, bIgnoreSelf, bTraceComplex);

	FCollisionObjectQueryParams ObjectParams(ObjectTypes);

	UWorld* World = GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::LogAndReturnNull);
	bool const bHit = World && World->SweepMultiByObjectType(OutHits, Origin, End, Rotation, ObjectParams, Shape, Params);

	Into = bHit;
}

void FVActBase::_Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, FQuat Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, FHitResult& OutHit, bool bIgnoreSelf, bool bTraceComplex)
{
	static const FName LineTraceActors(TEXT("ShapeTraceSingle"));
	FCollisionQueryParams Params = _ConfigureCollisionParams(WorldContextObject, LineTraceActors, Filter, bIgnoreSelf, bTraceComplex);

	FCollisionObjectQueryParams ObjectParams(ObjectTypes);

	UWorld* World = GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::LogAndReturnNull);
	bool const bHit = World && World->SweepSingleByObjectType(OutHit, Origin, End, Rotation, ObjectParams, Shape, Params);

	Into = bHit;
}

void FVActBase::_Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, TArray<FHitResult>& OutHits, bool bIgnoreSelf, bool bTraceComplex)
{
	static const FName LineTraceActors(TEXT("PointTraceMulti"));
	FCollisionQueryParams Params = _ConfigureCollisionParams(WorldContextObject, LineTraceActors, Filter, bIgnoreSelf, bTraceComplex);

	FCollisionObjectQueryParams ObjectParams(ObjectTypes);

	UWorld* World = GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::LogAndReturnNull);
	bool const bHit = World && World->LineTraceMultiByObjectType(OutHits, Origin, End, ObjectParams, Params);

	Into = bHit;
}

void FVActBase::_Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, FHitResult& OutHit, bool bIgnoreSelf, bool bTraceComplex)
{
	static const FName LineTraceActors(TEXT("PointTraceSingle"));
	FCollisionQueryParams Params = _ConfigureCollisionParams(WorldContextObject, LineTraceActors, Filter, bIgnoreSelf, bTraceComplex);

	FCollisionObjectQueryParams ObjectParams(ObjectTypes);

	UWorld* World = GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::LogAndReturnNull);
	bool const bHit = World && World->LineTraceSingleByObjectType(OutHit, Origin, End, ObjectParams, Params);

	Into = bHit;
}




void FVActBase::_Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, FQuat Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, TArray<AActor*>& OutActors, bool bIgnoreSelf, bool bTraceComplex)
{
	OutActors.Empty();

	TArray<FHitResult> Hits;
	_Unsafe_Trace(Into, WorldContextObject, Origin, End, Rotation, Shape, ObjectTypes, ActorClassFilter, Filter, Hits, bIgnoreSelf, bTraceComplex);

	for (int32 Index = 0; Index < Hits.Num(); ++Index)
	{
		FHitResult const& Hit = Hits[Index];
		AActor* const Owner = Hit.GetActor();
		if (!ActorClassFilter || Owner->IsA(ActorClassFilter))
		{
			OutActors.AddUnique(Owner);
		}
	}

	Into = (OutActors.Num() > 0);
}

void FVActBase::_Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, FQuat Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, AActor*& OutActor, bool bIgnoreSelf, bool bTraceComplex)
{
	FHitResult Hit;
	_Unsafe_Trace(Into, WorldContextObject, Origin, End, Rotation, Shape, ObjectTypes, ActorClassFilter, Filter, Hit, bIgnoreSelf, bTraceComplex);

	AActor* const Owner = Hit.GetActor();
	if (!ActorClassFilter || Owner->IsA(ActorClassFilter))
	{
		OutActor = Owner;
	}

	Into = OutActor != nullptr;
}



void FVActBase::_Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, TArray<AActor*>& OutActors, bool bIgnoreSelf, bool bTraceComplex)
{
	OutActors.Empty();

	TArray<FHitResult> Hits;
	_Unsafe_Trace(Into, WorldContextObject, Origin, End, ObjectTypes, ActorClassFilter, Filter, Hits, bIgnoreSelf, bTraceComplex);

	for (int32 Index = 0; Index < Hits.Num(); ++Index)
	{
		FHitResult const& Hit = Hits[Index];
		AActor* const Owner = Hit.GetActor();
		if (!ActorClassFilter || Owner->IsA(ActorClassFilter))
		{
			OutActors.AddUnique(Owner);
		}
	}

	Into = (OutActors.Num() > 0);
}

void FVActBase::_Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, AActor*& OutActor, bool bIgnoreSelf, bool bTraceComplex)
{
	FHitResult Hit;
	_Unsafe_Trace(Into, WorldContextObject, Origin, End, ObjectTypes, ActorClassFilter, Filter, Hit, bIgnoreSelf, bTraceComplex);

	AActor* const Owner = Hit.GetActor();
	if (!ActorClassFilter || Owner->IsA(ActorClassFilter))
	{
		OutActor = Owner;
	}

	Into = OutActor != nullptr;
}