#include "VActVR.h"

#include "Engine/EngineTypes.h"
#include "Engine/CollisionProfile.h"
#include "PhysicsEngine/PhysicsSettings.h"
#include "Kismet/GameplayStaticsTypes.h"

const TArray<AActor*> FVActVR::EmptyActorArray = TArray<AActor*>();

void FVActVR::_Unsafe_LocalOffset(FVector& Into, const FVector& Offset, AActor* Actor)
{
	Into = Actor->GetActorLocation()
		+ Actor->GetActorForwardVector() * Offset.X
		+ Actor->GetActorRightVector() * Offset.Y
		+ Actor->GetActorUpVector() * Offset.Z;
}

FVector FVActVR::LocalOffset(const FVector& Offset, AActor* Actor)
{
	FVector Result;
	_Unsafe_LocalOffset(Result, Offset, Actor);
	return Result;
}

void FVActVR::_Unsafe_Overlap(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FQuat& Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, TArray<FOverlapResult>& OutOverlaps, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex)
{
	static const FName LineTraceActors(TEXT("ShapeOverlapMulti"));
	FCollisionQueryParams Params = _ConfigureCollisionParamsOverlap(WorldContextObject, LineTraceActors, Filter, bIgnoreSelf, bTraceComplex);

	FCollisionObjectQueryParams ObjectParams(ObjectTypes);

	UWorld* World = GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::LogAndReturnNull);
	bool const bOverlap = World && World->OverlapMultiByObjectType(OutOverlaps, Origin, Rotation, ObjectParams, Shape, Params);

	Into = bOverlap;
}

void FVActVR::_Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, const FQuat& Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, FHitResult& OutHit, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex)
{
	static const FName LineTraceActors(TEXT("ShapeTraceSingle"));
	FCollisionQueryParams Params = _ConfigureCollisionParamsTrace(WorldContextObject, LineTraceActors, Filter, bIgnoreSelf, bTraceComplex);

	FCollisionObjectQueryParams ObjectParams(ObjectTypes);

	UWorld* World = GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::LogAndReturnNull);
	bool const bHit = World && World->SweepSingleByObjectType(OutHit, Origin, End, Rotation, ObjectParams, Shape, Params);

	Into = bHit;
}

void FVActVR::_Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, FHitResult& OutHit, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex)
{
	static const FName LineTraceActors(TEXT("PointTraceSingle"));
	FCollisionQueryParams Params = _ConfigureCollisionParamsTrace(WorldContextObject, LineTraceActors, Filter, bIgnoreSelf, bTraceComplex);

	FCollisionObjectQueryParams ObjectParams(ObjectTypes);

	UWorld* World = GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::LogAndReturnNull);
	bool const bHit = World && World->LineTraceSingleByObjectType(OutHit, Origin, End, ObjectParams, Params);

	Into = bHit;
}

void FVActVR::_Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, const FQuat& Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, TArray<FHitResult>& OutHits, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex)
{
	static const FName LineTraceActors(TEXT("ShapeTraceMulti"));
	FCollisionQueryParams Params = _ConfigureCollisionParamsTrace(WorldContextObject, LineTraceActors, Filter, bIgnoreSelf, bTraceComplex);

	FCollisionObjectQueryParams ObjectParams(ObjectTypes);

	UWorld* World = GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::LogAndReturnNull);
	bool const bHit = World && World->SweepMultiByObjectType(OutHits, Origin, End, Rotation, ObjectParams, Shape, Params);

	Into = bHit;
}

void FVActVR::_Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, TArray<FHitResult>& OutHits, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex)
{
	static const FName LineTraceActors(TEXT("PointTraceMulti"));
	FCollisionQueryParams Params = _ConfigureCollisionParamsTrace(WorldContextObject, LineTraceActors, Filter, bIgnoreSelf, bTraceComplex);

	FCollisionObjectQueryParams ObjectParams(ObjectTypes);

	UWorld* World = GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::LogAndReturnNull);
	bool const bHit = World && World->LineTraceMultiByObjectType(OutHits, Origin, End, ObjectParams, Params);

	Into = bHit;
}

bool FVActVR::OverlapSphere(const UObject* WorldContextObject, const FVector& Origin, float Radius, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, TArray<FOverlapResult>& OutOverlaps, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex)
{
	bool bTrue = false;
	_Unsafe_Overlap(bTrue, WorldContextObject, Origin, FQuat::Identity, FCollisionShape::MakeSphere(Radius), ObjectTypes, OutOverlaps, Filter, bIgnoreSelf, bTraceComplex);
	return bTrue;
}

bool FVActVR::OverlapPoint(const UObject* WorldContextObject, const FVector& Origin, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, TArray<FOverlapResult>& OutOverlaps, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex)
{
	bool bTrue = false;
	_Unsafe_Overlap(bTrue, WorldContextObject, Origin, FQuat::Identity, FCollisionShape::MakeSphere(KINDA_SMALL_NUMBER), ObjectTypes, OutOverlaps, Filter, bIgnoreSelf, bTraceComplex);
	return bTrue;
}

// Single

bool FVActVR::TraceLine(const UObject* WorldContextObject, const FVector& Origin, FVector& End, float Length, FQuat Rotation, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, FHitResult& OutHit, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex)
{
	bool bTrue = false;
	_Unsafe_Trace(bTrue, WorldContextObject, Origin, End, Rotation, FCollisionShape::MakeCapsule(KINDA_SMALL_NUMBER, Length * 0.5), ObjectTypes, OutHit, Filter, bIgnoreSelf, bTraceComplex);
	return bTrue;
}

bool FVActVR::TraceSphere(const UObject* WorldContextObject, const FVector& Origin, FVector& End, float Radius, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, FHitResult& OutHit, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex)
{
	bool bTrue = false;
	_Unsafe_Trace(bTrue, WorldContextObject, Origin, End, FQuat::Identity, FCollisionShape::MakeSphere(Radius), ObjectTypes, OutHit, Filter, bIgnoreSelf, bTraceComplex);
	return bTrue;
}

bool FVActVR::TracePoint(const UObject* WorldContextObject, const FVector& Origin, FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, FHitResult& OutHit, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex)
{
	bool bTrue = false;
	_Unsafe_Trace(bTrue, WorldContextObject, Origin, End, ObjectTypes, OutHit, Filter, bIgnoreSelf, bTraceComplex);
	return bTrue;
}

// Multi

bool FVActVR::TraceLine(const UObject* WorldContextObject, const FVector& Origin, FVector& End, float Length, FQuat Rotation, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, TArray<FHitResult>& OutHits, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex)
{
	bool bTrue = false;
	_Unsafe_Trace(bTrue, WorldContextObject, Origin, End, Rotation, FCollisionShape::MakeCapsule(KINDA_SMALL_NUMBER, Length * 0.5), ObjectTypes, OutHits, Filter, bIgnoreSelf, bTraceComplex);
	return bTrue;
}

bool FVActVR::TraceSphere(const UObject* WorldContextObject, const FVector& Origin, FVector& End, float Radius, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, TArray<FHitResult>& OutHits, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex)
{
	bool bTrue = false;
	_Unsafe_Trace(bTrue, WorldContextObject, Origin, End, FQuat::Identity, FCollisionShape::MakeSphere(Radius), ObjectTypes, OutHits, Filter, bIgnoreSelf, bTraceComplex);
	return bTrue;
}

bool FVActVR::TracePoint(const UObject* WorldContextObject, const FVector& Origin, FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, TArray<FHitResult>& OutHits, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex)
{
	bool bTrue = false;
	_Unsafe_Trace(bTrue, WorldContextObject, Origin, End, ObjectTypes, OutHits, Filter, bIgnoreSelf, bTraceComplex);
	return bTrue;
}

void FVActVR::_ResolveCollisionParmsIgnored(const UObject* WorldContextObject, FCollisionQueryParams& Params, bool bIgnoreSelf)
{
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
}

FCollisionQueryParams FVActVR::_ConfigureCollisionParamsTrace(const UObject* WorldContextObject, FName InTraceTag, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex)
{
	FCollisionQueryParams Params(InTraceTag, SCENE_QUERY_STAT_ONLY(LineTraceActors), bTraceComplex);
	Params.bReturnPhysicalMaterial = true;
	Params.bReturnFaceIndex = !UPhysicsSettings::Get()->bSuppressFaceRemapTable;
	Params.AddIgnoredActors(Filter);

	_ResolveCollisionParmsIgnored(WorldContextObject, Params, bIgnoreSelf);

	return Params;
}

FCollisionQueryParams FVActVR::_ConfigureCollisionParamsOverlap(const UObject* WorldContextObject, FName InTraceTag, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex)
{
	FCollisionQueryParams Params(InTraceTag, SCENE_QUERY_STAT_ONLY(ShapeOverlapActors), bTraceComplex);
	Params.bReturnPhysicalMaterial = true;
	Params.bReturnFaceIndex = !UPhysicsSettings::Get()->bSuppressFaceRemapTable;
	Params.AddIgnoredActors(Filter);

	_ResolveCollisionParmsIgnored(WorldContextObject, Params, bIgnoreSelf);

	return Params;
}