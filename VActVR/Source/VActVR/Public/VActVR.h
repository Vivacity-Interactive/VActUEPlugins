#pragma once

#include "Engine/EngineTypes.h"
#include "CollisionQueryParams.h"
#include "Engine/HitResult.h"
#include "Engine/OverlapResult.h"

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"

struct FCollisionQueryParams;
struct FHitResult;
struct FOverlapResult;
class AActor;
class UObject;

struct VACTVR_API FVActVR
{
	FVActVR() = delete;

	const static TArray<AActor*> EmptyActorArray;

	static void _Unsafe_LocalOffset(FVector& Into, const FVector& Offset, AActor* Actor);

	static FVector LocalOffset(const FVector& Offset, AActor* Actor);

	static void _Unsafe_Overlap(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FQuat& Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, TArray<FOverlapResult>& OutOverlaps, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex);

	static bool OverlapSphere(const UObject* WorldContextObject, const FVector& Origin, float Radius, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, TArray<FOverlapResult>& OutOverlaps, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex);

	static bool OverlapPoint(const UObject* WorldContextObject, const FVector& Origin, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, TArray<FOverlapResult>& OutOverlaps, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex);


	static void _Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, const FQuat& Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, FHitResult& OutHit, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex);

	static void _Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, FHitResult& OutHit, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex);

	static void _Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, const FQuat& Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, TArray<FHitResult>& OutHits, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex);

	static void _Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, TArray<FHitResult>& OutHits, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex);

	// Single 

	static bool TraceLine(const UObject* WorldContextObject, const FVector& Origin, FVector& End, float Length, FQuat Rotation, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, FHitResult& OutHit, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex);

	static bool TraceSphere(const UObject* WorldContextObject, const FVector& Origin, FVector& End, float Radius, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, FHitResult& OutHit, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex);

	static bool TracePoint(const UObject* WorldContextObject, const FVector& Origin, FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, FHitResult& OutHit, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex);

	// Multi

	static bool TraceLine(const UObject* WorldContextObject, const FVector& Origin, FVector& End, float Length, FQuat Rotation, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, TArray<FHitResult>& OutHits, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex);

	static bool TraceSphere(const UObject* WorldContextObject, const FVector& Origin, FVector& End, float Radius, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, TArray<FHitResult>& OutHits, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex);

	static bool TracePoint(const UObject* WorldContextObject, const FVector& Origin, FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, TArray<FHitResult>& OutHits, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex);


	static FCollisionQueryParams _ConfigureCollisionParamsTrace(const UObject* WorldContextObject, FName InTraceTag, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex);

	static FCollisionQueryParams _ConfigureCollisionParamsOverlap(const UObject* WorldContextObject, FName InTraceTag, const TArray<AActor*>& Filter, bool bIgnoreSelf, bool bTraceComplex);

	static void _ResolveCollisionParmsIgnored(const UObject* WorldContextObject, FCollisionQueryParams& Params, bool bIgnoreSelf);
};