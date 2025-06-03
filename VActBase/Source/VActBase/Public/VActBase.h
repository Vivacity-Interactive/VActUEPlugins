#pragma once

#include "EngineUtils.h"

#include "Engine/OverlapResult.h"
#include "Engine/HitResult.h"

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "VActBase.generated.h"

struct FOverlapResult;
struct FHitResult;

USTRUCT()
struct VACTBASE_API FVActBase
{
    GENERATED_BODY()

    static const FName _DefaultRootName;

    template<typename InElementType, typename InAllocatorType = FDefaultAllocator>
    static FORCEINLINE int32 NearestActors(TArray<InElementType*, InAllocatorType>& Into, const TArray<FOverlapResult, InAllocatorType>& Overlaps, const FVector& Origin, float MaxRadius = FLT_MAX)
    {
        float BestScore = FLT_MAX;
        return _Unsafe_NearestActors(Into, Overlaps, Origin, BestScore, MaxRadius);
    }

    template<typename InElementType, typename InAllocatorType = FDefaultAllocator>
    static FORCEINLINE int32 NearestActors(TArray<InElementType*, InAllocatorType>& Into, const TArray<FOverlapResult, InAllocatorType>& Overlaps, const FVector& Origin, int32 Count, float& BestScore, float MaxRadius = FLT_MAX)
    {
        Into.SetNum(Count);
        return _Unsafe_NearestActors(Into, Overlaps, Origin, BestScore, MaxRadius);
    }

    static FORCEINLINE int32 NearestActors(TArray<AActor*>& Into, const TArray<FOverlapResult>& Overlaps, const FVector& Origin, int32 Count, float& BestScore, float MaxRadius = FLT_MAX)
    {
        Into.SetNum(Count);
        return _Unsafe_NearestActors(Into, Overlaps, Origin, BestScore, MaxRadius);
    }

    template<typename InElementType = AActor, typename InAllocatorType = FDefaultAllocator>
    static int32 _Unsafe_NearestActors(TArray<InElementType*, InAllocatorType>& Into, const TArray<FOverlapResult, InAllocatorType>& Overlaps, const FVector& Origin, float& BestScore, float MaxRadius = FLT_MAX)
    {
        int32 Index = 0;
        float Score = FLT_MAX, NewScore = FLT_MAX;
        const int32 End = Into.Num();
        const int32 _End = Overlaps.Num();
        for (int32 _Index = 0; _Index < _End; ++_Index)
        {
            const FOverlapResult& Overlap = Overlaps[_Index];
            AActor* Owner = Overlap.GetActor();
            NewScore = FMath::Abs(FVector::Distance(Origin, Owner->GetActorLocation()));
            const bool bPick = NewScore < Score && NewScore <= MaxRadius;
            if (bPick)
            {
                BestScore = FMath::Min(BestScore, NewScore);
                Score = NewScore;
                Into[Index] = Owner;
                ++Index;
                Index = (Index < 0) * End + (Index % End);
            }
        }

        return Index - 1;
    }

    template<typename InElementType, typename InAllocatorType = FDefaultAllocator>
    static FORCEINLINE int32 NearestActors(TArray<InElementType*, InAllocatorType>& Into, const TArray<FHitResult, InAllocatorType>& Hits, const FVector& Origin, float MaxRadius = FLT_MAX)
    {
        float BestScore = FLT_MAX;
        return _Unsafe_NearestActors(Into, Hits, Origin, BestScore, MaxRadius);
    }

    template<typename InElementType, typename InAllocatorType = FDefaultAllocator>
    static FORCEINLINE int32 NearestActors(TArray<InElementType*, InAllocatorType>& Into, const TArray<FHitResult, InAllocatorType>& Hits, const FVector& Origin, int32 Count, float& BestScore, float MaxRadius = FLT_MAX)
    {
        Into.SetNum(Count);
        return _Unsafe_NearestActors(Into, Hits, Origin, BestScore, MaxRadius);
    }

    static FORCEINLINE int32 NearestActors(TArray<AActor*>& Into, const TArray<FHitResult>& Hits, const FVector& Origin, int32 Count, float& BestScore, float MaxRadius = FLT_MAX)
    {
        Into.SetNum(Count);
        return _Unsafe_NearestActors(Into, Hits, Origin, BestScore, MaxRadius);
    }

    template<typename InElementType = AActor, typename InAllocatorType = FDefaultAllocator>
    static int32 _Unsafe_NearestActors(TArray<InElementType*, InAllocatorType>& Into, const TArray<FHitResult, InAllocatorType>& Hits, const FVector& Origin, float& BestScore, float MaxRadius = FLT_MAX)
    {
        int32 Index = 0;
        float Score = FLT_MAX, NewScore = FLT_MAX;
        const int32 End = Into.Num();
        const int32 _End = Hits.Num();
        for (int32 _Index = 0; _Index < End; ++_Index)
        {
            const FHitResult& Overlap = Hits[_Index];
            AActor* Owner = Overlap.GetActor();
            NewScore = FMath::Abs(FVector::Distance(Origin, Owner->GetActorLocation()));
            const bool bPick = NewScore < Score && NewScore <= MaxRadius;
            if (bPick)
            {
                BestScore = FMath::Min(BestScore, NewScore);
                Score = NewScore;
                Into[Index] = Owner;
                ++Index;
                Index = (Index < 0) * End + (Index % End);
            }
        }

        return Index - 1;
    }

    template<typename InElementType, typename InAllocatorType = FDefaultAllocator>
    static FORCEINLINE int32 NearestActors(TArray<InElementType*, InAllocatorType>& Into, const UObject* WorldContextObject, const FVector& Origin, float MaxRadius = FLT_MAX)
    {
        float BestScore = FLT_MAX;
        return _Unsafe_NearestActors(Into, WorldContextObject, Origin, BestScore, MaxRadius);
    }

    template<typename InElementType, typename InAllocatorType = FDefaultAllocator>
    static FORCEINLINE int32 NearestActors(TArray<InElementType*, InAllocatorType>& Into, const UObject* WorldContextObject, const FVector& Origin, int32 Count, float& BestScore, float MaxRadius = FLT_MAX)
    {
        Into.SetNum(Count);
        return _Unsafe_NearestActors(Into, WorldContextObject, Origin, BestScore, MaxRadius);
    }

    static FORCEINLINE int32 NearestActors(TArray<AActor*>& Into, const UObject* WorldContextObject, const FVector& Origin, int32 Count, float& BestScore, float MaxRadius = FLT_MAX)
    {
        Into.SetNum(Count);
        return _Unsafe_NearestActors(Into, WorldContextObject, Origin, BestScore, MaxRadius);
    }

    template<typename InElementType = AActor, typename InAllocatorType = FDefaultAllocator>
    static int32 _Unsafe_NearestActors(TArray<InElementType*, InAllocatorType>& Into, const UObject* WorldContextObject, const FVector& Origin, float& BestScore, float MaxRadius = FLT_MAX)
    {
        UWorld* World = GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::LogAndReturnNull);
        int32 Index = 0;
        float Score = FLT_MAX, NewScore = FLT_MAX;
        const int32 End = Into.Num();
        for (TActorIterator<InElementType> It(World); It; ++It)
        {
            InElementType* Actor = *It;
            NewScore = FMath::Abs(FVector::Distance(Origin, Actor->GetActorLocation()));
            const bool bPick = NewScore < Score && NewScore <= MaxRadius;
            if (bPick)
            {
                BestScore = FMath::Min(BestScore, NewScore);
                Score = NewScore;
                Into[Index] = Actor;
                ++Index;
                Index = (Index < 0) * End + (Index % End);
            }
        }

        return Index - 1;
    }

    template<typename InAllocatorType = FDefaultAllocator>
    static FORCEINLINE int32 NearestPoints(TArray<FVector, InAllocatorType>& Into, const TArray<FVector, InAllocatorType>& Points, const FVector& Origin, float MaxRadius = FLT_MAX)
    {
        float BestScore = FLT_MAX;
        return _Unsafe_NearestPoints(Into, Points, Origin, BestScore, MaxRadius);
    }

    template<typename InAllocatorType = FDefaultAllocator>
    static FORCEINLINE int32 NearestPoints(TArray<FVector, InAllocatorType>& Into, const TArray<FVector, InAllocatorType>& Points, const FVector& Origin, int32 Count, float& BestScore, float MaxRadius = FLT_MAX)
    {
        Into.SetNum(Count);
        return _Unsafe_NearestPoints(Into, Points, Origin, BestScore, MaxRadius);
    }

    static FORCEINLINE int32 NearestPoints(TArray<FVector>& Into, const TArray<FVector>& Points, const FVector& Origin, int32 Count, float& BestScore, float MaxRadius = FLT_MAX)
    {
        Into.SetNum(Count);
        return _Unsafe_NearestPoints(Into, Points, Origin, BestScore, MaxRadius);
    }

    template<typename InAllocatorType = FDefaultAllocator>
    static int32 _Unsafe_NearestPoints(TArray<FVector, InAllocatorType>& Into, const TArray<FVector, InAllocatorType>& Points, const FVector& Origin, float& BestScore, float MaxRadius = FLT_MAX)
    {
        int32 Index = 0;
        float Score = FLT_MAX, NewScore = FLT_MAX;
        const int32 End = Into.Num();
        const int32 _End = Points.Num();
        for (int32 _Index = 0; _Index < End; ++_Index)
        {
            const FVector& Point = Points[_Index];
            NewScore = FMath::Abs(FVector::Distance(Origin, Point));
            const bool bPick = NewScore < Score && NewScore <= MaxRadius;
            if (bPick)
            {
                BestScore = FMath::Min(BestScore, NewScore);
                Score = NewScore;
                Into[Index] = Point;
                ++Index;
                Index = (Index < 0) * End + (Index % End);
            }
        }

        return Index - 1;
    }

    template<typename InAllocatorType = FDefaultAllocator>
    static FORCEINLINE void NearestPoint(FVector& Into, const TArray<FVector, InAllocatorType>& Points, const FVector& Origin, float& BestScore, float MaxRadius = FLT_MAX)
    {
        _Unsafe_NearestPoint(Into, Points, Origin, BestScore, MaxRadius);
    }

    template<typename InAllocatorType = FDefaultAllocator>
    static FORCEINLINE void NearestPoint(FVector& Into, const TArray<FVector, InAllocatorType>& Points, const FVector& Origin, float MaxRadius = FLT_MAX)
    {
        float BestScore = FLT_MAX;
        _Unsafe_NearestPoint(Into, Points, Origin, BestScore, MaxRadius);
    }

    template<typename InAllocatorType = FDefaultAllocator>
    static void _Unsafe_NearestPoint(FVector& Into, const TArray<FVector, InAllocatorType>& Points, const FVector& Origin, float& BestScore, float MaxRadius = FLT_MAX)
    {
        float Score = FLT_MAX, NewScore = FLT_MAX;
        const int32 End = Points.Num();
        for (int32 _Index = 0; _Index < End; ++_Index)
        {
            const FVector& Point = Points[_Index];
            NewScore = FMath::Abs(FVector::Distance(Origin, Point));
            const bool bPick = NewScore < Score && NewScore <= MaxRadius;
            if (bPick)
            {
                BestScore = FMath::Min(BestScore, NewScore);
                Score = NewScore;
                Into = Point;
            }
        }
    }

    static FORCEINLINE FVector LocalOffset(const FVector& Offset, AActor* Actor)
    {
        FVector Into;
        _Unsafe_LocalOffset(Into, Offset, Actor);
        return Into;
    }

    static FORCEINLINE FVector LocalOffset(FVector& Into, const FVector& Offset, AActor* Actor)
    {
        _Unsafe_LocalOffset(Into, Offset, Actor);
        return Into;
    }

    static void _Unsafe_LocalOffset(FVector& Into, const FVector& Offset, AActor* Actor);

    static FORCEINLINE bool OverlapPoint(const UObject* WorldContextObject, const FVector& Origin, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, const TArray<AActor*>& Filter, TArray<FOverlapResult>& OutOverlaps)
    {
        bool Into = false;
        _Unsafe_Overlap(Into, WorldContextObject, Origin, FQuat::Identity, FCollisionShape::MakeSphere(UE_SMALL_NUMBER), ObjectTypes, Filter, OutOverlaps);
        return Into;
    }

    static FORCEINLINE bool OverlapPoint(const UObject* WorldContextObject, const FVector& Origin, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, TArray<AActor*>& OutActors)
    {
        bool Into = false;
        _Unsafe_Overlap(Into, WorldContextObject, Origin, FQuat::Identity, FCollisionShape::MakeSphere(UE_SMALL_NUMBER), ObjectTypes, ActorClassFilter, Filter, OutActors);
        return Into;
    }

    static FORCEINLINE bool OverlapSphere(const UObject* WorldContextObject, const FVector& Origin, float Radius, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, const TArray<AActor*>& Filter, TArray<FOverlapResult>& OutOverlaps)
    {
        bool Into = false;
        _Unsafe_Overlap(Into, WorldContextObject, Origin, FQuat::Identity, FCollisionShape::MakeSphere(Radius), ObjectTypes, Filter, OutOverlaps);
        return Into;
    }

    static FORCEINLINE bool OverlapSphere(const UObject* WorldContextObject, const FVector& Origin, float Radius, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, TArray<AActor*>& OutActors)
    {
        bool Into = false;
        _Unsafe_Overlap(Into, WorldContextObject, Origin, FQuat::Identity, FCollisionShape::MakeSphere(Radius), ObjectTypes, ActorClassFilter, Filter, OutActors);
        return Into;
    }

    static FORCEINLINE void Overlap(bool& Into, const UObject* WorldContextObject, const FVector& Origin, FQuat Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, const TArray<AActor*>& Filter, TArray<FOverlapResult>& OutOverlaps)
    {
        _Unsafe_Overlap(Into, WorldContextObject, Origin, Rotation, Shape, ObjectTypes, Filter, OutOverlaps);
    }

    static FORCEINLINE void Overlap(bool& Into, const UObject* WorldContextObject, const FVector& Origin, FQuat Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, TArray<AActor*>& OutActors)
    {
        _Unsafe_Overlap(Into, WorldContextObject, Origin, Rotation, Shape, ObjectTypes, ActorClassFilter, Filter, OutActors);
    }

    static void _Unsafe_Overlap(bool& Into, const UObject* WorldContextObject, const FVector& Origin, FQuat Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, const TArray<AActor*>& Filter, TArray<FOverlapResult>& OutOverlaps);

    static void _Unsafe_Overlap(bool& Into, const UObject* WorldContextObject, const FVector& Origin, FQuat Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, TArray<AActor*>& OutActors);

    static FORCEINLINE void Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, FQuat Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, TArray<FHitResult>& OutHits, bool bIgnoreSelf = false, bool bTraceComplex = false)
    {
        _Unsafe_Trace(Into, WorldContextObject, Origin, End, Rotation, Shape, ObjectTypes, ActorClassFilter, Filter, OutHits, bIgnoreSelf, bTraceComplex);
    }

    static FORCEINLINE void Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, FQuat Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, FHitResult& OutHit, bool bIgnoreSelf = false, bool bTraceComplex = false)
    {
        _Unsafe_Trace(Into, WorldContextObject, Origin, End, Rotation, Shape, ObjectTypes, ActorClassFilter, Filter, OutHit, bIgnoreSelf, bTraceComplex);
    }

    static FORCEINLINE void Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, FQuat Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, TArray<AActor*>& OutActors, bool bIgnoreSelf = false, bool bTraceComplex = false)
    {
        _Unsafe_Trace(Into, WorldContextObject, Origin, End, Rotation, Shape, ObjectTypes, ActorClassFilter, Filter, OutActors, bIgnoreSelf, bTraceComplex);
    }

    static FORCEINLINE void Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, FQuat Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, AActor*& OutActor, bool bIgnoreSelf = false, bool bTraceComplex = false)
    {
        _Unsafe_Trace(Into, WorldContextObject, Origin, End, Rotation, Shape, ObjectTypes, ActorClassFilter, Filter, OutActor, bIgnoreSelf, bTraceComplex);
    }

    static void _Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, FQuat Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, TArray<FHitResult>& OutHits, bool bIgnoreSelf = false, bool bTraceComplex = false);

    static void _Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, FQuat Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, FHitResult& OutHit, bool bIgnoreSelf = false, bool bTraceComplex = false);

    static void _Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, FQuat Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, TArray<AActor*>& OutActors, bool bIgnoreSelf = false, bool bTraceComplex = false);

    static void _Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, FQuat Rotation, FCollisionShape Shape, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, AActor*& OutActor, bool bIgnoreSelf = false, bool bTraceComplex = false);

    static FORCEINLINE void Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, TArray<FHitResult>& OutHits, bool bIgnoreSelf = false, bool bTraceComplex = false)
    {
        _Unsafe_Trace(Into, WorldContextObject, Origin, End, ObjectTypes, ActorClassFilter, Filter, OutHits, bIgnoreSelf, bTraceComplex);
    }

    static FORCEINLINE void Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, FHitResult& OutHit, bool bIgnoreSelf = false, bool bTraceComplex = false)
    {
        _Unsafe_Trace(Into, WorldContextObject, Origin, End, ObjectTypes, ActorClassFilter, Filter, OutHit, bIgnoreSelf, bTraceComplex);
    }

    static FORCEINLINE void Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, TArray<AActor*>& OutActors, bool bIgnoreSelf = false, bool bTraceComplex = false)
    {
        _Unsafe_Trace(Into, WorldContextObject, Origin, End, ObjectTypes, ActorClassFilter, Filter, OutActors, bIgnoreSelf, bTraceComplex);
    }

    static FORCEINLINE void Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, AActor*& OutActor, bool bIgnoreSelf = false, bool bTraceComplex = false)
    {
        _Unsafe_Trace(Into, WorldContextObject, Origin, End, ObjectTypes, ActorClassFilter, Filter, OutActor, bIgnoreSelf, bTraceComplex);
    }

    static void _Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, TArray<FHitResult>& OutHits, bool bIgnoreSelf = false, bool bTraceComplex = false);

    static void _Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, FHitResult& OutHit, bool bIgnoreSelf = false, bool bTraceComplex = false);

    static void _Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, TArray<AActor*>& OutActors, bool bIgnoreSelf = false, bool bTraceComplex = false);

    static void _Unsafe_Trace(bool& Into, const UObject* WorldContextObject, const FVector& Origin, const FVector& End, TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes, UClass* ActorClassFilter, const TArray<AActor*>& Filter, AActor*& OutActor, bool bIgnoreSelf = false, bool bTraceComplex = false);


    static FCollisionQueryParams _ConfigureCollisionParams(const UObject* WorldContextObject, FName InTraceTag, const TArray<AActor*>& Filter, bool bIgnoreSelf = false, bool bTraceComplex = false);

    static void CreateDefaultRootComponet(AActor* Actor);
};