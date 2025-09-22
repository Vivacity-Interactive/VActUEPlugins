#pragma once

#include "CoreMinimal.h"
#include "VActCamera.generated.h"

// FCameraVector

#define VACTCAMERA_CAMERA_VECTOR_SIZE sizeof(FCameraVector) / sizeof(float)

USTRUCT(BlueprintType)
struct VACTCAMERA_API FCameraVector
{
	GENERATED_BODY()

	FCameraVector();

	FCameraVector(const FCameraVector& Other);

	FCameraVector(float Value);

	FCameraVector& operator=(const FCameraVector& Other);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ClampMin = "0.0", ClampMax = "1.0", UIMin = "0.0", UIMax = "1.0"))
	float Alpha;

	UPROPERTY(Interp, BlueprintReadWrite, meta = (UIMin = "-8.0", UIMax = "8.0"))
	float Exposure;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ClampMin = "0.0", UIMin = "1.0", UIMax = "600.0", SupportDynamicSliderMaxValue = "true"))
	float FocalLengthMin;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ClampMin = "0.0", UIMin = "1.0", UIMax = "600.0", SupportDynamicSliderMaxValue = "true"))
	float FocalLengthMax;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ClampMin = "0.0", UIMin = "0.0", UIMax = "8.0"))
	float FocalLengthSensitivity;

	UPROPERTY(interp, BlueprintReadWrite)
	float FocalDistance;

	UPROPERTY(interp, BlueprintReadWrite, meta = (ClampMin = "0.0", ClampMax = "32.0", UIMin = "1.0"))
	float Aperture;

	UPROPERTY(interp, EditAnywhere, BlueprintReadWrite, meta = (ClampMin = "0.0", ClampMax = "300.0", UIMin = "0.1"))
	float SensorWidth;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FVector FollowSensitivity;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FVector FollowOffset;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FVector FollowOffsetFactor;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FVector Location;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ClampMin = "0.0", UIMin = "0.0"))
    FVector FollowDistanceMax;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ClampMin = "0.0", ClampMax = "1.0", UIMin = "0.0", UIMax = "1.0"))
    FVector FollowDistanceMaxAlpha;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FVector TrackSensitivity;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FRotator TrackOffset;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FVector TrackOffsetFactor;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FRotator Rotation;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ClampMin = "0.0", UIMin = "0.0"))
    FRotator TrackAngleMax;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ClampMin = "0.0", ClampMax = "1.0", UIMin = "0.0", UIMax = "1.0"))
    FVector TrackAngleMaxAlpha;
};

struct VACTCAMERA_API _FCameraVector
{
	union
	{
		FCameraVector Vector;
		float Data[VACTCAMERA_CAMERA_VECTOR_SIZE];
		uint8 Bytes[sizeof(FCameraVector)];
	};

	constexpr int32 Size() const { return VACTCAMERA_CAMERA_VECTOR_SIZE; }
};


// FCameraOperatorVector

#define VACTCAMERA_CAMERA_GRIP_VECTOR_SIZE sizeof(FCameraOperatorVector) / sizeof(float)

USTRUCT(BlueprintType)
struct VACTCAMERA_API FCameraOperatorVector
{
	GENERATED_BODY()

	FCameraOperatorVector();

	FCameraOperatorVector(const FCameraOperatorVector& Other);

	FCameraOperatorVector(float Value);

	FCameraOperatorVector& operator=(const FCameraOperatorVector& Other);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ClampMin = "0.0", ClampMax = "1.0", UIMin = "0.0", UIMax = "1.0"))
	float Alpha;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ClampMin = "0.0", UIMin = "1.0", UIMax = "600.0", SupportDynamicSliderMaxValue = "true"))
	float FocalLengt;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FVector Follow;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FRotator Track;
};

struct VACTCAMERA_API _FCameraOperatorVector
{
	union
	{
		FCameraOperatorVector Vector;
		float Data[VACTCAMERA_CAMERA_GRIP_VECTOR_SIZE];
		uint8 Bytes[sizeof(FCameraOperatorVector)];
	};

	constexpr int32 Size() const { return VACTCAMERA_CAMERA_GRIP_VECTOR_SIZE; }
};

struct VACTCAMERA_API FVActCamera
{
    FVActCamera() = delete;

    // FCameraVector

    static FCameraVector Lerp(const FCameraVector& A, const FCameraVector& B, float Alpha);

    static FCameraVector Lerp(const FCameraVector& A, const FCameraVector& B, float Alpha, float Beta);

    static void Interp(FCameraVector& Into, const FCameraVector& Current, const FCameraVector& Target, float DeltaTime, float InterpSpeed);

    static void Interp(FCameraVector& Into, const FCameraVector& Current, const FCameraVector& Target, float DeltaTime, const FCameraVector& InterpSpeed);

    // FCameraOperatorVector

    static FCameraOperatorVector Lerp(const FCameraOperatorVector& A, const FCameraOperatorVector& B, float Alpha);

    static FCameraOperatorVector Lerp(const FCameraOperatorVector& A, const FCameraOperatorVector& B, float Alpha, float Beta);

    static void Interp(FCameraOperatorVector& Into, const FCameraOperatorVector& Current, const FCameraOperatorVector& Target, FCameraOperatorVector& Velocity, float SmoothTime, float MaxVelocity, float DeltaTime);

    static void Interp(FCameraOperatorVector& Into, const FCameraOperatorVector& Current, const FCameraOperatorVector& Target, FCameraOperatorVector& Velocity, const FCameraOperatorVector& SmoothTime, float MaxVelocity, float DeltaTime);

    // Helper Math

    static void AssignInto(FRotator& Rotation, const FVector& Vector);

    static void WeightInto(FRotator& Rotation, const FVector& Vector);

    static FRotator RWeight(FRotator& Rotation, const FVector& Vector);
    
    static float FSmoothTo(float Current, float Target, float& Velocity, float SmoothTime, float MaxVelocity, float DeltaTime);

    static float FLerp(const float A, float B, float Alpha, float Beta);

    static FVector VSmoothTo(const FVector& Current, const FVector& Target, FVector& Velocity, float SmoothTime, float MaxVelocity, float DeltaTime);

    static FVector VSmoothTo(const FVector& Current, const FVector& Target, FVector& Velocity, const FVector& SmoothTime, float MaxVelocity, float DeltaTime);

    static FVector VSmoothTo(const FVector& Current, const FVector& Target, FVector& Velocity, const FVector& SmoothTime, const FVector& MaxVelocity, float DeltaTime);

    static FRotator RInterpTo(const FRotator& Current, const FRotator& Target, float DeltaTime, const FVector& InterpSpeed);

    static FRotator RInterpTo(const FRotator& Current, const FRotator& Target, float DeltaTime, const FRotator& InterpSpeed);

    static FVector VClamp(const FVector& Value, const FVector& Min, const FVector& Max);

    static FRotator RClamp(const FRotator& Value, const FRotator& Min, const FRotator& Max);

    static FRotator RLerp(const FRotator& A, const FRotator& B, float Alpha);

    static FRotator RLerp(const FRotator& A, const FRotator& B, const FVector& Alpha);

    template<class T0>
    static void _Lerp(T0& Into, const T0& A, const T0& B, float Alpha);

    static void _Lerp(float* Into, const float* A, const float* B, float Alpha, const int32 End, const int32 Start);

    template<class T0>
    static void _Lerp(T0& Into, const T0& A, const T0& B, const T0& Alpha);

    static void _Lerp(float* Into, const float* A, const float* B, const float* Alpha, const int32 End, const int32 Start);

    template<class T0>
    static void _Lerp(T0& Into, const T0& A, const T0& B, float Alpha, float Beta);

    static void _Lerp(float* Into, const float* A, const float* B, const float Alpha, const float Beta, const int32 End, const int32 Start);

    template<class T0>
    static void _Lerp(T0& Into, const T0& A, const T0& B, const T0& Alpha, const T0& Beta);

    static void _Lerp(float* Into, const float* A, const float* B, const float* Alpha, const float* Beta, const int32 End, const int32 Start);

    template<class T0>
    static void _InterpTo(T0& Into, const T0& Current, const T0& Target, float DeltaTime, float InterpSpeed);

    static void _InterpTo(float* Into, const float* Current, const float* Target, float DeltaTime, float InterpSpeed, const int32 End, const int32 Start);

    template<class T0>
    static void _InterpTo(T0& Into, const T0& Current, const T0& Target, float DeltaTime, const T0& InterpSpeed);

    static void _InterpTo(float* Into, const float* Current, const float* Target, float DeltaTime, const float* InterpSpeed, const int32 End, const int32 Start);

    template<class T0>
    static void _IInterpTo(T0& Into, const T0& Current, const T0& Target, float DeltaTime, float InterpSpeed);

    template<class T0>
    static void _IInterpTo(T0& Into, const T0& Current, const T0& Target, float DeltaTime, const T0& InterpSpeed);

    template<class T0>
    static void _SmoothTo(T0& Into, const T0& Current, const T0& Target, T0& Velocity, float SmoothTime, float MaxVelocity, float DeltaTime);

    static void _SmoothTo(float* Into, const float* Current, const float* Target, float* Velocity, float SmoothTime, float MaxVelocity, float DeltaTime, const int32 End, const int32 Start);

    template<class T0>
    static void _SmoothTo(T0& Into, const T0& Current, const T0& Target, T0& Velocity, const T0& SmoothTime, float MaxVelocity, float DeltaTime);

    static void _SmoothTo(float* Into, const float* Current, const float* Target, float* Velocity, const float* SmoothTime, float MaxVelocity, float DeltaTime, const int32 End, const int32 Start);

    template<class T0>
    static void _SmoothTo(T0& Into, const T0& Current, const T0& Target, T0& Velocity, float SmoothTime, const T0& MaxVelocity, float DeltaTime);

    static void _SmoothTo(float* Into, const float* Current, const float* Target, float* Velocity, float SmoothTime, const float* MaxVelocity, float DeltaTime, const int32 End, const int32 Start);

    template<class T0>
    static void _SmoothTo(T0& Into, const T0& Current, const T0& Target, T0& Velocity, const T0& SmoothTime, const T0& MaxVelocity, float DeltaTime);

    static void _SmoothTo(float* Into, const float* Current, const float* Target, float* Velocity, const float* SmoothTime, const float* MaxVelocity, float DeltaTime, const int32 End, const int32 Start);
};


// Generic Vector Helpers Template Impl

template<class T0>
void FVActCamera::_Lerp(T0& Into, const T0& A, const T0& B, float Alpha)
{
    float* _Into = reinterpret_cast<float*>(&Into);
    const float* _A = reinterpret_cast<const float*>(&A);
    const float* _B = reinterpret_cast<const float*>(&B);

    _Lerp(_Into, _A, _B, Alpha, reinterpret_cast<const float*>(&Into + 1) - _Into, 0);
}

template<class T0>
void FVActCamera::_Lerp(T0& Into, const T0& A, const T0& B, const T0& Alpha)
{
    float* _Into = reinterpret_cast<float*>(&Into);
    const float* _A = reinterpret_cast<const float*>(&A);
    const float* _B = reinterpret_cast<const float*>(&B);
    const float* _Alpha = reinterpret_cast<const float*>(&Alpha);

    _Lerp(_Into, _A, _B, _Alpha, reinterpret_cast<const float*>(&Into + 1) - _Into, 0);
}

template<class T0>
void FVActCamera::_Lerp(T0& Into, const T0& A, const T0& B, float Alpha, float Beta)
{
    float* _Into = reinterpret_cast<float*>(&Into);
    const float* _A = reinterpret_cast<const float*>(&A);
    const float* _B = reinterpret_cast<const float*>(&B);

    _Lerp(_Into, _A, _B, Alpha, Beta, reinterpret_cast<const float*>(&Into + 1) - _Into, 0);
}

template<class T0>
void FVActCamera::_Lerp(T0& Into, const T0& A, const T0& B, const T0& Alpha, const T0& Beta)
{
    float* _Into = reinterpret_cast<float*>(&Into);
    const float* _A = reinterpret_cast<const float*>(&A);
    const float* _B = reinterpret_cast<const float*>(&B);
    const float* _Alpha = reinterpret_cast<const float*>(&Alpha);
    const float* _Beta = reinterpret_cast<const float*>(&Beta);

    _Lerp(_Into, _A, _B, _Alpha, _Beta, reinterpret_cast<const float*>(&Into + 1) - _Into, 0);
}

template<class T0>
void FVActCamera::_InterpTo(T0& Into, const T0& Current, const T0& Target, float DeltaTime, float InterpSpeed)
{
    float* _Into = reinterpret_cast<float*>(&Into);
    const float* _Current = reinterpret_cast<const float*>(&Current);
    const float* _Target = reinterpret_cast<const float*>(&Target);

    _InterpTo(_Into, _Current, _Target, DeltaTime, InterpSpeed, reinterpret_cast<const float*>(&Into + 1) - _Into, 0);
}

template<class T0>
void FVActCamera::_InterpTo(T0& Into, const T0& Current, const T0& Target, float DeltaTime, const T0& InterpSpeed)
{
    float* _Into = reinterpret_cast<float*>(&Into);
    const float* _Current = reinterpret_cast<const float*>(&Current);
    const float* _Target = reinterpret_cast<const float*>(&Target);
    const float* _InterpSpeed = reinterpret_cast<const float*>(&InterpSpeed);

    _InterpTo(_Into, _Current, _Target, DeltaTime, _InterpSpeed, reinterpret_cast<const float*>(&Into + 1) - _Into, 0);
}

template<class T0>
void FVActCamera::_IInterpTo(T0& Into, const T0& Current, const T0& Target, float DeltaTime, float InterpSpeed)
{
    int32* _Into = reinterpret_cast<int32*>(&Into);
    const int32* _Current = reinterpret_cast<const int32*>(&Current);
    const int32* _Target = reinterpret_cast<const int32*>(&Target);

    _IInterpTo(_Into, _Current, _Target, DeltaTime, InterpSpeed, reinterpret_cast<const int32*>(&Into + 1) - _Into, 0);
}

template<class T0>
void FVActCamera::_IInterpTo(T0& Into, const T0& Current, const T0& Target, float DeltaTime, const T0& InterpSpeed)
{
    int32* _Into = reinterpret_cast<float*>(&Into);
    const int32* _Current = reinterpret_cast<const float*>(&Current);
    const int32* _Target = reinterpret_cast<const float*>(&Target);
    const int32* _InterpSpeed = reinterpret_cast<const float*>(&InterpSpeed);

    _IInterpTo(_Into, _Current, _Target, DeltaTime, _InterpSpeed, reinterpret_cast<const int32*>(&Into + 1) - _Into, 0);
}

template<class T0>
void  FVActCamera::_SmoothTo(T0& Into, const T0& Current, const T0& Target, T0& Velocity, float SmoothTime, float MaxVelocity, float DeltaTime)
{
    float* _Into = reinterpret_cast<float*>(&Into);
    const float* _Current = reinterpret_cast<const float*>(&Current);
    const float* _Target = reinterpret_cast<const float*>(&Target);
    float* _Velocity = reinterpret_cast<float*>(&Velocity);

    _SmoothTo(_Into, _Current, _Target, _Velocity, SmoothTime, MaxVelocity, DeltaTime, reinterpret_cast<const float*>(&Into + 1) - _Into, 0);
}

template<class T0>
void  FVActCamera::_SmoothTo(T0& Into, const T0& Current, const T0& Target, T0& Velocity, const T0& SmoothTime, float MaxVelocity, float DeltaTime)
{
    float* _Into = reinterpret_cast<float*>(&Into);
    const float* _Current = reinterpret_cast<const float*>(&Current);
    const float* _Target = reinterpret_cast<const float*>(&Target);
    float* _Velocity = reinterpret_cast<float*>(&Velocity);
    const float* _SmoothTime = reinterpret_cast<const float*>(&SmoothTime);

    _SmoothTo(_Into, _Current, _Target, _Velocity, _SmoothTime, MaxVelocity, DeltaTime, reinterpret_cast<const float*>(&Into + 1) - _Into, 0);
}

template<class T0>
void  FVActCamera::_SmoothTo(T0& Into, const T0& Current, const T0& Target, T0& Velocity, float SmoothTime, const T0& MaxVelocity, float DeltaTime)
{
    float* _Into = reinterpret_cast<float*>(&Into);
    const float* _Current = reinterpret_cast<const float*>(&Current);
    const float* _Target = reinterpret_cast<const float*>(&Target);
    float* _Velocity = reinterpret_cast<float*>(&Velocity);
    const float* _MaxVelocity = reinterpret_cast<const float*>(&MaxVelocity);

    _SmoothTo(_Into, _Current, _Target, _Velocity, SmoothTime, MaxVelocity, DeltaTime, reinterpret_cast<const float*>(&Into + 1) - _Into, 0);
}

template<class T0>
void  FVActCamera::_SmoothTo(T0& Into, const T0& Current, const T0& Target, T0& Velocity, const T0& SmoothTime, const T0& MaxVelocity, float DeltaTime)
{
    float* _Into = reinterpret_cast<float*>(&Into);
    const float* _Current = reinterpret_cast<const float*>(&Current);
    const float* _Target = reinterpret_cast<const float*>(&Target);
    float* _Velocity = reinterpret_cast<float*>(&Velocity);
    const float* _SmoothTime = reinterpret_cast<const float*>(&SmoothTime);
    const float* _MaxVelocity = reinterpret_cast<const float*>(&MaxVelocity);

    _SmoothTo(_Into, _Current, _Target, _Velocity, _SmoothTime, _MaxVelocity, DeltaTime, reinterpret_cast<const float*>(&Into + 1) - _Into, 0);
}