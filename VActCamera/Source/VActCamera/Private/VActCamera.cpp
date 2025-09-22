#include "VActCamera.h"


// CameraVector

FCameraVector::FCameraVector()
	: Alpha(1.0f)
	, Exposure(0.0f)
	, FocalLengthMin(20.0f)
	, FocalLengthMax(100.0f)
	, FocalLengthSensitivity(4.0f)
	, FocalDistance(300.0f)
	, Aperture(2.8f)
	, SensorWidth(36.0f)
	, FollowSensitivity(FVector::OneVector)
	, FollowOffset(FVector::ZeroVector)
	, FollowOffsetFactor(FVector::OneVector)
	, Location(FVector::ZeroVector)
	, FollowDistanceMax(FVector::ZeroVector)
	, FollowDistanceMaxAlpha(FVector::ZeroVector)
	, TrackSensitivity(FVector::OneVector)
    , TrackOffset(FRotator::ZeroRotator)
    , TrackOffsetFactor(FVector::OneVector)
	, Rotation(FRotator::ZeroRotator)
    , TrackAngleMax(FRotator::ZeroRotator)
    , TrackAngleMaxAlpha(FVector::ZeroVector)
{
}

FCameraVector::FCameraVector(const FCameraVector& Other)
	: Alpha(Other.Alpha)
	, Exposure(Other.Exposure)
	, FocalLengthMin(Other.FocalLengthMin)
	, FocalLengthMax(Other.FocalLengthMax)
	, FocalLengthSensitivity(Other.FocalLengthSensitivity)
	, FocalDistance(Other.FocalDistance)
	, Aperture(Other.Aperture)
	, SensorWidth(Other.SensorWidth)
	, FollowSensitivity(Other.FollowSensitivity)
	, FollowOffset(Other.FollowOffset)
	, FollowOffsetFactor(Other.FollowOffsetFactor)
	, Location(Other.Location)
	, FollowDistanceMax(Other.FollowDistanceMax)
	, FollowDistanceMaxAlpha(Other.FollowDistanceMaxAlpha)
    , TrackSensitivity(Other.TrackSensitivity)
    , TrackOffset(Other.TrackOffset)
    , TrackOffsetFactor(Other.FollowOffsetFactor)
    , Rotation(Other.Rotation)
    , TrackAngleMax(Other.TrackAngleMax)
    , TrackAngleMaxAlpha(Other.TrackAngleMaxAlpha)
{
}

FCameraVector::FCameraVector(float Value)
	: Alpha(Value)
	, Exposure(Value)
	, FocalLengthMin(Value)
	, FocalLengthMax(Value)
	, FocalLengthSensitivity(Value)
	, FocalDistance(Value)
	, Aperture(Value)
	, SensorWidth(Value)
	, FollowSensitivity(Value)
	, FollowOffset(Value)
	, FollowOffsetFactor(Value)
	, Location(Value)
	, FollowDistanceMax(Value)
	, FollowDistanceMaxAlpha(Value)
    , TrackSensitivity(Value)
    , TrackOffset(Value)
    , TrackOffsetFactor(Value)
    , Rotation(Value)
    , TrackAngleMax(Value)
    , TrackAngleMaxAlpha(Value)
{
}

FCameraVector& FCameraVector::operator=(const FCameraVector& Other)
{
	if (this != &Other)
	{
		Alpha = Other.Alpha;
		Exposure = Other.Exposure;
		FocalLengthMin = Other.FocalLengthMin;
		FocalLengthMax = Other.FocalLengthMax;
		FocalLengthSensitivity = Other.FocalLengthSensitivity;
		FocalDistance = Other.FocalDistance;
		Aperture = Other.Aperture;
		SensorWidth = Other.SensorWidth;
		FollowSensitivity = Other.FollowSensitivity;
		FollowOffset = Other.FollowOffset;
		FollowOffsetFactor = Other.FollowOffsetFactor;
		Location = Other.Location;
		FollowDistanceMax = Other.FollowDistanceMax;
		FollowDistanceMaxAlpha = Other.FollowDistanceMaxAlpha;
        TrackSensitivity = Other.TrackSensitivity;
        TrackOffset = Other.TrackOffset;
        TrackOffsetFactor = Other.FollowOffsetFactor;
        Rotation = Other.Rotation;
        TrackAngleMax = Other.TrackAngleMax;
        TrackAngleMaxAlpha = Other.TrackAngleMaxAlpha;
	}
	return *this;
}


// CameraOperatorVector

FCameraOperatorVector::FCameraOperatorVector()
	: Alpha(1.0f)
	, FocalLengt(50.0f)
	, Follow(FVector::ZeroVector)
	, Track(FRotator::ZeroRotator)
{
}

FCameraOperatorVector::FCameraOperatorVector(const FCameraOperatorVector& Other)
	: Alpha(Other.Alpha)
	, FocalLengt(Other.FocalLengt)
	, Follow(Other.Follow)
	, Track(Other.Track)
{
}

FCameraOperatorVector::FCameraOperatorVector(float Value)
	: Alpha(Value)
	, FocalLengt(Value)
	, Follow(Value)
	, Track(Value)
{
}

FCameraOperatorVector& FCameraOperatorVector::operator=(const FCameraOperatorVector& Other)
{
	if (this != &Other)
	{
		Alpha = Other.Alpha;
		FocalLengt = Other.FocalLengt;
		Follow = Other.Follow;
		Track = Other.Track;
	}
	return *this;
}


// FCameraVector Math

FCameraVector FVActCamera::Lerp(const FCameraVector& A, const FCameraVector& B, float Alpha)
{
    FCameraVector Into;
    _Lerp(Into, A, B, Alpha);
    return Into;
}

FCameraVector FVActCamera::Lerp(const FCameraVector& A, const FCameraVector& B, float Alpha, float Beta)
{
    FCameraVector Into;
    _Lerp(Into, A, B, Alpha, Beta);
    return Into;
}

void FVActCamera::Interp(FCameraVector& Into, const FCameraVector& Current, const FCameraVector& Target, float DeltaTime, float InterpSpeed)
{
    _InterpTo(Into, Current, Target, DeltaTime, InterpSpeed);
}

void FVActCamera::Interp(FCameraVector& Into, const FCameraVector& Current, const FCameraVector& Target, float DeltaTime, const FCameraVector& InterpSpeed)
{
    _InterpTo(Into, Current, Target, DeltaTime, InterpSpeed);
}


// FCameraOperatorVector Math

FCameraOperatorVector FVActCamera::Lerp(const FCameraOperatorVector& A, const FCameraOperatorVector& B, float Alpha)
{
    FCameraOperatorVector Into;
    _Lerp(Into, A, B, Alpha);
    return Into;
}

FCameraOperatorVector FVActCamera::Lerp(const FCameraOperatorVector& A, const FCameraOperatorVector& B, float Alpha, float Beta)
{
    FCameraOperatorVector Into;
    _Lerp(Into, A, B, Alpha, Beta);
    return Into;
}

void FVActCamera::Interp(FCameraOperatorVector& Into, const FCameraOperatorVector& Current, const FCameraOperatorVector& Target, FCameraOperatorVector& Velocity, float SmoothTime, float MaxVelocity, float DeltaTime)
{
    Into.Follow = VSmoothTo(Current.Follow, Target.Follow, Velocity.Follow, 1.0f/SmoothTime, MaxVelocity, DeltaTime);
    Into.Track = FMath::RInterpTo(Current.Track, Target.Track, DeltaTime, SmoothTime);
    Into.FocalLengt = FMath::FInterpTo(Current.FocalLengt, Target.FocalLengt, DeltaTime, SmoothTime);
}

void FVActCamera::Interp(FCameraOperatorVector& Into, const FCameraOperatorVector& Current, const FCameraOperatorVector& Target, FCameraOperatorVector& Velocity, const FCameraOperatorVector& SmoothTime, float MaxVelocity, float DeltaTime)
{
    Into.Follow = VSmoothTo(Current.Follow, Target.Follow, Velocity.Follow, SmoothTime.Follow, MaxVelocity, DeltaTime);
    Into.Track = RInterpTo(Current.Track, Target.Track, DeltaTime, SmoothTime.Track);
    Into.FocalLengt = FMath::FInterpTo(Current.FocalLengt, Target.FocalLengt, DeltaTime, SmoothTime.FocalLengt);
}


// Helper Math

void FVActCamera::AssignInto(FRotator& Rotation, const FVector& Vector)
{
    Rotation.Pitch = Vector.X;
    Rotation.Yaw = Vector.Y;
    Rotation.Roll = Vector.Z;
}

void FVActCamera::WeightInto(FRotator& Rotation, const FVector& Vector)
{
    Rotation.Pitch *= Vector.X;
    Rotation.Yaw *= Vector.Y;
    Rotation.Roll *= Vector.Z;
}

FRotator FVActCamera::RWeight(FRotator& Rotation, const FVector& Vector)
{
    return FRotator(
        Rotation.Pitch * Vector.X,
        Rotation.Yaw * Vector.Y,
        Rotation.Roll * Vector.Z
    );
}

float FVActCamera::FSmoothTo(float Current, float Target, float& Velocity, float SmoothTime, float MaxVelocity, float DeltaTime)
{
    float Into;

    float _Smooth = FMath::Max(0.0001f, SmoothTime);
    float Omega = 2.0f / _Smooth;
    float X = Omega * DeltaTime;
    float Exp = 1.0f / (1.0f + X + 0.48f * X*X + 0.235f * X*X*X);
    float Gamma = 1.0f / DeltaTime;

    float _Change = Current - Target;
    float _Target = Target;

    float MaxChange = MaxVelocity * _Smooth;
    float MaxChangeSq = MaxChange * MaxChange;

    float SqrLength = _Change * _Change;
    if (SqrLength > MaxChangeSq)
    {
        float Length = FMath::Sqrt(SqrLength);
        float Beta = 1.0f / (Length * MaxChange);
        _Change *= Beta;
    }

    float NewTarget = Current - _Change;
    //float NewTarget = Current + _Change;
    float Temp = (Velocity + Omega * _Change) * DeltaTime;

    Velocity = (Velocity - Omega * Temp) * Exp;
    Into = NewTarget + (_Change + Temp) * Exp;

    float DiffTarget = _Target - Current;
    float DiffCurrent = Into - _Target;

    float Product = DiffTarget * DiffCurrent;

    //float Sum = Product.X + Product.Y + Product.Z;
    if (Product > KINDA_SMALL_NUMBER)
    {
        Into = _Target;
        //Velocity = (Into - _Target) * Gamma;
        Velocity = 0.0f;
    }

    return Into;
}

float FVActCamera::FLerp(float A, float B, float Alpha, float Beta)
{
    float Norm = Alpha + Beta;
    return Norm <= KINDA_SMALL_NUMBER
        ? 0.0f
        : (A * Alpha + B * Beta) / Norm;
}

FVector FVActCamera::VSmoothTo(const FVector& Current, const FVector& Target, FVector& Velocity, const FVector& SmoothTime, const FVector& MaxVelocity, float DeltaTime)
{
    float* _Velocity = reinterpret_cast<float*>(&Velocity);
    return FVector(
        FSmoothTo(Current.X, Target.X, _Velocity[0], SmoothTime.X, MaxVelocity.X, DeltaTime),
        FSmoothTo(Current.Y, Target.Y, _Velocity[1], SmoothTime.Y, MaxVelocity.Y, DeltaTime),
        FSmoothTo(Current.Z, Target.Z, _Velocity[2], SmoothTime.Z, MaxVelocity.Z, DeltaTime)
    );
}

FVector FVActCamera::VSmoothTo(const FVector& Current, const FVector& Target, FVector& Velocity, const FVector& SmoothTime, float MaxVelocity, float DeltaTime)
{
    float* _Velocity = reinterpret_cast<float*>(&Velocity);
    return FVector(
        FSmoothTo(Current.X, Target.X, _Velocity[0], SmoothTime.X, MaxVelocity, DeltaTime),
        FSmoothTo(Current.Y, Target.Y, _Velocity[1], SmoothTime.Y, MaxVelocity, DeltaTime),
        FSmoothTo(Current.Z, Target.Z, _Velocity[2], SmoothTime.Z, MaxVelocity, DeltaTime)
    );
}

FVector FVActCamera::VSmoothTo(const FVector& Current, const FVector& Target, FVector& Velocity, float SmoothTime, float MaxVelocity, float DeltaTime)
{
    float* _Velocity = reinterpret_cast<float*>(&Velocity);
    return FVector(
        FSmoothTo(Current.X, Target.X, _Velocity[0], SmoothTime, MaxVelocity, DeltaTime),
        FSmoothTo(Current.Y, Target.Y, _Velocity[1], SmoothTime, MaxVelocity, DeltaTime),
        FSmoothTo(Current.Z, Target.Z, _Velocity[2], SmoothTime, MaxVelocity, DeltaTime)
    );
}

FRotator FVActCamera::RInterpTo(const FRotator& Current, const FRotator& Target, float DeltaTime, const FVector& InterpSpeed)
{
    return FRotator(
        FMath::FInterpTo(Current.Pitch, Target.Pitch, DeltaTime, InterpSpeed.X),
        FMath::FInterpTo(Current.Yaw, Target.Yaw, DeltaTime, InterpSpeed.Y),
        FMath::FInterpTo(Current.Roll, Target.Roll, DeltaTime, InterpSpeed.Z)
    );
}

FVector FVActCamera::VClamp(const FVector& Value, const FVector& Min, const FVector& Max)
{
    return FVector(
        FMath::Clamp(Value.X, Min.X, Max.X),
        FMath::Clamp(Value.Y, Min.Y, Max.Y),
        FMath::Clamp(Value.Z, Min.Z, Max.Z)
    );
}

FRotator FVActCamera::RClamp(const FRotator& Value, const FRotator& Min, const FRotator& Max)
{
    return FRotator(
        FMath::Clamp(Value.Pitch, Min.Pitch, Max.Pitch),
        FMath::Clamp(Value.Yaw, Min.Yaw, Max.Yaw),
        FMath::Clamp(Value.Roll, Min.Roll, Max.Roll)
    );
}

FRotator FVActCamera::RInterpTo(const FRotator& Current, const FRotator& Target, float DeltaTime, const FRotator& InterpSpeed)
{
    return FRotator(
        FMath::FInterpTo(Current.Pitch, Target.Pitch, DeltaTime, InterpSpeed.Pitch),
        FMath::FInterpTo(Current.Yaw, Target.Yaw, DeltaTime, InterpSpeed.Yaw),
        FMath::FInterpTo(Current.Roll, Target.Roll, DeltaTime, InterpSpeed.Roll)
    );
}

FRotator FVActCamera::RLerp(const FRotator& A, const FRotator& B, float Alpha)
{
    FVector _A(A.Pitch, A.Yaw, A.Roll);
    FVector _B(B.Pitch, B.Yaw, B.Roll);
    FVector _Vector = FMath::Lerp(_A, _B, Alpha);
    return FRotator(_Vector.X, _Vector.Y, _Vector.Z);
}

FRotator FVActCamera::RLerp(const FRotator& A, const FRotator& B, const FVector& Alpha)
{
    FVector _A(A.Pitch, A.Yaw, A.Roll);
    FVector _B(B.Pitch, B.Yaw, B.Roll);
    FVector _Vector = FMath::Lerp(_A, _B, Alpha);
    return FRotator(_Vector.X, _Vector.Y, _Vector.Z);
}

// Helper Math Raw Functions

void FVActCamera::_Lerp(float* Into, const float* A, const float* B, float Alpha, const int32 End, const int32 Start)
{
    for (int32 Index = Start; Index < End; ++Index)
    {
        Into[Index] = FMath::Lerp(A[Index], B[Index], Alpha);
    }
}

void FVActCamera::_Lerp(float* Into, const float* A, const float* B, const float* Alpha, const int32 End, const int32 Start)
{
    for (int32 Index = Start; Index < End; ++Index)
    {
        Into[Index] = FMath::Lerp(A[Index], B[Index], Alpha[Index]);
    }
}

void FVActCamera::_Lerp(float* Into, const  float* A, const  float* B, float Alpha, float Beta, const int32 End, const int32 Start)
{
    for (int32 Index = Start; Index < End; ++Index)
    {
        Into[Index] = FLerp(A[Index], B[Index], Alpha, Beta);
    }
}

void FVActCamera::_Lerp(float* Into, const float* A, const float* B, const float* Alpha, const float* Beta, const int32 End, const int32 Start)
{
    for (int32 Index = Start; Index < End; ++Index)
    {
        Into[Index] = FLerp(A[Index], B[Index], Alpha[Index], Beta[Index]);
    }
}

void FVActCamera::_InterpTo(float* Into, const  float* Current, const  float* Target, float DeltaTime, float InterpSpeed, const int32 End, const int32 Start)
{
    for (int32 Index = Start; Index < End; ++Index)
    {
        Into[Index] = FMath::FInterpTo(Current[Index], Target[Index], DeltaTime, InterpSpeed);
    }
}

void FVActCamera::_InterpTo(float* Into, const  float* Current, const  float* Target, float DeltaTime, const float* InterpSpeed, const int32 End, const int32 Start)
{
    for (int32 Index = Start; Index < End; ++Index)
    {
        Into[Index] = FMath::FInterpTo(Current[Index], Target[Index], DeltaTime, InterpSpeed[Index]);
    }
}

void  FVActCamera::_SmoothTo(float* Into, const float* Current, const float* Target, float* Velocity, float SmoothTime, float MaxVelocity, float DeltaTime, const int32 End, const int32 Start)
{
    for (int32 Index = Start; Index < End; ++Index)
    {
        Into[Index] = FSmoothTo(Current[Index], Target[Index], Velocity[Index], SmoothTime, MaxVelocity, DeltaTime);
    }
}

void  FVActCamera::_SmoothTo(float* Into, const float* Current, const float* Target, float* Velocity, const float* SmoothTime, float MaxVelocity, float DeltaTime, const int32 End, const int32 Start)
{
    for (int32 Index = Start; Index < End; ++Index)
    {
        Into[Index] = FSmoothTo(Current[Index], Target[Index], Velocity[Index], SmoothTime[Index], MaxVelocity, DeltaTime);
    }
}

void  FVActCamera::_SmoothTo(float* Into, const float* Current, const float* Target, float* Velocity, float SmoothTime, const float* MaxVelocity, float DeltaTime, const int32 End, const int32 Start)
{
    for (int32 Index = Start; Index < End; ++Index)
    {
        Into[Index] = FSmoothTo(Current[Index], Target[Index], Velocity[Index], SmoothTime, MaxVelocity[Index], DeltaTime);
    }
}

void  FVActCamera::_SmoothTo(float* Into, const float* Current, const float* Target, float* Velocity, const float* SmoothTime, const float* MaxVelocity, float DeltaTime, const int32 End, const int32 Start)
{
    for (int32 Index = Start; Index < End; ++Index)
    {
        Into[Index] = FSmoothTo(Current[Index], Target[Index], Velocity[Index], SmoothTime[Index], MaxVelocity[Index], DeltaTime);
    }
}