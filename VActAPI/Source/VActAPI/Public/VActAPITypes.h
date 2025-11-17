#pragma once

#include "CoreMinimal.h"
#include "VActAPITypes.generated.h"

class UAPICallback;

UENUM(BlueprintType)
enum class EAPIImageFormat : uint8
{
    Unknown,
    Png,
    Jpeg,
    GrayscaleJpeg,
    Bmp,
    //Ico,
    Exr,
    //Icns,
    //Tga,
    Hdr,
    //Tiff,
    //Dds,
    UEJpeg,
    GrayscaleUEJpeg
};

UENUM(BlueprintType)
enum class EAPIAudioFormat : uint8
{
    Unknown,
    Ogg,
    Wav,
    Acc,
};

UENUM(BlueprintType)
enum class EAPIGammaSpace : uint8
{
    Linear,
    Pow22,
    sRGB,
    Unknown
};

UENUM(BlueprintType)
enum class EAPIImageRawFormat : uint8
{
    G8,
    BGRA8,
    BGRE8,
    RGBA16,
    RGBA16F,
    RGBA32F,
    G16,
    R16F,
    R32F,
    Unknown,
};

UENUM(BlueprintType)
enum class EAPIEntryMode : uint8
{
    Unknown,
    Post,
    Put,
    Get,
    Patch,
    Delete,
    Head,
    Options,
    Any
};

UENUM(BlueprintType)
enum class EAPIEntryContent : uint8
{
    Unknown,
    Image,
    ImagePng,
    ImageJpg,
    ImageBmp,
    ImageExr,
    ImageHdr,
    Audio,
    //AudioOgg,
    AudioWav,
    //AudioAcc,
    //AudioMp3,
    Video,
    VideoMp4,
    VideoMov,
    VideoAvi,
    Model,
    ModelGlb,
    ModelAbc,
    ModelFbx,
    Text,
    JavaScript,
    Html,
    Json,
    Xml,
    Ini,
    Binary,
    Form,
    FormUrl
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTAPI_API FAPIIdentity
{
    GENERATED_BODY()

    static const FString DefaultScopeUnique;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FName Name;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FGuid ServerUnique;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString ScopeUnique;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 Version;

    FAPIIdentity();

    FAPIIdentity(FString ServerUnique, FString ScopeUnique, int32 InVersion = 1);
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTAPI_API FAPICertificate
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int64 DurationFrom;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int64 DurationTo;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString Certificate;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString PrivateKey;

    FAPICertificate();

    FAPICertificate(FString InCertificate, FString InPrivateKey);
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTAPI_API FAPIToken
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString Id;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString PublicKey;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString ApiKey;

    FAPIToken();
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTAPI_API FAPISecret
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString PrivateKey;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString Passphrase;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString Password;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString Code;

    FAPISecret();
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTAPI_API FAPIEntry
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Settings")
    uint8 bSession : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Settings")
    uint8 bSessionPath : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Settings")
    uint8 bScoped : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Settings")
    uint8 bRespond : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Settings")
    uint8 bReceive : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Settings")
    uint8 bEncrypt : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Settings")
    uint8 bAuthenticate : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Settings")
    uint8 bRequestCode : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FName Name;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString Url;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    EAPIEntryMode Mode;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    EAPIEntryContent Content;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Instanced)
    TObjectPtr<UAPICallback> Callback;

    FAPIEntry();

    FAPIEntry(FString InUrl, EAPIEntryMode InMode = EAPIEntryMode::Get, EAPIEntryContent InContent = EAPIEntryContent::Text);

    FString GetEntryUrl(const FAPIIdentity& InIdentity) const;

    FString GetEntryUrlCode(const FAPIIdentity& InIdentity, const FString& Code, bool bUseCode = true) const;

    FString GetEntryUrlPath(const FAPIIdentity& InIdentity, const FString& PathToken, bool bUsePathToken = true) const;

    FString GetEntryUrl(const FAPIIdentity& InIdentity, const FString& PathToken, const FString& Code) const;

    FString GetEntryUrl(const FAPIIdentity& InIdentity, const FString& PathToken, const FString& Code, bool bUsePathToken, bool bUseCode) const;

    void SecurityIntersectInto(FAPIEntry& Into) const;
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTAPI_API FAPIHash
{
    GENERATED_BODY()

    FSHAHash Data;
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTAPI_API FAPITokenCondition
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Settings")
    uint8 bUser : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Settings")
    uint8 bContext : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Settings")
    uint8 bScope : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Settings")
    uint8 bIp : 1;

    UPROPERTY()
    FGuid UserId;

    UPROPERTY()
    uint32 ContextId;

    UPROPERTY()
    uint32 ScopeId;

    UPROPERTY()
    uint32 Ip;

    FAPITokenCondition();

    FORCEINLINE bool Assert(const int32& InIp) const
    {
        return !bUser && !bContext && !bScope
            && (!bIp || Ip == InIp);
    }

    FORCEINLINE bool Assert(const int32& InContextId, const int32& InIp) const
    {
        return !bUser && !bScope
            && (!bContext || ContextId == InContextId)
            && (!bIp || Ip == InIp);
    }

    FORCEINLINE bool Assert(const FGuid& InUserId, const int32& InContextId, const int32& InScopeId, const int32& InIp) const
    {
        return (!bUser || UserId == InUserId)
            && (!bContext || ContextId == InContextId)
            && (!bScope || ScopeId == InScopeId)
            && (!bIp || Ip == InIp);
    }

    FORCEINLINE bool operator==(const FAPITokenCondition& Other) const
    {
        return Assert(Other.UserId, Other.ContextId, Other.ScopeId, Other.Ip);
    }
};

FORCEINLINE uint32 GetTypeHash(const FAPITokenCondition& Key)
{
    uint32 Hash = 0;
    if (Key.bUser) { Hash = HashCombineFast(Hash, GetTypeHash(Key.UserId)); }
    if (Key.bContext) { Hash = HashCombineFast(Hash, Key.ContextId); };
    if (Key.bScope) { Hash = HashCombineFast(Hash, Key.ScopeId); };
    if (Key.bIp) { Hash = HashCombineFast(Hash, Key.Ip); }
    return Hash;
}

USTRUCT(BlueprintType, Blueprintable)
struct VACTAPI_API FAPITokenEvent
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Duration;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 Count;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FAPITokenCondition Condition;

    FAPITokenEvent();

    FAPITokenEvent(float InDuration, int32 InCount = 1);

    FAPITokenEvent(int32 InCount);
};

USTRUCT(BlueprintType, Blueprintable)
struct VACTAPI_API FAPIUser
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    uint8 bAuthenticated : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FAPITokenEvent SessionEvent;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float AuthenticationTime;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float CodeTime;

    UPROPERTY(BlueprintReadWrite)
    FAPIHash Hash;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FAPIToken Token;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FAPISecret Secret;

    FAPIUser();

    FAPIUser(FAPIToken InToken);

    FAPIUser(FAPIToken InToken, FAPISecret InSecret);
};

struct VACTAPI_API FAPIConstMultipartSegment
{
    FString Type;

    FString Disposition;

    FString Name;

    TArrayView<const uint8> Headers;

    TArrayView<const uint8> Body;
};