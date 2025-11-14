#pragma once

#include "VActAPITypes.h"
#include "APIRoute.h"
#include "IHttpRouter.h"

#include "Misc/SecureHash.h"

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "APIInstance.generated.h"

class UAPIRoute;

UCLASS(Blueprintable, BlueprintType, EditInlineNew)
class VACTAPI_API UAPIInstance : public UObject
{
    GENERATED_BODY()

    TSharedPtr<FInternetAddr> Address;
    
    TMap<FGuid, FAPIUser> Users;

    // Tokens and Codes needs some guards like ip, context-identifier, user-identifier and use count

    TMap<FGuid, float> Tokens;

    TMap<int64, float> Codes;

    TMap<FSHAHash, FAPIUser> Accounts;

public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    uint8 bTrackEntryHandles : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    uint8 bUseHttps : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    uint8 bEnableTick : 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 MaxCollisionRetry;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString CodeSymbols;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FAPIIdentity Identity;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FAPICertificate Certification;

    //UPROPERTY(EditAnywhere, BlueprintReadWrite)
    //FAPISecret InSecret;

    //UPROPERTY(EditAnywhere, BlueprintReadWrite)
    //FAPIToken InToken;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    TArray<FAPIUser> DefaultUsers;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Instanced)
    TArray<TObjectPtr<UAPIRoute>> Routes;

public:
    UAPIInstance();

    UFUNCTION(BlueprintCallable, Category = "VAct API")
    static void CreateCode(FString& Code, const FString& Symbols);

    UFUNCTION(BlueprintCallable, Category = "VAct API")
    static void HashUserName(const FString& UserName, FAPIHash& Hash);

    UFUNCTION(BlueprintCallable, Category = "VAct API")
    static void HashCode(const FString& Code, int64& Hash);

    UFUNCTION(BlueprintCallable)
    bool GetAddress(FString& OutAddress, int64& OutIp, int32& OutPort, bool bWithPort = true);

    UFUNCTION(BlueprintCallable)
    bool NewCode(FString& Code, float LifeTime);

    UFUNCTION(BlueprintCallable)
    bool NewToken(FGuid& Token, float LifeTime);

    UFUNCTION(BlueprintCallable)
    bool CreateUser(FGuid& UserId, FGuid PreferedUserId, bool bUsePreferedUserId = false);

    UFUNCTION(BlueprintCallable)
    bool FindUserInto(FAPIUser& User, const FGuid& UserId);

    UFUNCTION(BlueprintCallable)
    bool RemoveUser(const FGuid& UserId);

    UFUNCTION(BlueprintCallable)
    bool CommitUser(const FGuid& UserId, bool bAndRemove = false);

    UFUNCTION(BlueprintCallable)
    bool ContainsUser(const FGuid& UserId) const;

    FAPIUser* FindUser(const FGuid& UserId);

public:
    UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
    bool Init(
    );

    UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
    bool DeInit(
    );

    UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
    bool Tick(
        float DeltaTime 
    );

    UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
    bool Bouncer(
        const FString& SessionPathToken
    );

    UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
    bool Guard(
        const FString& SessionToken,
        FGuid& UserId
    );

    UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
    bool Reception(
        const FString& Code
    );

    UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
    bool Encrypt(
        const TArray<uint8>& DataIn,
        UPARAM(ref) TArray<uint8>& DataOut,
        const FGuid& UserId
    );

    UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
    bool Decrypt(
        const TArray<uint8>& DataIn,
        UPARAM(ref) TArray<uint8>& DataOut,
        const FGuid& UserId
    );

    UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
    bool Authenticate(
        const FString& UserName,
        const FString& HashedPassword,
        FString& Token,
        FGuid& UserId
    );

    UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
    bool Session(
        FString& Token,
        const FString& PreferedToken,
        bool bUsePreferedToken,
        FGuid& UserId
    );

    UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
    bool Keeper(
        FString& Token,
        const FString& PreferedToken,
        bool bUsePreferedToken,
        const FGuid& UserId
    );

};