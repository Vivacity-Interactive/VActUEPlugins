#include "APIInstance.h"
#include "VActAPI.h"

#include "Misc/SecureHash.h"
#include "Misc/Base64.h"

#include "Sockets.h"
#include "SocketSubsystem.h"
#include "Interfaces/IPv4/IPv4Address.h"
#include "IPAddress.h"

#include "Misc/Char.h"

UAPIInstance::UAPIInstance()
	: bTrackEntryHandles(true)
	, bUseHttps(false)
	, bEnableTick(false)
	, MaxCollisionRetry(10)
	, CodeSymbols(TEXT("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
{

}

void UAPIInstance::CreateCode(FString& Code, const FString& Symbols)
{
	FGuid Guid;
	FPlatformMisc::CreateGuid(Guid);
	uint8* Bytes = reinterpret_cast<uint8*>(&Guid);
	Code.Reserve(sizeof(uint64));
	for (int32 Index = 0; Index < sizeof(uint64); ++Index)
	{
		uint8 _Index = Bytes[Index % sizeof(FGuid)];
		Code += Symbols[_Index % Symbols.Len()];
	}
}

void UAPIInstance::HashUserName(const FString& UserName, FAPIHash& Hash)
{
	TArray<uint8> _UserName;
	FTCHARToUTF8 Converter(*UserName);
	_UserName.Append((uint8*)Converter.Get(), Converter.Length());
	FSHA1 _Hash;
	_Hash.Update(_UserName.GetData(), _UserName.Num());	
	Hash.Data = _Hash.Finalize();
}

void UAPIInstance::HashCode(const FString& Code, int64& Hash)
{
	uint64 _Hash = 0u;
	for (int32 Index = 0; Index < sizeof(uint64); ++Index)
	{
		TCHAR _Char = Code[Index];
		_Hash |= (uint64)(uint8)_Char << (8 * Index);
	}
	Hash = static_cast<int64>(_Hash);
}

void UAPIInstance::HashSecret(FString& Into, const FString& Secret, const FString& Salt, int32 Iterations)
{
	FTCHARToUTF8 _Secret(*Secret);
	FTCHARToUTF8 _Salt(*Salt);

	TArray<uint8> Temp;
	Temp.Append((uint8*)_Secret.Get(), _Secret.Length());
	Temp.Append((uint8*)_Salt.Get(), _Salt.Length());
	
	uint8 Digest[FSHA1::DigestSize];
	for (int32 i = 0; i < Iterations; i++)
	{
		FSHA1 Hash;
		Hash.Update(Temp.GetData(), Temp.Num());
		Hash.Final();
		Hash.GetHash(Digest);

		Temp.SetNum(FSHA1::DigestSize);
		FMemory::Memcpy(Temp.GetData(), Digest, FSHA1::DigestSize);
	}

	Into = FBase64::Encode(Digest, FSHA1::DigestSize);
}

bool UAPIInstance::GetAddress(FString& OutAddress, int64& OutIp, int32& OutPort, bool bWithPort)
{
	bool bCanBindAll;
	
	const bool bValid = Address.IsValid() && Address->IsValid()
		|| (Address = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->GetLocalHostAddr(*GLog, bCanBindAll))->IsValid();
	
	if (bValid)
	{
		uint32 _Ip;
		OutAddress = Address->ToString(bWithPort); 
		Address->GetIp(_Ip);
		OutIp = static_cast<int64>(_Ip);
		OutPort = Address->GetPort();
	}

	return bValid;
}

bool UAPIInstance::NewCode(FString& Code, float LifeTime, int32 UseCount)
{
	int64 Hash;
	CreateCode(Code, CodeSymbols);
	HashCode(Code, Hash);
	bool bCollision = Codes.Contains(Hash);

	int32 CollisionRetry = MaxCollisionRetry;
	while (bCollision && --CollisionRetry > 0)
	{
		CreateCode(Code, CodeSymbols);
		HashCode(Code, Hash);
		bCollision = Codes.Contains(Hash);
	}

	if (!bCollision)
	{
		Codes.Add(Hash, FAPITokenEvent(LifeTime, UseCount));
#if WITH_EDITOR
		UE_LOG(LogTemp, Warning, TEXT(" - Debug Code '%s'/%d"), *Code, Hash);
#endif
	}

	return !bCollision;
}

bool UAPIInstance::NewToken(FGuid& Token, float LifeTime, int32 UseCount)
{
	FPlatformMisc::CreateGuid(Token);
	bool bCollision = Tokens.Contains(Token);

	int32 CollisionRetry = MaxCollisionRetry;
	while (bCollision && --CollisionRetry > 0)
	{
		FPlatformMisc::CreateGuid(Token);
		bCollision = Tokens.Contains(Token);
	}

	if (!bCollision) { Tokens.Add(Token, FAPITokenEvent(LifeTime, UseCount)); }

	return !bCollision;
}

bool UAPIInstance::CreateUser(FGuid& UserId, FGuid PrefPreferedUserIderedId, bool bUsePreferedUserId)
{
	if (!bUsePreferedUserId) { FPlatformMisc::CreateGuid(UserId); }
	else { UserId = PrefPreferedUserIderedId; }
	
	bool bCollision = Users.Contains(UserId);

	int32 CollisionRetry = MaxCollisionRetry;
	while (bCollision && --CollisionRetry > 0)
	{
		FPlatformMisc::CreateGuid(UserId);
		bCollision = Users.Contains(UserId);
	}

	if (!bCollision) { Users.Add(UserId, FAPIUser()); }

	return !bCollision;
}

bool UAPIInstance::FindUserInto(FAPIUser& User, const FGuid& UserId)
{
	FAPIUser* _User = Users.Find(UserId);
	const bool bValid = _User != nullptr;
	if (bValid) { User = (*_User); }
	return bValid;
}

bool UAPIInstance::RemoveUser(const FGuid& UserId)
{
	return Users.Remove(UserId) > 0;
}

bool UAPIInstance::CommitUser(const FGuid& UserId, bool bAndRemove)
{
	FAPIUser* _Account = nullptr;
	FAPIUser* _User = Users.Find(UserId);
	const bool bValid = _User != nullptr && (_Account = Accounts.Find(_User->Hash.Data)) != nullptr;
	if (bValid)
	{
		(*_Account) = (*_User);
		if (bAndRemove) { Users.Remove(UserId); }
	}

	return bValid;
}

bool UAPIInstance::ContainsUser(const FGuid& UserId) const
{
	return Users.Contains(UserId);
}

FAPIUser* UAPIInstance::FindUser(const FGuid& UserId)
{
	return Users.Find(UserId);
}

bool UAPIInstance::Init_Implementation(
)
{
	FHttpServerModule& HttpModule = FHttpServerModule::Get();

	if (bUseHttps)
	{
		const bool bHttps = FVActAPI::Certificate(this);

#if WITH_EDITOR
		UE_LOG(LogTemp, Warning, TEXT("%s Https %s"), *GetNameSafe(this), bHttps ? TEXT("Success") : TEXT("Failur"));
#endif
	}

	FGuid UserId;
	for (auto& User : DefaultUsers)
	{
		CreateUser(UserId, UserId, false);
		FAPIUser* _User = Users.Find(UserId);
		if (_User != nullptr)
		{
			HashSecret(User.Secret.Passphrase, User.Secret.Passphrase, Identity.ScopeUnique);
			HashSecret(User.Secret.Password, User.Secret.Password, Identity.ScopeUnique);
			(*_User) = User;
		}
	}

	for (auto &Route : Routes)
	{
		if (!Route)
		{ 
			continue;
		}

		Route->HttpRouter = HttpModule.GetHttpRouter(Route->Port, true);
		int32 ServerPortMax = Route->Port + Route->PortMaxOffset;
		while (!Route->HttpRouter && ++Route->Port <= ServerPortMax)
		{
			Route->HttpRouter = HttpModule.GetHttpRouter(Route->Port, true);
		}

		if (!Route->HttpRouter)
		{ 
#if WITH_EDITOR
			UE_LOG(LogTemp, Warning, TEXT("%s Failed to Start Route '%s'"), *GetNameSafe(this), *GetNameSafe(Route.Get()));
#endif
			continue;
		}
#if WITH_EDITOR
		UE_LOG(LogTemp, Warning, TEXT("%s Route Started '%s' at '%d'"), *GetNameSafe(this), *GetNameSafe(Route.Get()), Route->Port);
#endif
		
		for (auto& Entry : Route->Entries)
		{
			FName EntryName;
			FHttpRouteHandle EntryHandle;
			const bool bEntryStarted = FVActAPI::Entry(this, Route, Entry, EntryName, EntryHandle);
			if (bEntryStarted)
			{
				if (bTrackEntryHandles) { Route->Handles.Add(EntryHandle); }
#if WITH_EDITOR
				UE_LOG(LogTemp, Warning, TEXT("%s Started Entry '%s'\n\tRoute: '%s'\n\tUrl: '%s'\n\tPath: '%s'"), *GetNameSafe(this), *Entry.Name.ToString(), *GetNameSafe(Route.Get()), *Entry.Url, *EntryHandle->Path);
#endif
			}
		}
	}

#if WITH_EDITOR
	int64 _DEBUG_HashCode;
	FGuid _DEBUG_Token;
	FString _DEBUG_Code = TEXT("T0Iuctvk");
	FGuid::Parse(TEXT("1dd006b5-d184-4b1b-8388-b949b913a055"), _DEBUG_Token);
	HashCode(_DEBUG_Code, _DEBUG_HashCode);
	Codes.Add(_DEBUG_HashCode, FAPITokenEvent(-1));
	Tokens.Add(_DEBUG_Token, FAPITokenEvent(-1));
	UE_LOG(LogTemp, Warning, TEXT("Debug Code '%s'/%d"), *_DEBUG_Code, _DEBUG_HashCode);
	UE_LOG(LogTemp, Warning, TEXT("Debug Token '%s'"), *_DEBUG_Token.ToString());
#endif
	HttpModule.StartAllListeners();
	return true;
}

bool UAPIInstance::DeInit_Implementation(
)
{
	FHttpServerModule& HttpModule = FHttpServerModule::Get();
	HttpModule.StopAllListeners();

	for (auto& Route : Routes)
	{
		const bool bRoute = Route != nullptr && Route->HttpRouter.IsValid();
		if (bRoute)
		{
			for (auto& Handle : Route->Handles)
			{
				const bool bHandle = Handle.IsValid();
				if (bHandle)
				{ 
					Route->HttpRouter->UnbindRoute(Handle);
					Handle.Reset();
				}
			}

			Route->HttpRouter.Reset();
		}	
	}

	return true;
}

bool UAPIInstance::Tick_Implementation(
	float DeltaTime
)
{
	if (bEnableTick)
	{
		for (auto It = Tokens.CreateIterator(); It; ++It)
		{
			auto& Event = It.Value();
			Event.Duration -= DeltaTime;
			const bool bRemove = Event.Duration <= 0.0f || Event.Count == 0;
			if (bRemove) { It.RemoveCurrent(); }
		}

		for (auto It = Codes.CreateIterator(); It; ++It)
		{
			auto& Event = It.Value();
			Event.Duration -= DeltaTime;
			const bool bRemove = Event.Duration <= 0.0f || Event.Count == 0;;
			if (bRemove) { It.RemoveCurrent(); }
		}

		for (auto It = Users.CreateIterator(); It; ++It)
		{
			auto& User = It.Value();
			User.SessionEvent.Duration -= DeltaTime;
			User.AuthenticationTime -= DeltaTime;
			User.bAuthenticated = User.AuthenticationTime > 0.0f;

			const bool bRemove = User.SessionEvent.Duration <= 0.0f;
			if (bRemove) { It.RemoveCurrent(); continue; }
			
			User.CodeTime -= DeltaTime;
			CreateCode(User.Secret.Code, CodeSymbols);
		}
	}

	return true;
}

bool UAPIInstance::Bouncer_Implementation(
	const FString& SessionPathToken
)
{
	FGuid Token;
	const bool bValid = FGuid::Parse(SessionPathToken, Token);
	FAPITokenEvent* Event;
	return bValid 
		&& (Event = Tokens.Find(Token)) != nullptr 
		&& (Event->Count < 0 || --Event->Count == 0) && Event->Duration > 0;
}

bool UAPIInstance::Guard_Implementation(
	const FString& SessionToken,
	FGuid& UserId
)
{
	const bool bValid = FGuid::Parse(SessionToken, UserId);
	FAPIUser* User;
	return bValid
		&& (User = Users.Find(UserId)) != nullptr
		&& User->SessionEvent.Duration > 0;
}

bool UAPIInstance::Reception_Implementation(
	const FString& Code
)
{
	int64 Hash;
	HashCode(Code, Hash);
	FAPITokenEvent* Event = Codes.Find(Hash);
	return Event != nullptr 
		&& (Event->Count < 0 || --Event->Count == 0) && Event->Duration > 0;
}

bool UAPIInstance::Encrypt_Implementation(
	const TArray<uint8>& DataIn,
	UPARAM(ref) TArray<uint8>& DataOut,
	const FGuid& UserId
)
{
	return false;
}

bool UAPIInstance::Decrypt_Implementation(
	const TArray<uint8>& DataIn,
	UPARAM(ref) TArray<uint8>& DataOut,
	const FGuid& UserId
)
{
	return false;
}

bool UAPIInstance::Authenticate_Implementation(
	const FString& UserName,
	const FString& UserPassword,
	FString& Token,
	FGuid& UserId
)
{
	FAPIHash UserHash;
	HashUserName(UserName, UserHash);
	FAPIUser* Account = Accounts.Find(UserHash.Data);
	
	const bool bGranted = Account != nullptr 
		&& !Account->Secret.Password.IsEmpty()
		&& !Account->Token.Id.IsEmpty()
		&& Account->Secret.Password == UserPassword
		&& Account->Token.Id == UserName
		&& Session(Token, Token, false, UserId);

	if (bGranted)
	{ 
		FAPIUser& User = Users.Add(UserId, (*Account)); 
		User.bAuthenticated = true;
		User.Hash = UserHash;
	}

	return bGranted;
}

bool UAPIInstance::Session_Implementation(
	FString& Token,
	const FString& PreferedToken,
	bool bUsePreferedToken,
	FGuid& UserId
)
{
	FAPIUser* User = nullptr;

	const bool bCreated = (!bUsePreferedToken || FGuid::Parse(PreferedToken, UserId)) 
		&& CreateUser(UserId, UserId, bUsePreferedToken) 
		&& (User = Users.Find(UserId)) != nullptr;

	if (bCreated) { Token = UserId.ToString(); }
	
	return bCreated;
}

bool UAPIInstance::Keeper_Implementation(
	FString& Token,
	const FString& PreferedToken,
	bool bUsePreferedToken,
	const FGuid& UserId
)
{
	return false;
}