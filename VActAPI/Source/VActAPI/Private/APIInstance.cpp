#include "APIInstance.h"
#include "VActAPI.h"

#include "Misc/Char.h"

UAPIInstance::UAPIInstance()
	: bTrackEntryHandles(true)
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

bool UAPIInstance::NewCode(FString& Code, float LifeTime)
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
		Codes.Add(Hash, LifeTime);
#if WITH_EDITOR
		UE_LOG(LogTemp, Warning, TEXT(" - Debug Code '%s'/%d"), *Code, Hash);
#endif
	}

	return !bCollision;
}

bool UAPIInstance::NewToken(FGuid& Token, float LifeTime)
{
	FPlatformMisc::CreateGuid(Token);
	bool bCollision = Tokens.Contains(Token);

	int32 CollisionRetry = MaxCollisionRetry;
	while (bCollision && --CollisionRetry > 0)
	{
		FPlatformMisc::CreateGuid(Token);
		bCollision = Tokens.Contains(Token);
	}

	if (!bCollision) { Tokens.Add(Token, LifeTime); }

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

	FGuid UserId;
	for (auto& User : DefaultUsers)
	{
		CreateUser(UserId, UserId, false);
		FAPIUser* _User = Users.Find(UserId);
		if (_User != nullptr) { (*_User) = User; }
	}

	for (auto &Route : Routes)
	{
		if (!Route)
		{ 
			continue;
		}

		int32 ServerPortMax = Route->Port + Route->PortMaxOffset;
		while (!Route->HttpRouter && Route->Port <= ServerPortMax)
		{
			Route->HttpRouter = HttpModule.GetHttpRouter(Route->Port, true);
			++Route->Port;
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
	FString _DEBUG_Code;
	int64 _DEBUG_HashCode;
	NewCode(_DEBUG_Code, 0.5f);
	HashCode(_DEBUG_Code, _DEBUG_HashCode);
	UE_LOG(LogTemp, Warning, TEXT("Debug Code '%s'/%d"), *_DEBUG_Code, _DEBUG_HashCode);
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
				if (bHandle) { Route->HttpRouter->UnbindRoute(Handle); }
			}

			Route->HttpRouter.Reset();
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
	return bValid && Tokens.Contains(Token);
}

bool UAPIInstance::Guard_Implementation(
	const FString& SessionToken,
	FGuid& UserId
)
{
	const bool bValid = FGuid::Parse(SessionToken, UserId);
	return bValid && Users.Contains(UserId);
}

bool UAPIInstance::Reception_Implementation(
	const FString& Code
)
{
	int64 Hash;
	HashCode(Code, Hash);
	return Codes.Contains(Hash);
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