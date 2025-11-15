#include "VActAPITypes.h"

#include "APICallback.h"


FAPIIdentity::FAPIIdentity()
	: Name(NAME_None)
	, ServerUnique(FGuid::NewGuid())
	, ScopeUnique(TEXT("59ae4b3f-79f5-497b-acc0-1f905c5f96d3"))
	, Version(1)
{

}

FAPIIdentity::FAPIIdentity(FString InServerUnique, FString InScopeUnique, int32 InVersion)
	: Name(NAME_None)
	, ServerUnique(InServerUnique)
	, ScopeUnique(InScopeUnique)
	, Version(InVersion)
{

}

FAPICertificate::FAPICertificate()
	: DurationFrom(0l)
	, DurationTo(60l * 60l * 24l)
	, Certificate(TEXT(""))
	, PrivateKey(TEXT(""))
{

}

FAPICertificate::FAPICertificate(FString InCertificate, FString InPrivateKey)
	: Certificate(InCertificate)
	, PrivateKey(InPrivateKey)
{

}

FAPIToken::FAPIToken()
	: Id(TEXT(""))
	, PublicKey(TEXT(""))
	, ApiKey(TEXT(""))
{

}

FAPISecret::FAPISecret()
	: PrivateKey(TEXT(""))
	, Passphrase(TEXT(""))
	, Password(TEXT(""))
	, Code(TEXT(""))
{

}

FAPIEntry::FAPIEntry()
	: bSession(false)
	, bSessionPath(false)
	, bScoped(false)
	, bRespond(true)
	, bReceive(true)
	, bEncrypt(false)
	, bAuthenticate(false)
	, bRequestCode(false)
	, Name(NAME_None)
	, Url(TEXT(""))
	, Mode(EAPIEntryMode::Get)
	, Content(EAPIEntryContent::Text)
{

}

FAPIEntry::FAPIEntry(FString InUrl, EAPIEntryMode InMode, EAPIEntryContent InContent)
	: bSession(false)
	, bSessionPath(false)
	, bScoped(false)
	, bRespond(true)
	, bReceive(true)
	, bEncrypt(false)
	, bRequestCode(false)
	, Name(NAME_None)
	, Url(InUrl)
	, Mode(InMode)
	, Content(InContent)
{

}

FString FAPIEntry::GetEntryUrl(const FAPIIdentity& InIdentity) const
{
	static const FString EmptyToken;
	static const FString EmptyCode;
	return GetEntryUrl(InIdentity, EmptyToken, EmptyCode, false, false);
}

FString FAPIEntry::GetEntryUrlCode(const FAPIIdentity& InIdentity, const FString& Code, bool bUseCode) const
{
	static const FString EmptyToken;
	return GetEntryUrl(InIdentity, EmptyToken, Code, false, bUseCode);
}

FString FAPIEntry::GetEntryUrlPath(const FAPIIdentity& InIdentity, const FString& PathToken, bool bUsePathToken) const
{
	static const FString EmptyCode;
	return GetEntryUrl(InIdentity, PathToken, EmptyCode, bUsePathToken, false);
}

FString FAPIEntry::GetEntryUrl(const FAPIIdentity& InIdentity, const FString& PathToken, const FString& Code) const
{
	return GetEntryUrl(InIdentity, PathToken, Code, true, true);
}

FString FAPIEntry::GetEntryUrl(const FAPIIdentity& InIdentity, const FString& PathToken, const FString& Code, bool bUsePathToken, bool bUseCode) const
{
	FString EntryUrl = Url;

	if (bRequestCode)
	{ 
		EntryUrl = bUseCode 
			? FString::Printf(TEXT("%s/%s"), *EntryUrl, *Code)
			: FString::Printf(TEXT("%s/:RequestCode"), *EntryUrl);
	}
	if (bSessionPath)
	{
		EntryUrl = bUsePathToken
			? FString::Printf(TEXT("/%s%s"), *PathToken, *EntryUrl)
			: FString::Printf(TEXT("/:SessionPathToken%s"), *EntryUrl);
	}
	if (bScoped) { EntryUrl = FString::Printf(TEXT("/%s%s"), *InIdentity.ScopeUnique, *EntryUrl); }
	if (InIdentity.Version > 0) { EntryUrl = FString::Printf(TEXT("/v%d%s"), InIdentity.Version, *EntryUrl); }
#if WITH_EDITOR
	UE_LOG(LogTemp, Warning, TEXT("Url '%s'"), *EntryUrl);
#endif
	return EntryUrl;
}

void FAPIEntry::SecurityIntersectInto(FAPIEntry& Into) const
{
	Into.bEncrypt |= bEncrypt;
	Into.bScoped |= bScoped;
	Into.bSession |= bSession;
	Into.bSessionPath |= bSessionPath;
	Into.bRequestCode |= bRequestCode;
	Into.bAuthenticate |= bAuthenticate;
}

FAPITokenCondition::FAPITokenCondition()
	: bUser(false)
	, bContext(false)
	, bScope(false)
	, bIp(false)
	, UserId()
	, ContextId(0)
	, ScopeId(0)
	, Ip(0)
{

}

FAPITokenEvent::FAPITokenEvent()
	: Duration(11.0f)
	, Count(1)
{

}

FAPITokenEvent::FAPITokenEvent(float InDuration, int32 InCount)
	: Duration(InDuration)
	, Count(InCount)
{

}

FAPITokenEvent::FAPITokenEvent(int32 InCount)
	: Duration(FLT_MAX)
	, Count(InCount)
{

}

FAPIUser::FAPIUser()
	: bAuthenticated(false)
	, SessionEvent(0.0f)
	, AuthenticationTime(0.0f)
	, CodeTime(0.0f)
	, Token()
	, Secret()
{

}

FAPIUser::FAPIUser(FAPIToken InToken)
	: bAuthenticated(false)
	, SessionEvent(0.0f)
	, AuthenticationTime(0.0f)
	, CodeTime(0.0f)
	, Token(InToken)
	, Secret()
{

}

FAPIUser::FAPIUser(FAPIToken InToken, FAPISecret InSecret)
	: bAuthenticated(false)
	, SessionEvent(0.0f)
	, AuthenticationTime(0.0f)
	, CodeTime(0.0f)
	, Token(InToken)
	, Secret(InSecret)
{

}