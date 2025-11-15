#include "VActAPI.h"

#include "APICallback.h"
#include "APIInstance.h"

THIRD_PARTY_INCLUDES_START
#ifdef _WIN32
#define UI UI_ST
#endif

#include <openssl/x509.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>
#include <openssl/bn.h>
#include <openssl/rand.h>
#include <openssl/x509v3.h>

#ifdef _WIN32
#undef UI
#endif
THIRD_PARTY_INCLUDES_END

#include "Misc/Base64.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

#define _VACTAPI_CONSTUINT8(InLiteral)\
	TArray<uint8>(reinterpret_cast<const uint8*>(InLiteral), strlen(InLiteral))

const TArray<uint8> FVActAPI::MPFD_Ctx = { '-','-' };

const TArray<uint8> FVActAPI::MPFD_Hdr = { '\r', '\n', '\r', '\n' };

const TArray<uint8> FVActAPI::MPFD_Ign = { '\0',' ' };

const uint8 FVActAPI::MPFD_Lst = ';';

const uint8 FVActAPI::MPFD_Val = '=';

const uint8 FVActAPI::MPFD_Key = ':';

const FString FVActAPI::AuthorizationKey = TEXT("Authorization");

const FString FVActAPI::SessionPathTokenKey = TEXT("SessionPathToken");

const FString FVActAPI::RequestCodeKey = TEXT("RequestCode");

const TMap<EAPIImageFormat, EImageFormat> FVActAPI::ImageFormatMapE = {
	{  EAPIImageFormat::Unknown, EImageFormat::Invalid },
	{  EAPIImageFormat::Png, EImageFormat::PNG },
	{  EAPIImageFormat::Jpeg, EImageFormat::JPEG },
	{  EAPIImageFormat::GrayscaleJpeg, EImageFormat::GrayscaleJPEG },
	{  EAPIImageFormat::Bmp, EImageFormat::BMP },
	//{  EAPIImageFormat::Ico, EImageFormat::ICO },
	{  EAPIImageFormat::Exr, EImageFormat::EXR },
	//{  EAPIImageFormat::Icns, EImageFormat::ICNS },
	//{  EAPIImageFormat::Tga, EImageFormat::TGA },
	{  EAPIImageFormat::Hdr, EImageFormat::HDR },
	//{  EAPIImageFormat::Tiff, EImageFormat::TIFF },
	//{  EAPIImageFormat::Dds, EImageFormat::DDS },
	{  EAPIImageFormat::UEJpeg, EImageFormat::UEJPEG },
	{  EAPIImageFormat::GrayscaleUEJpeg, EImageFormat::GrayscaleUEJPEG },
};

const TMap<EAPIGammaSpace, EGammaSpace> FVActAPI::GammaSpaceMapE = {
	{ EAPIGammaSpace::Linear, EGammaSpace::Linear },
	{ EAPIGammaSpace::Pow22, EGammaSpace::Pow22 },
	{ EAPIGammaSpace::sRGB, EGammaSpace::sRGB },
	{ EAPIGammaSpace::Unknown, EGammaSpace::Invalid }
};

const TMap<EAPIImageRawFormat, EPixelFormat> FVActAPI::ImageRawFormatMapE = {
	{ EAPIImageRawFormat::G8, EPixelFormat::PF_G8 },
	{ EAPIImageRawFormat::BGRA8, EPixelFormat::PF_B8G8R8A8 },
	{ EAPIImageRawFormat::BGRE8, EPixelFormat::PF_B8G8R8A8 },
	{ EAPIImageRawFormat::RGBA16, EPixelFormat::PF_R16G16B16A16_UNORM },
	{ EAPIImageRawFormat::RGBA16F, EPixelFormat::PF_FloatRGBA },
	{ EAPIImageRawFormat::RGBA32F, EPixelFormat::PF_A32B32G32R32F },
	{ EAPIImageRawFormat::G16, EPixelFormat::PF_G16 },
	{ EAPIImageRawFormat::R16F, EPixelFormat::PF_R16F },
	{ EAPIImageRawFormat::R32F, EPixelFormat::PF_R32_FLOAT },
	{ EAPIImageRawFormat::Unknown, EPixelFormat::PF_Unknown }
};

const TMap<ERawImageFormat::Type, EPixelFormat> FVActAPI::_ImageRawFormatMapE = {
	{ ERawImageFormat::Type::G8, EPixelFormat::PF_G8 },
	{ ERawImageFormat::Type::BGRA8, EPixelFormat::PF_B8G8R8A8 },
	{ ERawImageFormat::Type::BGRE8, EPixelFormat::PF_B8G8R8A8 },
	{ ERawImageFormat::Type::RGBA16, EPixelFormat::PF_R16G16B16A16_UNORM },
	{ ERawImageFormat::Type::RGBA16F, EPixelFormat::PF_FloatRGBA },
	{ ERawImageFormat::Type::RGBA32F, EPixelFormat::PF_A32B32G32R32F },
	{ ERawImageFormat::Type::G16, EPixelFormat::PF_G16 },
	{ ERawImageFormat::Type::R16F, EPixelFormat::PF_R16F },
	{ ERawImageFormat::Type::R32F, EPixelFormat::PF_R32_FLOAT },
	{ ERawImageFormat::Type::Invalid, EPixelFormat::PF_Unknown }
};

const TMap<EPixelFormat, ERawImageFormat::Type> FVActAPI::_ImageRawFormatMapInvE = {
	{ EPixelFormat::PF_G8, ERawImageFormat::Type::G8 },
	{ EPixelFormat::PF_B8G8R8A8, ERawImageFormat::Type::BGRA8 },
	{ EPixelFormat::PF_B8G8R8A8, ERawImageFormat::Type::BGRE8 },
	{ EPixelFormat::PF_R16G16B16A16_UNORM, ERawImageFormat::Type::RGBA16 },
	{ EPixelFormat::PF_FloatRGBA, ERawImageFormat::Type::RGBA16F },
	{ EPixelFormat::PF_A32B32G32R32F, ERawImageFormat::Type::RGBA32F },
	{ EPixelFormat::PF_G16, ERawImageFormat::Type::G16 },
	{ EPixelFormat::PF_R16F, ERawImageFormat::Type::R16F },
	{ EPixelFormat::PF_R32_FLOAT, ERawImageFormat::Type::R32F },
	{ EPixelFormat::PF_Unknown, ERawImageFormat::Type::Invalid }
};

const TMap<EAPIEntryMode, FString> FVActAPI::EntryModeMap = {
	{ EAPIEntryMode::Unknown, TEXT("UNKNOWN")},
	{ EAPIEntryMode::Post, TEXT("POST")},
	{ EAPIEntryMode::Put, TEXT("PUT")},
	{ EAPIEntryMode::Get, TEXT("GET")},
	{ EAPIEntryMode::Patch, TEXT("PATCH")},
	{ EAPIEntryMode::Delete, TEXT("DELETE")},
	{ EAPIEntryMode::Head, TEXT("HEAD")},
	{ EAPIEntryMode::Options, TEXT("OPTIONS")},
	{ EAPIEntryMode::Any, TEXT("ANY")}
};

const TMap<FString,EAPIEntryMode> FVActAPI::EntryModeMapInv = {
	{ TEXT("UNKNOWN"), EAPIEntryMode::Unknown},
	{ TEXT("POST"), EAPIEntryMode::Post},
	{ TEXT("PUT"), EAPIEntryMode::Put},
	{ TEXT("GET"), EAPIEntryMode::Get},
	{ TEXT("PATCH"), EAPIEntryMode::Patch},
	{ TEXT("DELETE"), EAPIEntryMode::Delete},
	{ TEXT("HEAD"), EAPIEntryMode::Head},
	{ TEXT("OPTIONS"), EAPIEntryMode::Options},
	{ TEXT("ANY"), EAPIEntryMode::Any}
};

const TMap<EAPIEntryMode, EHttpServerRequestVerbs> FVActAPI::EntryModeMapE = {
	{ EAPIEntryMode::Unknown, EHttpServerRequestVerbs::VERB_NONE},
	{ EAPIEntryMode::Post, EHttpServerRequestVerbs::VERB_POST},
	{ EAPIEntryMode::Put, EHttpServerRequestVerbs::VERB_PUT},
	{ EAPIEntryMode::Get, EHttpServerRequestVerbs::VERB_GET},
	{ EAPIEntryMode::Patch, EHttpServerRequestVerbs::VERB_PATCH},
	{ EAPIEntryMode::Delete, EHttpServerRequestVerbs::VERB_DELETE},
	{ EAPIEntryMode::Head, EHttpServerRequestVerbs::VERB_NONE},
	{ EAPIEntryMode::Options, EHttpServerRequestVerbs::VERB_OPTIONS},
	{ EAPIEntryMode::Any, EHttpServerRequestVerbs::VERB_NONE}
};

const TMap<EAPIEntryContent, FString> FVActAPI::EntryContentMap = {
	{ EAPIEntryContent::Unknown, TEXT("unknown")},
	{ EAPIEntryContent::Image, TEXT("multipart/form-data")},
	{ EAPIEntryContent::ImagePng, TEXT("image/png")},
	{ EAPIEntryContent::ImageJpg, TEXT("image/jpeg")},
	{ EAPIEntryContent::ImageBmp, TEXT("image/bmp")},
	{ EAPIEntryContent::ImageExr, TEXT("image/exr")},
	{ EAPIEntryContent::ImageHdr, TEXT("image/hdr")},
	{ EAPIEntryContent::Audio, TEXT("multipart/form-data")},
	{ EAPIEntryContent::AudioOgg, TEXT("audio/ogg")},
	{ EAPIEntryContent::AudioWav, TEXT("audio/wav")},
	{ EAPIEntryContent::AudioAcc, TEXT("audio/acc")},
	{ EAPIEntryContent::Video, TEXT("multipart/form-data")},
	{ EAPIEntryContent::VideoMp4, TEXT("video/mp4")},
	{ EAPIEntryContent::VideoMov, TEXT("video/mov")},
	{ EAPIEntryContent::VideoAvi, TEXT("video/avi")},
	{ EAPIEntryContent::Model, TEXT("multipart/form-data")},
	{ EAPIEntryContent::ModelGlb, TEXT("model/glb")},
	{ EAPIEntryContent::ModelAbc, TEXT("model/abc")},
	{ EAPIEntryContent::ModelFbx, TEXT("model/fbx")},
	{ EAPIEntryContent::Text, TEXT("text/plain")},
	{ EAPIEntryContent::JavaScript, TEXT("text/javascript")},
	{ EAPIEntryContent::Html, TEXT("text/html")},
	{ EAPIEntryContent::Json, TEXT("application/json")},
	{ EAPIEntryContent::Xml, TEXT("application/xml")},
	{ EAPIEntryContent::Ini, TEXT("application/ini")},
	{ EAPIEntryContent::Binary, TEXT("application/octet-stream")},
	{ EAPIEntryContent::Form, TEXT("multipart/form-data")},
	{ EAPIEntryContent::FormUrl, TEXT("/application/x-www-form-urlencoded")}
};

const TMap<FString, EAPIEntryContent> FVActAPI::EntryContentMapInv = {
	{ TEXT("unknown"), EAPIEntryContent::Unknown},
	//{ TEXT("multipart/form-data"), EAPIEntryContent::Image},
	{ TEXT("image/png"), EAPIEntryContent::ImagePng},
	{ TEXT("image/jpeg"), EAPIEntryContent::ImageJpg},
	{ TEXT("image/bmp"), EAPIEntryContent::ImageBmp},
	{ TEXT("image/exr"), EAPIEntryContent::ImageExr},
	{ TEXT("image/hdr"), EAPIEntryContent::ImageHdr},
	//{ TEXT("multipart/form-data"), EAPIEntryContent::Audio},
	{ TEXT("audio/ogg"), EAPIEntryContent::AudioOgg},
	{ TEXT("audio/wav"), EAPIEntryContent::AudioWav},
	{ TEXT("audio/acc"), EAPIEntryContent::AudioAcc},
	//{ TEXT("multipart/form-data"), EAPIEntryContent::Video},
	{ TEXT("video/mp4"), EAPIEntryContent::VideoMp4},
	{ TEXT("video/mov"), EAPIEntryContent::VideoMov},
	{ TEXT("video/avi"), EAPIEntryContent::VideoAvi},
	//{ TEXT("multipart/form-data"), EAPIEntryContent::Model},
	{ TEXT("model/glb"), EAPIEntryContent::ModelGlb},
	{ TEXT("model/abc"), EAPIEntryContent::ModelAbc},
	{ TEXT("model/fbx"), EAPIEntryContent::ModelFbx},
	{ TEXT("text/plain"), EAPIEntryContent::Text},
	{ TEXT("text/javascript"), EAPIEntryContent::JavaScript},
	{ TEXT("text/html"), EAPIEntryContent::Html},
	{ TEXT("application/json"), EAPIEntryContent::Json},
	{ TEXT("application/xml"), EAPIEntryContent::Xml},
	{ TEXT("application/ini"), EAPIEntryContent::Ini},
	{ TEXT("application/octet-stream"), EAPIEntryContent::Binary},
	{ TEXT("multipart/form-data"), EAPIEntryContent::Form},
	{ TEXT("multipart/x-www-form-urlencoded"), EAPIEntryContent::FormUrl }
};

const TMap<EAPIImageFormat, EAPIEntryContent> FVActAPI::EntryImageContentMapE = {
	{  EAPIImageFormat::Unknown, EAPIEntryContent::Unknown },
	{  EAPIImageFormat::Png, EAPIEntryContent::ImagePng },
	{  EAPIImageFormat::Jpeg, EAPIEntryContent::ImageJpg },
	{  EAPIImageFormat::GrayscaleJpeg, EAPIEntryContent::ImageJpg },
	{  EAPIImageFormat::Bmp, EAPIEntryContent::ImageBmp },
	//{  EAPIImageFormat::Ico, EAPIEntryContent::ImageIco },
	{  EAPIImageFormat::Exr, EAPIEntryContent::ImageExr },
	//{  EAPIImageFormat::Icns, EAPIEntryContent::ImageIco },
	//{  EAPIImageFormat::Tga, EAPIEntryContent::ImageTga },
	{  EAPIImageFormat::Hdr, EAPIEntryContent::ImageHdr },
	//{  EAPIImageFormat::Tiff, EAPIEntryContent::ImageTiff },
	//{  EAPIImageFormat::Dds, EAPIEntryContent::ImageDds },
	{  EAPIImageFormat::UEJpeg, EAPIEntryContent::ImageJpg },
	{  EAPIImageFormat::GrayscaleUEJpeg, EAPIEntryContent::ImageJpg },
};

bool FVActAPI::_Unsafe_Entry(UAPIInstance* InAPIInstance, UAPIRoute* InRoute, FAPIEntry& InEntry, FName& OutName, FHttpRouteHandle& Handle)
{
	TWeakObjectPtr<UAPIInstance> WeakInstance = InAPIInstance;
	Handle = InRoute->HttpRouter->BindRoute(FHttpPath(InEntry.GetEntryUrl(InAPIInstance->Identity)), EntryModeMapE[InEntry.Mode], FHttpRequestHandler::CreateLambda(
		[&InEntry, InRoute, WeakInstance](const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
		{
			if (!WeakInstance.IsValid())
			{
#if WITH_EDITOR
				UE_LOG(LogTemp, Warning, TEXT("invalid API Instance object, probably destroyed"));
#endif
				return false;
			}

			const FAPIIdentity& Identity = WeakInstance->Identity;
			TUniquePtr<FHttpServerResponse> Response;
			FGuid UserId;

			if (InEntry.bSessionPath)
			{
				FString SessionPathToken = Request.PathParams.Contains(FVActAPI::SessionPathTokenKey) 
					? Request.PathParams[FVActAPI::SessionPathTokenKey]
					: TEXT("");

				const bool bBounce = !WeakInstance->Bouncer(SessionPathToken);
				if (bBounce)
				{
					Response = FHttpServerResponse::Create(TEXT(""), EntryContentMap[EAPIEntryContent::Text]);
					Response->Code = EHttpServerResponseCodes::Denied;
					OnComplete(MoveTemp(Response));
					return true;
				}
			}

			if (InEntry.bSession)
			{
				const TArray<FString>* AuthHeader = Request.Headers.Find(FVActAPI::AuthorizationKey);
				FString SessionToken = AuthHeader ? (*AuthHeader)[0].RightChop(7) : TEXT("");

				const bool bReject = !WeakInstance->Guard(SessionToken, UserId);
				if (bReject)
				{
					Response = FHttpServerResponse::Create(TEXT(""), EntryContentMap[EAPIEntryContent::Text]);
					Response->Code = EHttpServerResponseCodes::Denied;
					OnComplete(MoveTemp(Response));
					return true;
				}
			}

			if (InEntry.bRequestCode)
			{
				FString RequestCode = Request.PathParams.Contains(FVActAPI::RequestCodeKey) 
					? Request.PathParams[FVActAPI::RequestCodeKey]
					: TEXT("");
#if WITH_EDITOR
				UE_LOG(LogTemp, Warning, TEXT("request with Code '%s', %d url params"), *RequestCode, Request.PathParams.Num());
#endif
				const bool bDecline = !WeakInstance->Reception(RequestCode);
				if (bDecline)
				{
					Response = FHttpServerResponse::Create(TEXT(""), EntryContentMap[EAPIEntryContent::Text]);
					Response->Code = EHttpServerResponseCodes::Denied;
					OnComplete(MoveTemp(Response));
					return true;
				}
			}

			if (InEntry.bAuthenticate)
			{
				const TArray<FString>* AuthHeader = Request.Headers.Find(FVActAPI::AuthorizationKey);
				FString UserNamePasswordEncoded = AuthHeader ? (*AuthHeader)[0].RightChop(6) : TEXT("");
				FString UserNamePassword;
				FString UserName, HashedPassword;
				FString SessionToken;

				const bool bDenied = !UserNamePasswordEncoded.IsEmpty()
					|| !FBase64::Decode(UserNamePasswordEncoded, UserNamePassword, EBase64Mode::Standard)
					|| !UserNamePassword.Split(TEXT(":"), &UserName, &HashedPassword)
					|| !WeakInstance->Authenticate(UserName, HashedPassword, SessionToken, UserId);

				if (bDenied)
				{
					Response = FHttpServerResponse::Create(TEXT(""), EntryContentMap[EAPIEntryContent::Text]);
					Response->Code = EHttpServerResponseCodes::Denied;
					OnComplete(MoveTemp(Response));
					return true;
				}
			}

			if (InEntry.bReceive && InEntry.Callback)
			{
				const bool bFailed = !InEntry.Callback->OnDataIn(Request, InEntry, InRoute, WeakInstance.Get(), UserId);
#if WITH_EDITOR
				UE_LOG(LogTemp, Warning, TEXT("request handling receive  %s '%s'"), bFailed ? TEXT("Failed") : TEXT("Success"), *InEntry.GetEntryUrl(Identity));
#endif
				if (bFailed)
				{
					Response = FHttpServerResponse::Create(TEXT(""), EntryContentMap[EAPIEntryContent::Text]);
					Response->Code = EHttpServerResponseCodes::ServerError;
					OnComplete(MoveTemp(Response));
					return true;
				}
			}

			if (InEntry.bRespond && InEntry.Callback)
			{
				TArray<uint8> Data;
				Response = FHttpServerResponse::Create(TEXT(""), EntryContentMap[InEntry.Content]);

				const bool bFailed = !InEntry.Callback->OnDataOut(Response, Request, InEntry, InRoute, WeakInstance.Get(), UserId)
					|| !(!InEntry.bEncrypt || WeakInstance->Encrypt(Data, Data, UserId));
#if WITH_EDITOR
				UE_LOG(LogTemp, Warning, TEXT("request handling response %s '%s'"), bFailed ? TEXT("Failed") : TEXT("Success"), *InEntry.GetEntryUrl(Identity));
#endif
				if (bFailed)
				{
					Response = FHttpServerResponse::Create(TEXT(""), EntryContentMap[EAPIEntryContent::Text]);
					Response->Code = EHttpServerResponseCodes::ServerError;
					OnComplete(MoveTemp(Response));
					return true;
				}

				Response->Code = EHttpServerResponseCodes::Ok;
			}

			if (!Response.IsValid())
			{
				Response = FHttpServerResponse::Create(TEXT(""), EntryContentMap[InEntry.Content]);
				Response->Code = EHttpServerResponseCodes::Ok;
			}

			OnComplete(MoveTemp(Response));
			return true;
		})
	);

	return Handle.IsValid();
}


bool FVActAPI::_Unsafe_Certificate(UAPIInstance* InAPIInstance)
{
	// https://blog.devgenius.io/how-to-generate-self-signed-ssl-certificates-b85562830ab

	FString Address;
	int64 Ip;
	int32 Port;

	bool bSuccess = InAPIInstance->GetAddress(Address, Ip, Port, false);
	if (bSuccess)
	{
		FAPICertificate& Certification = InAPIInstance->Certification;
		RSA* Key = RSA_new();

		BIGNUM* Exp = BN_new();
		BN_set_word(Exp, RSA_F4);
		RSA_generate_key_ex(Key, 2048, Exp, nullptr);

		EVP_PKEY* PrivateKey = EVP_PKEY_new();
		EVP_PKEY_assign_RSA(PrivateKey, Key);

		X509* Certificate = X509_new();
		X509_set_version(Certificate, 2);
		ASN1_INTEGER_set(X509_get_serialNumber(Certificate), static_cast<long>(FPlatformTime::Cycles()));

		X509_NAME* Name = X509_NAME_new();
		X509_NAME_add_entry_by_txt(Name, "CN", MBSTRING_ASC, (unsigned char*)TCHAR_TO_UTF8(*Address), -1, -1, 0);
		X509_set_subject_name(Certificate, Name);
		X509_set_issuer_name(Certificate, Name);

		X509_gmtime_adj(X509_get_notBefore(Certificate), Certification.DurationFrom);
		X509_gmtime_adj(X509_get_notAfter(Certificate), Certification.DurationTo);

		X509_set_pubkey(Certificate, PrivateKey);

		X509_EXTENSION* Extension = X509V3_EXT_conf_nid(nullptr, nullptr, NID_subject_alt_name, 
			TCHAR_TO_UTF8(*FString::Printf(TEXT("DNS:localhost,IP:127.0.0.1,IP:%s"), *Address))
		);
		X509_add_ext(Certificate, Extension, -1);

		X509_sign(Certificate, PrivateKey, EVP_sha256());

		Certification.Certificate = FPaths::ProjectSavedDir() / TEXT("server.crt");
		Certification.PrivateKey = FPaths::ProjectSavedDir() / TEXT("server.key");

		BIO* BinaryOut = BIO_new(BIO_s_mem());
		TArray<uint8> Buffer;
		
		bSuccess &= BinaryOut != nullptr;
		if (bSuccess)
		{
			PEM_write_bio_X509(BinaryOut, Certificate);
			const int32 CertificateSize = BIO_pending(BinaryOut);
			Buffer.SetNumUninitialized(CertificateSize, EAllowShrinking::No);
			BIO_read(BinaryOut, Buffer.GetData(), CertificateSize);

			bSuccess &= FFileHelper::SaveArrayToFile(Buffer, *Certification.Certificate);
		}

		BIO_reset(BinaryOut);

		if (bSuccess)
		{
			PEM_write_bio_PrivateKey(BinaryOut, PrivateKey, nullptr, nullptr, 0, nullptr, nullptr);
			const int32 PrivateKeySize = BIO_pending(BinaryOut);
			Buffer.SetNumUninitialized(PrivateKeySize, EAllowShrinking::No);
			BIO_read(BinaryOut, Buffer.GetData(), PrivateKeySize);

			bSuccess &= FFileHelper::SaveArrayToFile(Buffer, *Certification.PrivateKey);
		}

		BIO_free(BinaryOut);
		X509_EXTENSION_free(Extension);
		BN_free(Exp);
		X509_free(Certificate);
		EVP_PKEY_free(PrivateKey);

#if WITH_EDITOR
		UE_LOG(LogTemp, Warning, TEXT("Generated Https for '%s' at\n\t - Certificate '%s'\n\t - Key '%s'"), *Address, *Certification.Certificate, *Certification.PrivateKey);
#endif
	}

	return bSuccess;
}

bool FVActAPI::_Unsafe_Token(const uint8* Buffer, int32 NumBuffer, const uint8* Token, int32 NumToken, int32& Pivot, bool bConsume)
{
	if (Pivot + NumToken > NumBuffer) { return false; }

	int32 _Pivot = Pivot;
	for (int32 Index = 0; Index < NumToken && _Pivot < NumBuffer; ++Index, ++_Pivot)
	{
		if (Buffer[_Pivot] != Token[Index]) { return false; }
	}
	if (bConsume) { Pivot = _Pivot; }
	return true;
}

bool FVActAPI::_Unsafe_TokenFirst(const uint8* Buffer, int32 NumBuffer, const uint8* Token, int32 NumToken, int32& Pivot, bool bConsume)
{
	if (Pivot + NumToken > NumBuffer) { return false; }

	int32 _Pivot = Pivot;
	bool bValid = false;
	for (int32 Index = 0; Index < NumToken && _Pivot < NumBuffer; ++Index, ++_Pivot)
	{
		bValid = Buffer[_Pivot] == Token[Index];
		if (!bValid) { _Pivot -= Index; Index = -1; bValid = false; }
	}
	if (bConsume) { Pivot = _Pivot; }
	return bValid;
}

bool FVActAPI::_Unsafe_TokenFirst(const uint8* Buffer, int32 NumBuffer, const uint8 Token, int32& Pivot, bool bConsume)
{
	int32 _Pivot = Pivot;
	bool bValid = false;
	for (; _Pivot < NumBuffer; ++_Pivot)
	{
		if (Buffer[_Pivot] == Token) { bValid = true; break; }
	}
	if (bValid && bConsume) { Pivot = _Pivot; }
	return bValid;
}

bool FVActAPI::_Unsafe_TokenSkip(const uint8* Buffer, int32 NumBuffer, const uint8 Range[2], int32& Pivot)
{
	for ( ; Pivot < NumBuffer; ++Pivot)
	{
		const uint8 Byte = Buffer[Pivot];
		if (Byte < Range[0] || Byte > Range[1]) { return true; }
	}
	return true;
}

bool FVActAPI::_Unsafe_TokenSkipNot(const uint8* Buffer, int32 NumBuffer, const uint8 Range[2], int32& Pivot)
{
	for (; Pivot < NumBuffer; ++Pivot)
	{
		const uint8 Byte = Buffer[Pivot];
		if (Byte >= Range[0] && Byte <= Range[1]) { return true; }
	}
	return true;
}

bool FVActAPI::_Unsafe_Headers(TArrayView<const uint8>& Headers, TMap<FString, TArray<FString>>& OutHeaders)
{
	const uint8* Data = Headers.GetData();
	const int32 Num = Headers.Num();

	bool bHeader = Headers.Num() > 0;
	int32 Start = 0, End = 0, Pivot = 0;
	while (bHeader && Pivot < Num)
	{
		bHeader = _Unsafe_TokenFirst(Data, Num, MPFD_Key, Pivot, true);
		if (bHeader)
		{
			FUTF8ToTCHAR _Key(reinterpret_cast<const ANSICHAR*>(Data + Start), Pivot - Start);
			FString Key(_Key.Length(), _Key.Get());
			_Unsafe_TokenSkip(Data, End, MPFD_Ign.GetData(), Pivot);
			int32 _Start = Pivot;
			_Unsafe_TokenSkipNot(Data, End, MPFD_Ign.GetData(), Pivot);
			FUTF8ToTCHAR _Value(reinterpret_cast<const ANSICHAR*>(Data + _Start), Pivot - _Start);
			FString Value(_Value.Length(), _Value.Get());

			TArray<FString>* Values = OutHeaders.Find(Key);
			if (!Values) { OutHeaders.Add(Key, { Value }); }
			else { Values->Add(Value); }
			_Unsafe_TokenSkip(Data, End, MPFD_Ign.GetData(), Pivot);
			Start = Pivot;
		}
	}
	return true;
}

bool FVActAPI::_Unsafe_Multipart(const FHttpServerRequest& Request, TArray<FAPIConstMultipartSegment>& Segments)
{
	static const FString FieldBoudaryKey = TEXT("boundary=");
	static const FString HeaderContentType = TEXT("Content-Type");
	const TArray<FString>& HeaderValues = Request.Headers[HeaderContentType];
	
	int32 _BondaryPos = HeaderValues[0].Find(FieldBoudaryKey);
	FString Bondary = HeaderValues[0].Mid(_BondaryPos + FieldBoudaryKey.Len());
	FTCHARToUTF8 _Boundary(*Bondary);
	TArray<uint8> MPFD_Bnd((uint8*)_Boundary.Get(), _Boundary.Length());
	MPFD_Bnd.Insert(MPFD_Ctx, 0);
#if WITH_EDITOR
	FUTF8ToTCHAR _DEBUG_CxBnd(reinterpret_cast<const ANSICHAR*>(MPFD_Bnd.GetData()), MPFD_Bnd.Num());
	FString _DEBUG_Headers(_DEBUG_CxBnd.Length(), _DEBUG_CxBnd.Get());
	UE_LOG(LogTemp, Warning, TEXT("bondary '%s'"), *_DEBUG_Headers);
#endif

	const uint8* Data = Request.Body.GetData();
	const int32 Num = Request.Body.Num();

	int32 Start = 0, End = 0, Pivot = 0;	
	bool bSegment = _Unsafe_Token(Data, Num, MPFD_Bnd.GetData(), MPFD_Bnd.Num(), Pivot, true);
	Start = Pivot;
	bool bValid = false;
	while (bSegment && Pivot < Num)
	{
		bSegment = _Unsafe_TokenFirst(Data, Num, MPFD_Bnd.GetData(), MPFD_Bnd.Num(), Pivot, true);

		if (bSegment)
		{
			bValid = true;
			End = Pivot - MPFD_Bnd.Num();
#if WITH_EDITOR
			UE_LOG(LogTemp, Warning, TEXT("\tsegment end %d:%d %s"), Start, End, bSegment ? TEXT("True") : TEXT("False"));
#endif	
			FAPIConstMultipartSegment Segment;
			int32 _Pivot = Start;
			_Unsafe_TokenSkip(Data, End, MPFD_Ign.GetData(), _Pivot);
#if WITH_EDITOR
			UE_LOG(LogTemp, Warning, TEXT("\tskipped %d(%d)"), _Pivot, _Pivot - Start);
#endif
			int32 _Start = _Pivot;
			bool bHeader = _Unsafe_TokenFirst(Data, End, MPFD_Hdr.GetData(), MPFD_Hdr.Num(), _Pivot, true);
			int32 _End = _Pivot;
			if (bHeader)
			{ 
				Segment.Headers = TArrayView<const uint8>(Data + _Start, _End - _Start);
			}
			Segment.Body = TArrayView<const uint8>(Data + _End, End - _End);
#if WITH_EDITOR
			FUTF8ToTCHAR _DEBUG_CxHdr(reinterpret_cast<const ANSICHAR*>(Segment.Headers.GetData()), Segment.Headers.Num());
			FString _DEBUG_Headers(_DEBUG_CxHdr.Length(), _DEBUG_CxHdr.Get());
			UE_LOG(LogTemp, Warning, TEXT("\theaders %d:%d\n%s"), _Start, _End, *_DEBUG_Headers);
			//FString _DEBUG_Body = BytesToHex(Segment.Body.GetData(), Segment.Body.Num());
			//UE_LOG(LogTemp, Warning, TEXT("\tbody %d:%d\n%s"), _End, End, *_DEBUG_Body);
#endif
			Segments.Add(Segment);
		}
	}

	return bValid;
}