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

#define _VACTAPI_MPFD_CTX_E(InType) static const TArray<uint8> InType({ '-', '-' })
#define _VACTAPI_MPFD_HDR_E(InType) static const TArray<uint8> InType({ '\r', '\n' })
#define _VACTAPI_MPFD_CTX_D(InType) static const TArray<uint8> InType({ 'C' ,'o' ,'n' ,'t' ,'e' ,'n' ,'t' ,'-' ,'D' ,'i' ,'s' ,'p' ,'o' ,'s' ,'i' ,'t' ,'i' ,'o' ,'n'  })
#define _VACTAPI_MPFD_CTX_T(InType) static const TArray<uint8> InType({ 'C' ,'o' ,'n' ,'t' ,'e' ,'n' ,'t' ,'-' ,'T' ,'y' ,'p' ,'e'  })
//#define _VACTAPI_MPFD_CTX_N(InType) static const TArray<uint8> InType({ 'C' ,'o' ,'n' ,'t' ,'e' ,'n' ,'t' ,'-' ,'L' ,'e' ,'n' ,'g' ,'t' ,'h'  })

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

bool FVActAPI::_Unsafe_Token(const uint8* Buffer, int32 NumBuffer, const uint8* Token, int32 NumToken, int32& Pivot, bool bConsume = true)
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

bool FVActAPI::_Unsafe_TokenNot(const uint8* Buffer, int32 NumBuffer, const uint8* Token, int32 NumToken, int32& Pivot, bool bConsume = true)
{
	if (Pivot + NumToken > NumBuffer)
	{
		if (bConsume) { Pivot = NumBuffer; }
		return true;
	}

	int32 _Pivot = Pivot;
	bool bValid = false;
	for (; _Pivot < NumBuffer; ++_Pivot)
	{
		const bool bMatch = _Unsafe_Token(Buffer, NumBuffer, Token, NumToken, _Pivot, false);
		if (bMatch) { break; }
		bValid = true;
	}
	
	if (bValid && bConsume) { Pivot = _Pivot; }
	return true;
}

bool FVActAPI::_Unsafe_TokenRangeNot(const uint8* Buffer, int32 NumBuffer, uint8 From, uint8 To, int32& Pivot, bool bConsume = true)
{
	int32 _Pivot = Pivot;
	bool bValid = false;
	for (; _Pivot < NumBuffer; ++_Pivot)
	{
		uint8 _Byte = Buffer[_Pivot];
		if (_Byte >= From && _Byte <= To) { break; }
		bValid = true;
	}
	if (bValid && bConsume) { Pivot = _Pivot; }
	return bValid;
}

bool FVActAPI::_Unsafe_Multipart(const FHttpServerRequest& Request, TMap<FString, FAPIConstFormEntry>& Map)
{
	static const FString FieldBoudaryKey = TEXT("boundary=");
	static const FString HeaderContentType = TEXT("Content-Type");
	const TArray<FString>& HeaderValues = Request.Headers[HeaderContentType];
	
	int32 _BondaryPos = HeaderValues[0].Find(FieldBoudaryKey);
	FTCHARToUTF8 Boundary(*HeaderValues[0].Mid(_BondaryPos + FieldBoudaryKey.Len()));
	
	TArray<uint8> BDN_T((uint8*)Boundary.Get(), Boundary.Length());
	_VACTAPI_MPFD_CTX_E(CTX_E);
	_VACTAPI_MPFD_HDR_E(HDR_E);

	const uint8* Buffer = Request.Body.GetData();
	const int32 NumBuffer = Request.Body.Num();
	
	const uint8* _BDN_T = BDN_T.GetData();
	const uint8* _CTX_E = CTX_E.GetData();
	const uint8* _HDR_E = HDR_E.GetData();
	
	const int32 N_BDN_T = BDN_T.Num();
	const int32 N_CTX_E = CTX_E.Num();
	const int32 N_HDR_E = HDR_E.Num();

	TArrayView<const uint8> Segment;
	FAPIConstFormEntry Entry;

	int32 Pivot = 0;
	int32 _Start = 0;
	int32 _End = 0;
	bool bNext = true
		&& _Unsafe_Token(Buffer, NumBuffer, _CTX_E, N_CTX_E, Pivot, true)
		&& _Unsafe_Token(Buffer, NumBuffer, _BDN_T, N_BDN_T, Pivot, true);
	
	while (Pivot < NumBuffer)
	{
		_End = Pivot;
		bNext = _Unsafe_Token(Buffer, NumBuffer, _CTX_E, N_CTX_E, Pivot, true)
			&& _Unsafe_Token(Buffer, NumBuffer, _BDN_T, N_BDN_T, Pivot, true)
			&& _Unsafe_Token(Buffer, NumBuffer, _HDR_E, N_HDR_E, Pivot, true);

		if (bNext)
		{
			if (_End > 0)
			{
				Segment = TArrayView<const uint8>(Buffer + _Start, _End - _Start);
				_Unsafe_Multipart(Segment, Entry);
			}
			_Start = Pivot;
		}
	}

	return false;
}

bool FVActAPI::_Unsafe_Multipart(const TArrayView<const uint8>& Segment, FAPIConstFormEntry& Entry)
{
	_VACTAPI_MPFD_CTX_E(CTX_E);
	_VACTAPI_MPFD_HDR_E(HDR_E);
	_VACTAPI_MPFD_CTX_D(CTX_D);
	_VACTAPI_MPFD_CTX_T(CTX_T);

	const uint8* _HDR_E = HDR_E.GetData();
	const int32 N_HDR_E = HDR_E.Num();

	const uint8* _CTX_D = CTX_D.GetData();
	const uint8* _CTX_T = CTX_T.GetData();

	const int32 N_CTX_D = CTX_D.Num();
	const int32 N_CTX_T = CTX_T.Num();

	const uint8* Buffer = Segment.GetData();
	const int32 NumBuffer = Segment.Num();

	int32 Pivot = 0;
	int32 _Start = 0;
	int32 _End = 0;
	int32 _BodyStart = 0;
	bool bNextHeader = true;
	bool bBody = false;

	while (Pivot < NumBuffer)
	{
		_End = Pivot;
		bNextHeader = _Unsafe_TokenRangeNot(Buffer, NumBuffer, '\n', '\r', Pivot, true)
			&& _Unsafe_Token(Buffer, NumBuffer, _HDR_E, N_HDR_E, Pivot, true);

		if (bNextHeader)
		{
			// handle headers;
		}
		else
		{
			bBody = _Unsafe_Token(Buffer, NumBuffer, _HDR_E, N_HDR_E, Pivot, true);
			if (bBody)
			{
				Entry.Body = TArrayView<const uint8>(Buffer + Pivot, NumBuffer);
				// consume untill !boundary if not aware of NumBuffer
				return true;
			}
		}
	}
}

bool FVActAPI::_Unsafe_Multipart(const FHttpServerRequest& Request, FAPIConstFormEntry& Entry)
{

}

bool FVActAPI::_Unsafe_Multipart(const FHttpServerRequest& Request, const FString& Target, TArray<FAPIConstFormEntry>& Entries)
{

}