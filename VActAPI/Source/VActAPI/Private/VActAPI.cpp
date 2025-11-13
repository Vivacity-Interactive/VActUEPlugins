#include "VActAPI.h"

#include "APICallback.h"
#include "APIInstance.h"

#include "Misc/Base64.h"

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
	Handle = InRoute->HttpRouter->BindRoute(FHttpPath(InEntry.GetEntryUrl(InAPIInstance->Identity)), EntryModeMapE[InEntry.Mode], FHttpRequestHandler::CreateLambda(
		[&InEntry, InRoute, InAPIInstance](const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
		{

			const FAPIIdentity& Identity = InAPIInstance->Identity;
			TUniquePtr<FHttpServerResponse> Response;
			FGuid UserId;

			if (InEntry.bSessionPath)
			{
				FString SessionPathToken = Request.PathParams.Contains(FVActAPI::SessionPathTokenKey) 
					? Request.PathParams[FVActAPI::SessionPathTokenKey]
					: TEXT("");

				const bool bBounce = !InAPIInstance->Bouncer(SessionPathToken);
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

				const bool bReject = !InAPIInstance->Guard(SessionToken, UserId);
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
				const bool bDecline = !InAPIInstance->Reception(RequestCode);
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
					|| !InAPIInstance->Authenticate(UserName, HashedPassword, SessionToken, UserId);

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
				const bool bFailed = !InEntry.Callback->OnDataIn(Request, InEntry, InRoute, InAPIInstance, UserId);

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

				const bool bFailed = !InEntry.Callback->OnDataOut(Response, Request, InEntry, InRoute, InAPIInstance, UserId)
					&& !(!InEntry.bEncrypt || InAPIInstance->Encrypt(Data, Data, UserId));

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
