#include "APIRoute.h"

UAPIRoute::UAPIRoute()
	: Name(NAME_None)
	, Port(8080)
	, PortMaxOffset(10)
	, Route(TEXT(""))
	, Domain(TEXT(""))
{

}

UAPIRoute::UAPIRoute(FString InRoute, int32 InPort, FString InDomain)
	: Name(NAME_None)
	, Port(InPort)
	, PortMaxOffset(10)
	, Route(InRoute)
	, Domain(InDomain)
{

}