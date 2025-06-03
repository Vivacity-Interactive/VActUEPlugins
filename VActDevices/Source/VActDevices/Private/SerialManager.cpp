#include "SerialManager.h"

ASerialManager::ASerialManager()
{
	Component = CreateDefaultSubobject<USerialManagerComponent>(TEXT("SerialManagerComponent0"));
	RootComponent = Component;
}