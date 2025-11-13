// Copyright Vivaicty Interactive, Inc. All Rights Reserved.

#pragma once

#include "Modules/ModuleManager.h"

class FVActAPIModule : public IModuleInterface
{
public:

	/** IModuleInterface implementation */
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};
