// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "GenericPlatform/GenericPlatformMisc.h"

#include "CoreMinimal.h"
#include "Containers/ContainerAllocationPolicies.h"

template<class InOwnerClass>
class VACTBASE_API TIterateHandler
{
	template<class OtherAllocator>
	friend class TIterateHandler;

public:
	typedef InOwnerClass OwnerClass;
	typedef typename InOwnerClass::SizeType SizeType;

public:
	SizeType Index;

	SizeType Counter;

private:
	SizeType _Tracker;
	
	const OwnerClass* _Owner;

	TIterateHandler() : TIterateHandler(nullptr, 0)
	{
	}

public:
	TIterateHandler(const OwnerClass& Owner, SizeType Tracker = 0) : TIterateHandler(&Owner, Tracker)
	{
	}


	TIterateHandler(const OwnerClass* Owner, SizeType Tracker = 0) : _Owner(Owner)
	{
		Reset();
		_Tracker = Tracker;
	}

	~TIterateHandler()
	{
	}

	FORCEINLINE SizeType Tracker() const
	{
		return _Tracker;
	}

	FORCEINLINE bool Next()
	{
		return _Owner->_Next(Counter, _Tracker, Index);
	}

	FORCEINLINE bool Next(SizeType& OutIndex)
	{
		const bool bNext = Next();
		OutIndex = Index;
		return bNext;
	}

	FORCEINLINE void Reset()
	{
		_Tracker = 0;
		Counter = 0;
		Index = 0;
	}

	FORCEINLINE bool IsValid()
	{
		return _Owner != nullptr;
	}
};
