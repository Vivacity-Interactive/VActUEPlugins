#pragma once

#include "GenericPlatform/GenericPlatformMisc.h"

#include "CoreMinimal.h"
#include "Containers/ContainerAllocationPolicies.h"

//template<typename ElementType, template<typename> class HandlerType>
//struct FIteratorHandler;

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


//template<typename ElementType, template<typename> class HandlerType>
//struct VACTBASE_API FIteratorHandler
//{
//	int32 Index;
//
//	HandlerType<ElementType>& Handler;
//
//	FIteratorHandler(HandlerType<ElementType>& InHandler)
//		: Index(0)
//		, Handler(InHandler)
//	{
//	}
//
//	operator bool() const { return Index < Handler.Num(); }
//
//	bool operator !=(const FIteratorHandler& Other) const
//	{
//		return Index != Other.Index;
//	}
//
//	FIteratorHandler& operator++()
//	{
//		++Index;
//		return *this;
//	}
//
//	FIteratorHandler& operator--()
//	{
//		--Index;
//		return *this;
//	}
//
//	ElementType& operator*()
//	{
//		int32 _Index = Handler.BufferIndex(Index);
//		return (*Handler.Array)[_Index];
//	}
//
//	ElementType* operator->()
//	{
//		int32 _Index = Handler.BufferIndex(Index);
//		return &(*Handler.Array)[_Index];
//	}
//};
//
//template<typename ElementType, template<typename> class HandlerType>
//FIteratorHandler(HandlerType<ElementType>&) -> FIteratorHandler<ElementType, HandlerType>;