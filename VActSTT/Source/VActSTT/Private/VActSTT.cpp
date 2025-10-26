#include "VActSTT.h"


void FVActSTT::_Unsafe_Create(FSTTModel& Into)
{

}
	
void FVActSTT::_Unsafe_Destroy(FSTTModel& Tensor, bool bData = true)
{

}

FSTTModel FVActSTT::CreateModel()
{
	FSTTModel Into;
	_Unsafe_Create(Into);
	return Into;
}

void FVActSTT::Destroy(FSTTModel& Tensor, bool bData = true)
{
	_Unsafe_Destroy(Tensor, bData);
}