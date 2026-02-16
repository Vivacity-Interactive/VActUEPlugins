#include "VActBeats.h"

const TArray<FName> FVActBeats::NameBeatVector = 
{
	FName("Causality"),
	FName("Branch"),
	FName("Correlation"),
	FName("Used"),
	FName("Interrupt"),
	FName("Distance"),
	FName("Facing"),
	FName("Direction"),
	FName("Approach"),
	FName("Focus"),
	FName("BAC"),
	FName("Shame"),
	FName("Resentment"),
	FName("Anxious"),
	FName("Excitement"),
	FName("Confidence"),
	FName("Comfort"),
	FName("Arousal"),
	FName("Mindfool"),
	FName("Empathy"),
	FName("Sympathy"),
	FName("_Noise")
};

const TArray<FName> FVActBeats::NameBeatVectorMinimal =
{
	FName("Causality"),
	FName("Used"),
	FName("Interrupt"),
	FName("Distance"),
	FName("Focus"),
	FName("Aware"),
	FName("Comfort"),
	FName("Secure"),
	FName("Trusting"),
	FName("_Noise")
};

const TMap<FName, int32> FVActBeats::IdBeatVector = 
{
	{ NameBeatVector[0], 0},
	{ NameBeatVector[1], 1},
	{ NameBeatVector[2], 2},
	{ NameBeatVector[3], 3},
	{ NameBeatVector[4], 4},
	{ NameBeatVector[5], 5},
	{ NameBeatVector[6], 6},
	{ NameBeatVector[7], 7},
	{ NameBeatVector[8], 8},
	{ NameBeatVector[9], 9},
	{ NameBeatVector[10], 10},
	{ NameBeatVector[11], 11},
	{ NameBeatVector[12], 12},
	{ NameBeatVector[13], 13},
	{ NameBeatVector[14], 14},
	{ NameBeatVector[15], 15},
	{ NameBeatVector[16], 16},
	{ NameBeatVector[17], 17},
	{ NameBeatVector[18], 18},
	{ NameBeatVector[19], 19},
	{ NameBeatVector[20], 20},
	{ NameBeatVector[21], 21}
};

const TMap<FName, int32> FVActBeats::IdBeatVectorMinimal = 
{
	{ NameBeatVectorMinimal[0], 0},
	{ NameBeatVectorMinimal[1], 1},
	{ NameBeatVectorMinimal[2], 2},
	{ NameBeatVectorMinimal[3], 3},
	{ NameBeatVectorMinimal[4], 4},
	{ NameBeatVectorMinimal[5], 5},
	{ NameBeatVectorMinimal[6], 6},
	{ NameBeatVectorMinimal[7], 7},
	{ NameBeatVectorMinimal[8], 8},
	{ NameBeatVectorMinimal[9], 9}
};

