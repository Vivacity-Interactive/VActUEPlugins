#include "BeatVectorModern.h"

const TArray<FName> FBeatVectorModern::FeatureNames = {
	FName(TEXT("Causality")),
	FName(TEXT("Branch")),
	FName(TEXT("Correlation")),

	FName(TEXT("Interrupt")),
	FName(TEXT("Distance")),
	FName(TEXT("Facing")),
	FName(TEXT("Direction")),
	FName(TEXT("Approach")),
	FName(TEXT("Focus")),
	FName(TEXT("BAC")),
	FName(TEXT("Shame")),
	FName(TEXT("Resentment")),
	FName(TEXT("Anxious")),
	FName(TEXT("Excitement")),
	FName(TEXT("Confidence")),
	FName(TEXT("Confort")),
	FName(TEXT("Arousal")),
	FName(TEXT("Mindfull")),
	FName(TEXT("Empathy")),
	FName(TEXT("Sympathy"))
};

const TMap<FName, int32> FBeatVectorModern::MapFeatureNames = {
	{ FeatureNames[0], 0},
	{ FeatureNames[1], 1},
	{ FeatureNames[2], 2},

	{ FeatureNames[3], 3},
	{ FeatureNames[4], 4},
	{ FeatureNames[5], 5},
	{ FeatureNames[6], 6},
	{ FeatureNames[7], 7},
	{ FeatureNames[8], 8},
	{ FeatureNames[9], 9},
	{ FeatureNames[10], 10},
	{ FeatureNames[11], 11},
	{ FeatureNames[12], 12},
	{ FeatureNames[13], 13},
	{ FeatureNames[14], 14},
	{ FeatureNames[15], 15},
	{ FeatureNames[16], 16},
	{ FeatureNames[17], 17},
	{ FeatureNames[18], 18},
	{ FeatureNames[19], 19}
};