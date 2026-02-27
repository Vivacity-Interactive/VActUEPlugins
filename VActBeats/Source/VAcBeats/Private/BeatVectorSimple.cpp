#include "BeatVectorSimple.h"

const TArray<FName> FBeatVectorSimple::FeatureNames = {
	FName(TEXT("Causality")),
	FName(TEXT("Branch")),
	FName(TEXT("Correlation")),

	FName(TEXT("Distance")),
	FName(TEXT("Focus")),
	FName(TEXT("Friendly")),
	FName(TEXT("Safe")),
	FName(TEXT("Aware"))
};

const TMap<FName, int32> FBeatVectorSimple::MapFeatureNames = {
	{ FeatureNames[0], 0},
	{ FeatureNames[1], 1},
	{ FeatureNames[2], 2},

	{ FeatureNames[3], 3},
	{ FeatureNames[4], 4},
	{ FeatureNames[5], 5},
	{ FeatureNames[6], 6},
	{ FeatureNames[7], 7}
};