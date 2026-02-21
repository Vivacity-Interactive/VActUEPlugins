#include "BeatVectorReduced.h"

const TArray<FName> FBeatVectorReduced::FeatureNames = {
	FName(TEXT("Causality")),
	FName(TEXT("Branch")),
	FName(TEXT("Correlation")),

	FName(TEXT("Interrupt")),
	FName(TEXT("Distance")),
	FName(TEXT("Facing")),
	FName(TEXT("Focus")),
	FName(TEXT("Comfort")),
	FName(TEXT("Secure")),
	FName(TEXT("Trusting")),
	FName(TEXT("Awareness"))
};

const TMap<FName, int32> FBeatVectorReduced::MapFeatureNames = {
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
	{ FeatureNames[10], 10}
};