
#pragma once
#include "LayerConfig.h"
#include "globals.h"
using namespace std;

class UpsampleConfig : public LayerConfig
{
public:
	size_t imageHeight = 0;
	size_t imageWidth = 0;
	size_t inputFeatures = 0;	//#Input feature maps
	size_t batchSize = 0;
	size_t upsampleFactor = 0;


	UpsampleConfig(size_t _imageHeight, size_t _imageWidth, size_t _inputFeatures, size_t _batchSize, size_t _upsampleFactor)
	:imageHeight(_imageHeight),
	 imageWidth(_imageWidth),
	 inputFeatures(_inputFeatures),
	 batchSize(_batchSize),
	 upsampleFactor(_upsampleFactor),
	 LayerConfig("Upsample")
	{};
};
