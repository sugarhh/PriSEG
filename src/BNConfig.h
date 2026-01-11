
#pragma once
#include "LayerConfig.h"
#include "globals.h"
using namespace std;

// class BNConfig : public LayerConfig
// {
// public:
// 	size_t inputSize = 0;
// 	size_t numBatches = 0;

// 	BNConfig(size_t _inputSize, size_t _numBatches)
// 	:inputSize(_inputSize),
// 	 numBatches(_numBatches),
// 	 LayerConfig("BN")
// 	{};
// };

class BNConfig : public LayerConfig
{
public:
    size_t inputSize = 0;
    size_t numBatches = 0;
    size_t channels = 0;
    size_t height = 0;
    size_t width = 0;

    BNConfig(size_t _inputSize, size_t _numBatches, size_t _channels, size_t _height, size_t _width)
    : inputSize(_inputSize),
      numBatches(_numBatches),
      channels(_channels),
      height(_height),
      width(_width),
      LayerConfig("BN")
    {};
};
