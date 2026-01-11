
#pragma once
#include "NeuralNetConfig.h"
#include "Layer.h"
#include "globals.h"
using namespace std;

class NeuralNetwork
{
public:
	RSSVectorMyType inputData;
	RSSVectorMyType outputData;
	vector<Layer*> layers;

	NeuralNetwork(NeuralNetConfig* config);
	~NeuralNetwork();
	void forward();
	void concatenateAndForward(size_t current_layer_index, size_t index1, size_t index2, size_t channel_size1, size_t channel_size2, size_t height, size_t width);
	void addResidualConnection(size_t current_layer_index, size_t index1, size_t index2, size_t channel_size,  size_t height, size_t width);
	void concatenateAndForward_6(size_t current_layer_index);
	void concatenateResidualAndForward(size_t current_layer_index, size_t index1, size_t index2, size_t channel_size1, size_t channel_size2, size_t channel_size3, size_t height, size_t width);
	void backward();
	void computeDelta();
	void updateEquations();
	void predict(RSSVectorMyType &maxIndex);
	void getAccuracy(const RSSVectorMyType &maxIndex, vector<size_t> &counter);
};