#pragma once
#include "UpsampleConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;


extern int partyNum;
class UpsampleLayer : public Layer
{
private:
    UpsampleConfig conf;
    RSSVectorMyType activations;

public:
    //Constructor and initializer
    UpsampleLayer(UpsampleConfig* conf, int _layerNum);

	//Functions
	void printLayer() override;
	void forward(const RSSVectorMyType& inputActivation) override;

    //Getters
	RSSVectorMyType* getActivation() {return &activations;};
};