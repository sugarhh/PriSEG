#pragma once
#include "SigmoidConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;

extern int partyNum;

class SigmoidLayer : public Layer
{
private:
    SigmoidConfig conf;
    RSSVectorMyType activations;
    // RSSVectorMyType deltas;
    // RSSVectorSmallType sigmoidPrime;

public:
    // Constructor and initializer
    SigmoidLayer(SigmoidConfig* conf, int _layerNum);

    // Functions
    void printLayer() override;
    void forward(const RSSVectorMyType& inputActivation) override;
    // void computeDelta(RSSVectorMyType& prevDelta) override;
    // void updateEquations(const RSSVectorMyType& prevActivations) override;

    // Getters
    RSSVectorMyType* getActivation() { return &activations; };
    // RSSVectorMyType* getDelta() { return &deltas; };
};