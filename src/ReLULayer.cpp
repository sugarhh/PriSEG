
#pragma once
#include "ReLULayer.h"
#include "Functionalities.h"
using namespace std;

ReLULayer::ReLULayer(ReLUConfig* conf, int _layerNum)
:Layer(_layerNum),
 conf(conf->inputDim, conf->batchSize),
 activations(conf->batchSize * conf->inputDim), 
 deltas(conf->batchSize * conf->inputDim),
 reluPrime(conf->batchSize * conf->inputDim)
{}


void ReLULayer::printLayer()
{
	cout << "----------------------------------------------" << endl;  	
	cout << "(" << layerNum+1 << ") ReLU Layer\t\t  " << conf.batchSize << " x " << conf.inputDim << endl;
}


void ReLULayer::forward(const RSSVectorMyType &inputActivation)
{
	log_print("ReLU.forward");

	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t size = rows*columns;
    size_t EPSILON = (myType)(1 << (FLOAT_PRECISION - 6));
    size_t INITIAL_GUESS = (myType)(1 << (FLOAT_PRECISION));
    size_t SQRT_ROUNDS = 6;  //6
	
	vector<myType>  initG(size, INITIAL_GUESS);
	RSSVectorMyType epsilon(size,make_pair(0,0));
	RSSVectorMyType temp(size,make_pair(0,0));
	
	// cout<<inputActivation.size()<<endl;
	funcDotProduct(inputActivation, inputActivation, temp, size, true, FLOAT_PRECISION);
	// print_vector(temp, "FLOAT", "x2:", 30);
	addVectors<RSSMyType>(temp, epsilon, temp, size);
	// print_vector(temp, "FLOAT", "开根号qian:", 30);

	RSSVectorMyType temp1(size), b(size);
	funcGetShares(temp1, initG);
    
    for (int i = 0; i < SQRT_ROUNDS; ++i)
    {   
        funcDivision(temp, temp1, b, size);
        // print_vector(temp3, "FLOAT", "t3:", 3);
        // print_vector(sigma, "FLOAT", "s:", 3);
        // print_vector(b, "FLOAT", "b:", 3);
        addVectors<RSSMyType>(temp1, b, temp1, size);
        funcTruncatePublic(temp1, 2, size);
        // print_vector(sigma, "FLOAT", "sigma0:", 3);
    }
	// print_vector(temp1, "FLOAT", "开根号:", 30);
	addVectors<RSSMyType>(temp1, inputActivation, activations, size);
	// print_vector(activations, "FLOAT", "fenmu :", 30);
	funcTruncatePublic(activations, 2, size);
	funcTruncate(activations, 9, size);
	multiplyByScalar(activations, 512, activations);
	
	// if (FUNCTION_TIME)
	// 	cout << "funcRELU: " << funcTime(funcRELU, inputActivation, reluPrime, activations, size) << endl;
	// else
	// funcRELU(inputActivation, reluPrime, activations, size);
	
}


void ReLULayer::computeDelta(RSSVectorMyType& prevDelta)
{
	log_print("ReLU.computeDelta");

	//Back Propagate	
	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t size = rows*columns;
	
	if (FUNCTION_TIME)
		cout << "funcSelectShares: " << funcTime(funcSelectShares, deltas, reluPrime, prevDelta, size) << endl;
	else
		funcSelectShares(deltas, reluPrime, prevDelta, size);
}


void ReLULayer::updateEquations(const RSSVectorMyType& prevActivations)
{
	log_print("ReLU.updateEquations");
}
