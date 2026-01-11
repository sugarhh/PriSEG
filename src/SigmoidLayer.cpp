#pragma once
#include "SigmoidLayer.h"
#include "Functionalities.h"
using namespace std;

const myType HALF = floatToMyType(0.5);
const myType ONE_SIXTH = floatToMyType(1.0/6);
const myType ONE_TWELFTH = floatToMyType(1.0/12);


SigmoidLayer::SigmoidLayer(SigmoidConfig* conf, int _layerNum)
    : Layer(_layerNum),
      conf(conf->inputDim, conf->batchSize),
      activations(conf->batchSize * conf->inputDim)
    //   deltas(conf->batchSize * conf->inputDim),
    //   sigmoidPrime(conf->batchSize * conf->inputDim)
{}

void SigmoidLayer::printLayer()
{
    cout << "----------------------------------------------" << endl;
    cout << "(" << layerNum + 1 << ") Sigmoid Layer\t\t  " << conf.batchSize << " x " << conf.inputDim << endl;
}

void SigmoidLayer::forward(const RSSVectorMyType& inputActivation)
{
    log_print("Sigmoid.forward");

    size_t rows = conf.batchSize;
    size_t columns = conf.inputDim;
    size_t size = rows * columns;

    // if (FUNCTION_TIME)
    //     cout << "funcSigmoid: " << funcTime(funcSigmoid, inputActivation, sigmoidPrime, activations, size) << endl;
    // else
    //    funcSigmoid(inputActivation, activations, size);
    cout<<rows<<endl;
    cout<<columns<<endl;
    cout<<size<<endl;
    RSSVectorMyType numerator(size,make_pair(0,0)), denominator(size,make_pair(0,0)), y(size);

    // 计算分子
    size_t t=5;
    //multiplyByScalar(inputActivation, t, numerator); //0.5x
    multiplyByScalar(inputActivation, 6, numerator);
    // print_vector(numerator, "FLOAT", "6x:", 3);
    RSSVectorMyType a2(size, make_pair(0,0));
   
    funcDotProduct(inputActivation, inputActivation, a2, size, true, FLOAT_PRECISION); // a^2
    //multiplyByScalar(a2, ONE_TWELFTH , temp);
    // print_vector(a2, "FLOAT", "x^2:", 3);
    addVectors<RSSMyType>(numerator, a2, numerator, size);
    //生成一个常数向量
    vector<myType> num1(size, floatToMyType(12.0f));
    RSSVectorMyType c1(num1.size());
    funcGetShares(c1, num1);
    addVectors<RSSMyType>(numerator, c1, numerator, size);
    
    // 计算分母
    t = 2;
    multiplyByScalar(a2, t, denominator);
    vector<myType> num2(size, floatToMyType(24.0f));
    RSSVectorMyType c2(num2.size());
    funcGetShares(c2, num2);
    addVectors<RSSMyType>(denominator, c2, denominator, size);
    
    // print_vector(denominator, "FLOAT", "denominator:", 3);
    // print_vector(numerator, "FLOAT", "fenzi:", 3);

    // STR
    vector<myType> fenmu(size, 0);
    funcReconstruct(denominator, fenmu, size, "sigmoid Reconst", false);    //重构分母

    vector<float> fenmu0(size, 0);
    RSSVectorMyType fenmu1(size);
    for (int i = 0; i < size; ++i){                                         //计算分母的倒数
        fenmu0[i] = 1.0 * float(1 << (FLOAT_PRECISION))  /  float(fenmu[i]);
        //cout<<sigma_y0[c]<<endl;
        fenmu[i] = floatToMyType(fenmu0[i]);
        //cout<<sigma_y[c]<<endl;
    } 
    funcGetShares(fenmu1, fenmu);

    funcDotProduct(fenmu1, numerator, activations, size, true, FLOAT_PRECISION);
    // 计算 y = 分子 / 分母
    // funcDivision(numerator, denominator, activations, size);


}

// void SigmoidLayer::computeDelta(RSSVectorMyType& prevDelta)
// {
//     log_print("Sigmoid.computeDelta");

//     // Back Propagate
//     size_t rows = conf.batchSize;
//     size_t columns = conf.inputDim;
//     size_t size = rows * columns;

//     if (FUNCTION_TIME)
//         cout << "funcSelectShares: " << funcTime(funcSelectShares, deltas, sigmoidPrime, prevDelta, size) << endl;
//     else
//         funcSelectShares(deltas, sigmoidPrime, prevDelta, size);
// }

// void SigmoidLayer::updateEquations(const RSSVectorMyType& prevActivations)
// {
//     log_print("Sigmoid.updateEquations");
// }