#pragma once
#include <iostream>
#include "BNLayer.h"
#include "Functionalities.h"
#include "tools.h"
using namespace std;


// BNLayer::BNLayer(BNConfig* conf, int _layerNum)
// :Layer(_layerNum),
//  conf(conf->inputSize, conf->numBatches),
//  gamma(conf->numBatches),
//  beta(conf->numBatches),
//  xhat(conf->numBatches * conf->inputSize),
//  sigma(conf->numBatches),
//  activations(conf->inputSize * conf->numBatches),
//  deltas(conf->inputSize * conf->numBatches)
// {initialize();};
BNLayer::BNLayer(BNConfig* conf, int _layerNum)
    : Layer(_layerNum),
      conf(conf->inputSize, conf->numBatches, conf->channels, conf->height, conf->width),
      gamma(conf->channels),
      beta(conf->channels),
      var(conf->channels),
      mean(conf->channels),
      xhat(conf->numBatches * conf->channels * conf->height * conf->width),
      sigma(conf->numBatches * conf->channels),
      activations(conf->numBatches * conf->channels * conf->height * conf->width),
      deltas(conf->numBatches * conf->channels * conf->height * conf->width)
{
    initialize();
};


void BNLayer::initialize() {};


void BNLayer::printLayer()
{
	cout << "----------------------------------------------" << endl;  	
	cout << "(" << layerNum+1 << ") BN Layer\t\t  " << conf.inputSize << " x " 
		 << conf.numBatches << endl;
}
extern void funcReconstruct(const RSSVectorMyType &a, vector<myType> &b, size_t size, string str, bool print);
void BNLayer::forward(const RSSVectorMyType& inputActivation)
{   
    log_print("BN.forward");

    size_t B = conf.numBatches;  // batch size
    size_t C = conf.channels;    // number of channels
    size_t H = conf.height;      // image height
    size_t W = conf.width;       // image width
    size_t EPSILON = (myType)(1 << (FLOAT_PRECISION - 8));
    size_t INITIAL_GUESS = (myType)(1 << (FLOAT_PRECISION));
    size_t SQRT_ROUNDS = 10;

    vector<myType> eps(C, EPSILON), initG(C, INITIAL_GUESS);
    RSSVectorMyType epsilon(C,make_pair(0,0)), mu(C, make_pair(0,0)), b(C, make_pair(0,0));
    RSSVectorMyType divisor(B*C, make_pair(0,0));

    RSSVectorMyType temp(B*C*H*W, make_pair(0,0)), var1(B*C*H*W, make_pair(0,0));
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w){
                    temp[b*C*H*W + c*H*W + h*W + w] = inputActivation[b*C*H*W + c*H*W + h*W + w] - mean[c];
                    var1[b*C*H*W + c*H*W + h*W + w] = var[c];
                }
	
    
    funcDotProduct(temp, var1, xhat, B*C*H*W, true, FLOAT_PRECISION);
    
    RSSVectorMyType g_repeat(B*C*H*W), b_repeat(B*C*H*W);
 
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                {
                    g_repeat[b*C*H*W + c*H*W + h*W + w] = gamma[c];
                    b_repeat[b*C*H*W + c*H*W + h*W + w] = beta[c];
                }
    funcDotProduct(xhat, g_repeat, xhat, B*C*H*W, true, FLOAT_PRECISION);
    addVectors<RSSMyType>(xhat, b_repeat, activations, B*C*H*W);


    // vector<myType> ss(B*C*H*W);
	// funcReconstruct(sigma1, ss, B*C*H*W, "anything", false);
    // vector<myType> tt(B*C*H*W);
	// funcReconstruct(temp1, tt, B*C*H*W, "anything", false);
    // vector<myType> temp11(B*C*H*W, 0);
    // for (int i = 0; i < B*C*H*W; ++i){
	// 	float sss = (static_cast<int>(ss[i]))/(float)(1 << FLOAT_PRECISION);
    //     float ttt = (static_cast<int>(tt[i]))/(float)(1 << FLOAT_PRECISION);
    //     float st = sss*ttt;
    //     temp11[i] = floatToMyType(st);
	// }
    // funcGetShares(xhat, temp11);

    /** 训练阶段
    // Compute mean for each channel
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
        {
            for (int h = 0; h < H; ++h)
            {
                RSSVectorMyType mut(C,make_pair(0,0));
                for (int w = 0; w < W; ++w){
                    // mu[i*C+c] =mu[i*C+c] + inputActivation[i*C*H*W + c*H*W + h*W + w];
                    mut[c] =mut[c] + inputActivation[b*C*H*W + c*H*W + h*W + w];
                }
                funcTruncatePublic(mut, B*H*W, C);
                mu[c]=mut[c] + mu[c];
            }
                

        }
    print_vector(mu, "FLOAT", "均值:", C);


    // //计算mean,平均值
    // funcTruncatePublic(mu, B*H*W, C);


    // 计算x - mean for each channel
    RSSVectorMyType temp1(B*C*H*W, make_pair(0,0));
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    temp1[b*C*H*W + c*H*W + h*W + w] = inputActivation[b*C*H*W + c*H*W + h*W + w] - mu[c];
	
    print_vector(temp1, "FLOAT", "temp1:", 30);
	//cout<<2<<endl;

    // 计算 (x-mean)^2的和 for each channel
    RSSVectorMyType temp2(B*C*H*W, make_pair(0,0)), temp3(C, make_pair(0,0));
    funcDotProduct(temp1, temp1, temp2, B*C*H*W, true, FLOAT_PRECISION);   //temp2存所有值对应的(xi-mu)^2
    
    print_vector(temp2, "FLOAT", "temp2:", 32);
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
            {   
                RSSVectorMyType temp33(C, make_pair(0,0));
                for (int w = 0; w < W; ++w)
                    temp33[c] =temp33[c] + temp2[b*C*H*W + c*H*W + h*W + w];    //计算三个通道的和
                funcTruncatePublic(temp33, B*H*W, C);  //除法
                temp3[c] = temp3[c] +temp33[c];
            }
                

	//cout<<3<<endl;
    // 计算方差 (variance + epsilon)
    // funcTruncatePublic(temp3, B*H*W, C);  //除法

    //计算方差+eps
    print_vector(temp3, "FLOAT", "方差:", C);
    //print_vector(epsilon, "FLOAT", "epsilon0:", 6);
    funcGetShares(epsilon, eps);
    addVectors<RSSMyType>(temp3, epsilon, temp3, C);
    // print_vector(temp3, "FLOAT", "方差+episino:", 3);
    // print_vector(epsilon, "FLOAT", "epsilon:", 3);
	//cout<<4<<endl;

    // 方差+eps开根号
    //print_vector(sigma, "FLOAT", "sigma0:", 3);
    funcGetShares(sigma, initG);
    /////换一个开根号的方式
    for (int i = 0; i < SQRT_ROUNDS; ++i)
    {   
        funcDivision(temp3, sigma, b, C);
        // print_vector(temp3, "FLOAT", "t3:", 3);
        // print_vector(sigma, "FLOAT", "s:", 3);
        // print_vector(b, "FLOAT", "b:", 3);
        addVectors<RSSMyType>(sigma, b, sigma, C);
        funcTruncatePublic(sigma, 2, C);
        // print_vector(sigma, "FLOAT", "sigma0:", 3);
    }
    print_vector(sigma, "FLOAT", "方差+episino开根号:", C);

    // Normalized x (xhat)  !!!!!!!!!!
    RSSVectorMyType xhat(B*C*H*W, make_pair(0,0));
    RSSVectorMyType sigma0(C, make_pair(0,0));
    RSSVectorMyType sigma1(B*C*H*W, make_pair(0,0));
    // 除法不行
    //funcBatchNorm(temp1, sigma, xhat, B, C, H, W);

    //STR_begin
    vector<myType> sigma_y(C, 0);
    funcReconstruct(sigma, sigma_y, C, "Sigma Reconst", false);

    vector<float> sigma_y0(C, 0);
    for (int c = 0; c < C; ++c){
        sigma_y0[c] = 1.0 * float(1 << (FLOAT_PRECISION))  /  static_cast<int> (sigma_y[c]);
        // cout<<sigma_y0[c]<<endl;
        sigma_y[c] = floatToMyType(sigma_y0[c]);
        //cout<<sigma_y[c]<<endl;
    } 
    funcGetShares(sigma0, sigma_y);
    // print_vector(sigma0, "FLOAT", "sigma_0:", 3);

    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    sigma1[b*C*H*W + c*H*W + h*W + w] = sigma0[c];
    // print_vector(sigma1, "FLOAT", "sigma1:", 100);
    // print_vector(temp1, "FLOAT", "temp1:", 100);
    funcDotProduct(temp1, sigma1, xhat, B*C*H*W, true, FLOAT_PRECISION);
    vector<myType> ss(B*C*H*W);
	funcReconstruct(sigma1, ss, B*C*H*W, "anything", false);
    vector<myType> tt(B*C*H*W);
	funcReconstruct(temp1, tt, B*C*H*W, "anything", false);
    vector<myType> temp11(B*C*H*W, 0);
    for (int i = 0; i < B*C*H*W; ++i){
		float sss = (static_cast<int>(ss[i]))/(float)(1 << FLOAT_PRECISION);
        float ttt = (static_cast<int>(tt[i]))/(float)(1 << FLOAT_PRECISION);
        float st = sss*ttt;
        temp11[i] = floatToMyType(st);
	}
    funcGetShares(xhat, temp11);
    // vector<myType> sigma11(B*C*H*W, 0);
    // funcReconstruct(sigma1, sigma11, B*C*H*W, "Sigma Reconst", false);
    // cout<<sigma11[0]<<endl;
    // vector<myType> temp11(B*C*H*W, 0);
    // funcReconstruct(temp1, temp11, B*C*H*W, "temp Reconst", false);
    // float t = (static_cast<int64_t>(temp11[2]))/(float)(1 << FLOAT_PRECISION);
    // cout<< t <<endl;

    // vector<float> ans(B*H*C*W, 0);
    // for (int i = 0; i < B*H*C*W; ++i){
    //     ans[i] = 1.0 * float(temp11[i]) * float(1 << (FLOAT_PRECISION)) * float(sigma11[i])* float(1 << (FLOAT_PRECISION));
    //    // cout<<ans[i]<<endl;
    //     temp11[i] = floatToMyType(ans[i]);
    // } 
    // funcGetShares(xhat, temp11);

    //print_vector(xhat, "FLOAT", "xhat:", 10);

    //STR_end

    // Scaling
    RSSVectorMyType g_repeat(B*C*H*W), b_repeat(B*C*H*W);
    // print_vector(gamma, "FLOAT", "gamma:", 3);
    // print_vector(beta, "FLOAT", "beta:", 3);
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                {
                    g_repeat[b*C*H*W + c*H*W + h*W + w] = gamma[c];
                    b_repeat[b*C*H*W + c*H*W + h*W + w] = beta[c];
                }

    // funcDotProduct(xhat, xhat, activations, B*C*H*W, true, FLOAT_PRECISION);
    funcReconstruct(g_repeat, ss, B*C*H*W, "anything", false);
    funcReconstruct(xhat, tt, B*C*H*W, "anything", false);
    for (int i = 0; i < B*C*H*W; ++i){
		float sss = (static_cast<int32_t>(ss[i]))/(float)(1 << FLOAT_PRECISION);
        float ttt = (static_cast<int32_t>(tt[i]))/(float)(1 << FLOAT_PRECISION);
        float st = sss*ttt;
        temp11[i] = floatToMyType(st);
	}
     funcGetShares(xhat, temp11);
    //print_vector(activations, "FLOAT", "activations:", 3);
    addVectors<RSSMyType>(xhat, b_repeat, activations, B*C*H*W);
    **/

}

// void BNLayer::forward(const RSSVectorMyType& inputActivation)
// {
// 	log_print("BN.forward");

// 	size_t B = conf.numBatches;
// 	size_t m = conf.inputSize;
// 	size_t EPSILON = (myType)(1 << (FLOAT_PRECISION - 8));
// 	// TODO: Accept initialization from the paper
// 	size_t INITIAL_GUESS = (myType)(1 << (FLOAT_PRECISION));
// 	size_t SQRT_ROUNDS = 4;

// 	vector<myType> eps(B, EPSILON), initG(B, INITIAL_GUESS);
// 	RSSVectorMyType epsilon(B), mu(B, make_pair(0,0)), b(B);
// 	RSSVectorMyType divisor(B, make_pair(0,0));

// 	//Compute mean
// 	for (int i = 0; i < B; ++i)
// 		for (int j = 0; j < m; ++j)
// 			mu[i] = mu[i] + inputActivation[i*m+j];
// 	funcTruncatePublic(mu, m, B);	

// 	//Compute x - mean
// 	RSSVectorMyType temp1(B*m);
// 	for (int i = 0; i < B; ++i)
// 		for (int j = 0; j < m; ++j)
// 			temp1[i*m+j] = inputActivation[i*m+j] - mu[i];

// 	//Compute (x-mean)^2
// 	RSSVectorMyType temp2(B*m), temp3(B, make_pair(0,0));
// 	funcDotProduct(temp1, temp1, temp2, B*m, true, FLOAT_PRECISION); 
// 	for (int i = 0; i < B; ++i)
// 		for (int j = 0; j < m; ++j)
// 			temp3[i] = temp3[i] + temp2[i*m+j];

// 	//Compute (variance + epsilon)
// 	funcTruncatePublic(temp3, m, B);
// 	funcGetShares(epsilon, eps);
// 	addVectors<RSSMyType>(temp3, epsilon, temp3, B);
		
// 	//Square Root
// 	funcGetShares(sigma, initG);
// 	for (int i = 0; i < SQRT_ROUNDS; ++i)
// 	{
// 		funcDivision(temp3, sigma, b, B);
// 		addVectors<RSSMyType>(sigma, b, sigma, B);
// 		funcTruncatePublic(sigma, 2, B);
// 	}

// 	//Normalized x (xhat)
// 	funcBatchNorm(temp1, sigma, xhat, m, B);

// 	//Scaling
// 	RSSVectorMyType g_repeat(B*m);
// 	for (int i = 0; i < B; ++i)
// 		for (int j = 0; j < m; ++j)
// 			g_repeat[i*m+j] = gamma[i];

// 	funcDotProduct(g_repeat, xhat, activations, B*m, true, FLOAT_PRECISION);
// 	for (int i = 0; i < B; ++i)
// 		for (int j = 0; j < m; ++j)
// 			activations[i*m+j] = activations[i*m+j] + beta[i];
// }


//https://kevinzakka.github.io/2016/09/14/batch_normalization/
void BNLayer::computeDelta(RSSVectorMyType& prevDelta)
{
	log_print("BN.computeDelta");

	size_t B = conf.numBatches;
	size_t m = conf.inputSize;

	//Derivative with xhat
	RSSVectorMyType g_repeat(B*m), dxhat(B*m);
	for (int i = 0; i < B; ++i)
		for (int j = 0; j < m; ++j)
			g_repeat[i*m+j] = gamma[i];

	funcDotProduct(g_repeat, deltas, dxhat, B*m, true, FLOAT_PRECISION);

	//First term
	RSSVectorMyType temp1(B*m);
	for (int i = 0; i < B; ++i)
		for (int j = 0; j < m; ++j)
			temp1[i*m+j] = ((myType)m) * dxhat[i*m+j];

	//Second term	
	RSSVectorMyType temp2(B*m, make_pair(0,0));
	for (int i = 0; i < B; ++i)
		for (int j = 0; j < m; ++j)
			temp2[i*m] = temp2[i*m] + dxhat[i*m+j];

	for (int i = 0; i < B; ++i)
		for (int j = 0; j < m; ++j)
			temp2[i*m + j] = temp2[i*m];

	//Third term
	RSSVectorMyType temp3(B*m, make_pair(0,0));
	funcDotProduct(dxhat, xhat, temp3, B*m, true, FLOAT_PRECISION);
	for (int i = 0; i < B; ++i)
		for (int j = 1; j < m; ++j)
			temp3[i*m] = temp3[i*m] + temp3[i*m+j];

	for (int i = 0; i < B; ++i)
		for (int j = 0; j < m; ++j)
			temp3[i*m + j] = temp3[i*m];

	funcDotProduct(temp3, xhat, temp3, B*m, true, FLOAT_PRECISION);

	//Numerator
	subtractVectors<RSSMyType>(temp1, temp2, temp1, B*m);
	subtractVectors<RSSMyType>(temp1, temp3, temp1, B*m);

	RSSVectorMyType temp4(B);
	for (int i = 0; i < B; ++i)
		temp4[i] = ((myType)m) * sigma[i];

	//funcBatchNorm(temp1, temp4, prevDelta, m, B);
}

void BNLayer::updateEquations(const RSSVectorMyType& prevActivations)
{
	log_print("BN.updateEquations");

	size_t B = conf.numBatches;
	size_t m = conf.inputSize;

	//Update beta
	RSSVectorMyType temp1(B, make_pair(0,0));
	for (int i = 0; i < B; ++i)
		for (int j = 0; j < m; ++j)
			temp1[i] = temp1[i] + deltas[i*m + j];

	subtractVectors<RSSMyType>(beta, temp1, beta, B);


	//Update gamma
	RSSVectorMyType temp2(B*m, make_pair(0,0)), temp3(B, make_pair(0,0));
	funcDotProduct(xhat, deltas, temp2, B*m, true, FLOAT_PRECISION);
	for (int i = 0; i < B; ++i)
		for (int j = 0; j < m; ++j)
			temp3[i] = temp3[i] + temp2[i*m + j];

	subtractVectors<RSSMyType>(gamma, temp3, gamma, B);
}