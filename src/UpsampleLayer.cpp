#pragma once
#include <omp.h>
#include "UpsampleLayer.h"
#include "tools.h"
#include "Functionalities.h"
using namespace std;

extern bool LARGE_NETWORK;

UpsampleLayer::UpsampleLayer(UpsampleConfig* conf, int _layerNum)
:Layer(_layerNum),
 conf(conf->imageHeight, conf->imageWidth, conf->inputFeatures, 
	  conf->batchSize, conf->upsampleFactor),
 activations(conf->batchSize *conf->imageHeight* conf->imageWidth* conf->inputFeatures* conf->upsampleFactor* conf->upsampleFactor)
{}


void UpsampleLayer::printLayer()
{
	cout << "----------------------------------------------" << endl;  	
	cout << "(" << layerNum+1 << ") Upsample Layer\t\t  "<< conf.batchSize<< " x " << conf.imageHeight *  conf.upsampleFactor << " x " << conf.imageWidth *  conf.upsampleFactor
		 << " x " << conf.inputFeatures << endl;
}


void UpsampleLayer::forward(const RSSVectorMyType& inputActivation)
{
    log_print("Upsample.forward");

    size_t B = conf.batchSize;
    size_t iw = conf.imageWidth;
    size_t ih = conf.imageHeight;
    size_t Din = conf.inputFeatures;
    size_t fa = conf.upsampleFactor;

    size_t h_scale = fa * ih, w_scale = fa * iw;
    float fa_t = static_cast<float>(ih-1) / static_cast<float>(h_scale-1);
    
    float x, y, t1, t2, t3, t4, t=1.0/(float)(1 << FLOAT_PRECISION);
    size_t x1, x2, y1, y2;
    vector<float> temp1(4);
    vector<myType> temp(B * Din * ih * iw), res(1);  
    funcReconstruct(inputActivation, temp, B * Din * ih * iw , "sigmoid Reconst", false);

    size_t total_features = B * Din;
    size_t total_pixels = h_scale * w_scale;

    for (size_t bd = 0; bd < total_features; ++bd) {
        size_t b = bd / Din;
        size_t d = bd % Din;

        for (size_t hw = 0; hw < total_pixels; ++hw) {
            size_t h = hw / w_scale;
            size_t w = hw % w_scale;

            y = fa_t * h;
            x = fa_t * w;
            
            y1 = std::max(0, std::min(static_cast<int>(y), static_cast<int>(ih - 1)));
            x1 = std::max(0, std::min(static_cast<int>(x), static_cast<int>(iw - 1)));
            y2 = std::min(y1 + 1, ih - 1);
            x2 = std::min(x1 + 1, iw - 1);

            temp1[0] = ((float)((int32_t)temp[b * Din * ih * iw + d * ih * iw + y1 * iw + x1])*t);
            temp1[1] = ((float)((int32_t)temp[b * Din * ih * iw + d * ih * iw + y1 * iw + x2])*t);
            temp1[2] = ((float)((int32_t)temp[b * Din * ih * iw + d * ih * iw + y2 * iw + x1])*t);
            temp1[3] = ((float)((int32_t)temp[b * Din * ih * iw + d * ih * iw + y2 * iw + x2])*t);

            t1 = (float(x2)-x)*(float(y2)-y);
            t3 = (float(x2)-x)*(y-float(y1));
            t2 = (x-float(x1))*(float(y2)-y);
            t4 = (x-float(x1))*(y-float(y1));

            float z = t1*temp1[0] + t2*temp1[1] + t3*temp1[2] + t4*temp1[3];
            res[0] = floatToMyType(z);
            RSSVectorMyType coeffs(1);
            funcGetShares(coeffs, res);

            activations[bd * total_pixels + hw] = coeffs[0];
        }
    }
}
