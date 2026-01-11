
#pragma once
#include "tools.h"
#include "FCLayer.h"
#include "CNNLayer.h"
#include "MaxpoolLayer.h"
#include "ReLULayer.h"
#include "BNLayer.h"
#include "NeuralNetwork.h"
#include "Functionalities.h"
#include "SigmoidLayer.h"
#include "UpsampleLayer.h"
using namespace std;

extern size_t INPUT_SIZE;
extern size_t LAST_LAYER_SIZE;
extern bool WITH_NORMALIZATION;
extern bool LARGE_NETWORK;

NeuralNetwork::NeuralNetwork(NeuralNetConfig* config)
:inputData(INPUT_SIZE * MINI_BATCH_SIZE),
 outputData(LAST_LAYER_SIZE * MINI_BATCH_SIZE)
{
	for (size_t i = 0; i < NUM_LAYERS; ++i)
	{
		if (config->layerConf[i]->type.compare("FC") == 0) {
			FCConfig *cfg = static_cast<FCConfig *>(config->layerConf[i]);
			layers.push_back(new FCLayer(cfg, i));
		}
		else if (config->layerConf[i]->type.compare("CNN") == 0) {
			CNNConfig *cfg = static_cast<CNNConfig *>(config->layerConf[i]);
			layers.push_back(new CNNLayer(cfg, i));
		}
		else if (config->layerConf[i]->type.compare("Maxpool") == 0) {
			MaxpoolConfig *cfg = static_cast<MaxpoolConfig *>(config->layerConf[i]);
			layers.push_back(new MaxpoolLayer(cfg, i));
		}
		else if (config->layerConf[i]->type.compare("ReLU") == 0) {
			ReLUConfig *cfg = static_cast<ReLUConfig *>(config->layerConf[i]);
			layers.push_back(new ReLULayer(cfg, i));
		}
		else if (config->layerConf[i]->type.compare("BN") == 0) {
			BNConfig *cfg = static_cast<BNConfig *>(config->layerConf[i]);
			layers.push_back(new BNLayer(cfg, i));
		}
		else if (config->layerConf[i]->type.compare("Sigmoid") == 0) {
			SigmoidConfig *cfg = static_cast<SigmoidConfig *>(config->layerConf[i]);
			layers.push_back(new SigmoidLayer(cfg, i));
		}
		else if (config->layerConf[i]->type.compare("Upsample") == 0) {
			UpsampleConfig *cfg = static_cast<UpsampleConfig *>(config->layerConf[i]);
			layers.push_back(new UpsampleLayer(cfg, i));
		}
		else
			error("Only FC, CNN, ReLU, Maxpool, and BN layer types currently supported");
	}
}


NeuralNetwork::~NeuralNetwork()
{
	for (vector<Layer*>::iterator it = layers.begin() ; it != layers.end(); ++it)
		delete (*it);

	layers.clear();
}
extern int iter;

void NeuralNetwork::forward()
{
	log_print("NN.forward");

	layers[0]->forward(inputData);
	if (LARGE_NETWORK)
		cout << "Forward \t" << layers[0]->layerNum << " completed..." << endl;
	
	// cout << "----------------------------------------------" << endl;
	// cout << "DEBUG: forward() at NeuralNetwork.cpp" << endl;
	print_vector(inputData, "FLOAT", "inputData:", 16*3);
	// print_vector(*((BNLayer*)layers[0])->getbeta(), "FLOAT", "w0:", 3);
	// print_vector((*layers[0]->getActivation()), "FLOAT", "1:", 16*12);

	for (size_t i = 1; i < NUM_LAYERS; i++)
	{
		if (i == 29)
    	{
			//print_vector((*layers[i-1]->getActivation()), "FLOAT", "1111"+to_string(i), 10);
			concatenateAndForward(i, 28, 25, 16, 16, 10, 10);
			//print_vector((*layers[i]->getActivation()), "FLOAT", "222"+to_string(i), 10);
    	}
		else if (i == 33)
			concatenateAndForward(i, 32, 21, 16, 16, 20, 20);
		else if (i == 37)
			concatenateAndForward(i, 36, 17, 16, 16, 40, 40);
		else if (i == 41)
			concatenateAndForward(i, 40, 13, 16, 16, 80, 80);
		else if (i == 45)
			concatenateAndForward(i, 44, 9, 16, 16, 160, 160);
		else if (i == 49)
			concatenateAndForward(i, 48, 5, 16, 16, 320, 320);
		else if (i == 52)
			addResidualConnection(i, 51, 2, 64, 320, 320);	//1 end
		else if (i == 78)
			concatenateAndForward(i, 77, 74, 16, 16, 10, 10);
		else if (i == 82)
			concatenateAndForward(i, 81, 70, 16, 16, 20, 20);
		else if (i == 86)
			concatenateAndForward(i, 85, 66, 16, 16, 40, 40);
		else if (i == 90)
			concatenateAndForward(i, 89, 62, 16, 16, 80, 80);
		else if (i == 94)
			concatenateAndForward(i, 93, 58, 16, 16, 160, 160);
		else if (i == 97)
			addResidualConnection(i, 96, 55, 64, 160, 160);   //2 end
		else if (i == 119)
			concatenateAndForward(i, 118, 115, 16, 16, 10, 10);
		else if (i == 123)
			concatenateAndForward(i, 122, 111, 16, 16, 20, 20);
		else if (i == 127)
			concatenateAndForward(i, 126, 107, 16, 16, 40, 40);
		else if (i == 131)
			concatenateAndForward(i, 130, 103, 16, 16, 80, 80);
		else if (i == 134)
			addResidualConnection(i, 133, 100, 64, 80, 80);  //3 end
		else if (i == 152)
			concatenateAndForward(i, 151, 148, 16, 16, 10, 10);
		else if (i == 156)
			concatenateAndForward(i, 155, 144, 16, 16, 20, 20);
		else if (i == 160)
			concatenateAndForward(i, 159, 140, 16, 16, 40, 40);
		else if (i == 163)
			addResidualConnection(i, 162, 137, 64, 40, 40);   //4 end
		else if (i == 179)
			concatenateAndForward(i, 178, 175, 16, 16, 20, 20);
		else if (i == 182)
			concatenateAndForward(i, 181, 172, 16, 16, 20, 20);
		else if (i == 185)
			concatenateAndForward(i, 184, 169, 16, 16, 20, 20);
		else if (i == 188)
			addResidualConnection(i, 187, 166, 64, 20, 20);  //5 end (4f)
		else if (i == 204)
			concatenateAndForward(i, 203, 200, 16, 16, 10, 10);
		else if (i == 207)
			concatenateAndForward(i, 206, 197, 16, 16, 10, 10);
		else if (i == 210)
			concatenateAndForward(i, 209, 194, 16, 16, 10, 10);
		else if (i == 213)
			addResidualConnection(i, 212, 191, 64, 10, 10);	//6 end (4f)
		else if (i == 214)
			concatenateResidualAndForward(i, 213, 187, 166, 64, 64, 20, 20);
			//concatenateAndForward(i, 213, 187, 512, 512, 20, 20);
		else if (i == 229)
			concatenateAndForward(i, 228, 225, 16, 16, 20, 20);
		else if (i == 232)
			concatenateAndForward(i, 231, 222, 16, 16, 20, 20);
		else if (i == 235)
			concatenateAndForward(i, 234, 219, 16, 16, 20, 20);
		else if (i == 238)
			addResidualConnection(i, 237, 216, 64, 20, 20);	//5d end (4f)
		else if (i == 239)
			concatenateResidualAndForward(i, 238, 162, 137, 64, 64, 40, 40);
			//concatenateAndForward(i, 238, 162, 512, 512, 40, 40);
		else if (i == 256)
			concatenateAndForward(i, 255, 252, 16, 16, 10, 10);
		else if (i == 260)
			concatenateAndForward(i, 259, 248, 16, 16, 20, 20);
		else if (i == 264)
			concatenateAndForward(i, 263, 244, 16, 16, 40, 40);
		else if (i == 267)
			addResidualConnection(i, 266, 241, 64, 40, 40);	//4d end
		else if (i == 268)
			concatenateResidualAndForward(i, 267, 133, 100, 64, 64, 80, 80);
			//concatenateAndForward(i, 267, 133, 256, 256, 80, 80);
		else if (i == 289)
			concatenateAndForward(i, 288, 285, 16, 16, 10, 10);
		else if (i == 293)
			concatenateAndForward(i, 292, 281, 16, 16, 20, 20);
		else if (i == 297)
			concatenateAndForward(i, 296, 277, 16, 16, 40, 40);
		else if (i == 301)
			concatenateAndForward(i, 300, 273, 16, 16, 80, 80);
		else if (i == 304)
			addResidualConnection(i, 303, 270, 64, 80, 80);	//3d end
		else if (i == 305)
			addResidualConnection(i, 303, 270, 64, 80, 80);	//side3
		else if (i == 305)
			concatenateResidualAndForward(i, 304, 96, 55, 64, 64, 160, 160);
			//concatenateAndForward(i, 304, 96, 128, 128, 160, 160);
		else if (i == 330)
			concatenateAndForward(i, 329, 326, 16, 16, 10, 10);
		else if (i == 334)
			concatenateAndForward(i, 333, 322, 16, 16, 20, 20);
		else if (i == 338)
			concatenateAndForward(i, 337, 318, 16, 16, 40, 40);
		else if (i == 342)
			concatenateAndForward(i, 341, 314, 16, 16, 80, 80);
		else if (i == 346)
			concatenateAndForward(i, 345, 310, 16, 16, 160, 160);
		// else if (i == 349)
		// 	addResidualConnection(i, 348, 307, 64, 160, 160);	//2d end
		else if (i == 349)
			addResidualConnection(i, 348, 307, 64, 80, 80);   //提前计算side2
		else if (i == 350)
			concatenateResidualAndForward(i, 349, 51, 2, 64, 64, 320, 320);
			//concatenateAndForward(i, 349, 51, 64, 64, 320, 320);
		else if (i == 379)
			concatenateAndForward(i, 378, 375, 16, 16, 10, 10);
		else if (i == 383)
			concatenateAndForward(i, 382, 371, 16, 16, 20, 20);
		else if (i == 387)
			concatenateAndForward(i, 386, 367, 16, 16, 40, 40);
		else if (i == 391)
			concatenateAndForward(i, 390, 363, 16, 16, 80, 80);
		else if (i == 395)
			concatenateAndForward(i, 394, 359, 16, 16, 160, 160);
		else if (i == 399)
			concatenateAndForward(i, 398, 355, 16, 16, 320, 320);  //1d end
		else if (i == 402)
			addResidualConnection(i, 401, 352, 64, 320, 320);	//side1
		else if (i == 403)
			addResidualConnection(i, 348, 307, 64, 160, 160);	//side2
		else if (i == 405)
			addResidualConnection(i, 303, 270, 128, 80, 80);	//side3
		else if (i == 407)
			addResidualConnection(i, 266, 241, 256, 40, 40);	//side4
		else if (i == 409)
			addResidualConnection(i, 237, 216, 512, 20, 20);	//side5
		else if (i == 411)
			addResidualConnection(i, 212, 191, 512, 10, 10);	//side6
		else if (i == 413)
			concatenateAndForward_6(413);	//side0
		else if (i == 414)
			layers[i]->forward(*(layers[413]->getActivation()));
		else if (i == 415)
			layers[i]->forward(*(layers[402]->getActivation()));
		else if (i == 416)
			layers[i]->forward(*(layers[404]->getActivation()));
		else if (i == 417)
			layers[i]->forward(*(layers[406]->getActivation()));
		else if (i == 418)
			layers[i]->forward(*(layers[408]->getActivation()));
		else if (i == 419)
			layers[i]->forward(*(layers[410]->getActivation()));
		else if (i == 420)
			layers[i]->forward(*(layers[412]->getActivation()));
		else
		{	
			layers[i]->forward(*(layers[i-1]->getActivation()));
		}
		if(i==NUM_LAYERS-1||i==NUM_LAYERS-2)
			print_vector((*layers[i]->getActivation()), "FLOAT", std::to_string(i), (*layers[i]->getActivation()).size(), "output1_",true);
			// print_vector((*layers[i]->getActivation()), "FLOAT", "Activation Layer "+to_string(i), 100);
		
		
		
		if (LARGE_NETWORK)
			cout << "Forward \t" << layers[i]->layerNum << " completed..." << endl;
		// print_vector((*layers[i]->getActivation()), "FLOAT", "Activation Layer"+to_string(i), 
		//  			(*layers[i]->getActivation()).size());
		
	}
	
	print_vector((*layers[NUM_LAYERS-1]->getActivation()), "FLOAT", std::to_string(iter), (*layers[NUM_LAYERS-1]->getActivation()).size(), "output_",true);
	// cout << "size of output: " << (*layers[NUM_LAYERS-1]->getActivation()).size() << endl;
	
	printf("done\n\n\n");
}

void NeuralNetwork::concatenateAndForward(size_t current_layer_index, size_t index1, size_t index2, size_t channel_size1, size_t channel_size2, size_t height, size_t width)
{
    
    RSSVectorMyType output1 = (*layers[index1]->getActivation());
    RSSVectorMyType output2 = (*layers[index2]->getActivation());

    size_t concatenated_channel_size = channel_size1 + channel_size2;
	RSSVectorMyType concatenated_output(MINI_BATCH_SIZE * height * width * concatenated_channel_size);
	for (size_t b = 0; b < MINI_BATCH_SIZE; b++) {
		for (size_t c = 0; c < channel_size1; c++) {
			for (size_t h = 0; h < height; h++) {
				for (size_t w = 0; w < width; w++) {
					size_t index = b * height * width * concatenated_channel_size + c * height * width + h * width + w;
					concatenated_output[index] = output1[b * channel_size1 * height * width + c * height * width + h * width + w];
				}
			}
		}
		for (size_t c = 0; c < channel_size2; c++) {
			for (size_t h = 0; h < height; h++) {
				for (size_t w = 0; w < width; w++) {
					size_t index = b * height * width * concatenated_channel_size + (c + channel_size1) * height * width + h * width + w;
					concatenated_output[index] = output2[b * channel_size2 * height * width + c * height * width + h * width + w];
				}
			}
		}
	}

    // 将拼接后的输出传递给后续的网络层
    layers[current_layer_index]->forward(concatenated_output);
}
void NeuralNetwork::concatenateAndForward_6(size_t current_layer_index)
{

    size_t HEIGHT = 320;
    size_t WIDTH = 320;
    size_t ORIGINAL_CHANNEL_SIZE = 1;
    size_t CONCATENATED_CHANNEL_SIZE = 6;

    RSSVectorMyType output1 = (*layers[402]->getActivation());
    RSSVectorMyType output2 = (*layers[403]->getActivation());
    RSSVectorMyType output3 = (*layers[405]->getActivation());
    RSSVectorMyType output4 = (*layers[407]->getActivation());
    RSSVectorMyType output5 = (*layers[409]->getActivation());
    RSSVectorMyType output6 = (*layers[411]->getActivation());

    RSSVectorMyType concatenated_output(MINI_BATCH_SIZE * HEIGHT * WIDTH * CONCATENATED_CHANNEL_SIZE);

    for (size_t b = 0; b < MINI_BATCH_SIZE; b++) {
        for (size_t c = 0; c < ORIGINAL_CHANNEL_SIZE; c++) {
            for (size_t h = 0; h < HEIGHT; h++) {
                for (size_t w = 0; w < WIDTH; w++) {
                    size_t index = b * HEIGHT * WIDTH * CONCATENATED_CHANNEL_SIZE + c * HEIGHT * WIDTH + h * WIDTH + w;
                    concatenated_output[index] = output1[b * ORIGINAL_CHANNEL_SIZE * HEIGHT * WIDTH + c * HEIGHT * WIDTH + h * WIDTH + w];
                }
            }
        }
        for (size_t c = 0; c < ORIGINAL_CHANNEL_SIZE; c++) {
            for (size_t h = 0; h < HEIGHT; h++) {
                for (size_t w = 0; w < WIDTH; w++) {
                    size_t index = b * HEIGHT * WIDTH * CONCATENATED_CHANNEL_SIZE + (c + ORIGINAL_CHANNEL_SIZE) * HEIGHT * WIDTH + h * WIDTH + w;
                    concatenated_output[index] = output2[b * ORIGINAL_CHANNEL_SIZE * HEIGHT * WIDTH + c * HEIGHT * WIDTH + h * WIDTH + w];
                }
            }
        }
        for (size_t c = 0; c < ORIGINAL_CHANNEL_SIZE; c++) {
            for (size_t h = 0; h < HEIGHT; h++) {
                for (size_t w = 0; w < WIDTH; w++) {
                    size_t index = b * HEIGHT * WIDTH * CONCATENATED_CHANNEL_SIZE + (c + 2 * ORIGINAL_CHANNEL_SIZE) * HEIGHT * WIDTH + h * WIDTH + w;
                    concatenated_output[index] = output3[b * ORIGINAL_CHANNEL_SIZE * HEIGHT * WIDTH + c * HEIGHT * WIDTH + h * WIDTH + w];
                }
            }
        }
        for (size_t c = 0; c < ORIGINAL_CHANNEL_SIZE; c++) {
            for (size_t h = 0; h < HEIGHT; h++) {
                for (size_t w = 0; w < WIDTH; w++) {
                    size_t index = b * HEIGHT * WIDTH * CONCATENATED_CHANNEL_SIZE + (c + 3 * ORIGINAL_CHANNEL_SIZE) * HEIGHT * WIDTH + h * WIDTH + w;
                    concatenated_output[index] = output4[b * ORIGINAL_CHANNEL_SIZE * HEIGHT * WIDTH + c * HEIGHT * WIDTH + h * WIDTH + w];
                }
            }
        }
        for (size_t c = 0; c < ORIGINAL_CHANNEL_SIZE; c++) {
            for (size_t h = 0; h < HEIGHT; h++) {
                for (size_t w = 0; w < WIDTH; w++) {
                    size_t index = b * HEIGHT * WIDTH * CONCATENATED_CHANNEL_SIZE + (c + 4 * ORIGINAL_CHANNEL_SIZE) * HEIGHT * WIDTH + h * WIDTH + w;
                    concatenated_output[index] = output5[b * ORIGINAL_CHANNEL_SIZE * HEIGHT * WIDTH + c * HEIGHT * WIDTH + h * WIDTH + w];
                }
            }
        }
        for (size_t c = 0; c < ORIGINAL_CHANNEL_SIZE; c++) {
            for (size_t h = 0; h < HEIGHT; h++) {
                for (size_t w = 0; w < WIDTH; w++) {
                    size_t index = b * HEIGHT * WIDTH * CONCATENATED_CHANNEL_SIZE + (c + 5 * ORIGINAL_CHANNEL_SIZE) * HEIGHT * WIDTH + h * WIDTH + w;
                    concatenated_output[index] = output6[b * ORIGINAL_CHANNEL_SIZE * HEIGHT * WIDTH + c * HEIGHT * WIDTH + h * WIDTH + w];
                }
            }
        }
    }

    // 将拼接后的输出传递给后续的网络层
    layers[current_layer_index]->forward(concatenated_output);
}
void NeuralNetwork::addResidualConnection(size_t current_layer_index, size_t index1, size_t index2, size_t channel_size,  size_t height, size_t width)
{
	RSSVectorMyType output1 = (*layers[index1]->getActivation());
    RSSVectorMyType output2 = (*layers[index2]->getActivation());
	// cout<<output1.size()<<endl;
	// cout<<output2.size()<<endl;
    RSSVectorMyType residual_output(MINI_BATCH_SIZE * height * width *channel_size);

    for (size_t b = 0; b < MINI_BATCH_SIZE; b++) {
        for (size_t c = 0; c < channel_size; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    size_t index = b  * channel_size * height * width + c * height * width + h * width + w;
                	residual_output[index] = output1[index] + output2[index];
                }
            }
        }
    }

    // 将相加后的结果传递给后续网络层
    layers[current_layer_index]->forward(residual_output);
}
void NeuralNetwork::concatenateResidualAndForward(size_t current_layer_index, size_t index1, size_t index2, size_t index3, size_t channel_size1, size_t channel_size2, size_t height, size_t width)
{
    // 从三个不同的网络层中获取激活输出
    RSSVectorMyType output1 = (*layers[index1]->getActivation());
    RSSVectorMyType output2 = (*layers[index2]->getActivation());
    RSSVectorMyType output3 = (*layers[index3]->getActivation());

    // 计算残差输出
    RSSVectorMyType residual_output(MINI_BATCH_SIZE * height * width * channel_size2);
    for (size_t b = 0; b < MINI_BATCH_SIZE; b++) {
        for (size_t c = 0; c < channel_size2; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    size_t index = b * channel_size2 * height * width + c * height * width + h * width + w;
                    residual_output[index] = output2[index] + output3[index];
                }
            }
        }
    }

    // 拼接output1和残差输出
    size_t concatenated_channel_size = channel_size1 + channel_size2;
    RSSVectorMyType concatenated_output(MINI_BATCH_SIZE * height * width * concatenated_channel_size);
    for (size_t b = 0; b < MINI_BATCH_SIZE; b++) {
        for (size_t c = 0; c < channel_size1; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    size_t index = b * height * width * concatenated_channel_size + c * height * width + h * width + w;
                    concatenated_output[index] = output1[b * channel_size1 * height * width + c * height * width + h * width + w];
                }
            }
        }
        for (size_t c = 0; c < channel_size2; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    size_t index = b * height * width * concatenated_channel_size + (c + channel_size1) * height * width + h * width + w;
                    concatenated_output[index] = residual_output[b * channel_size2 * height * width + c * height * width + h * width + w];
                }
            }
        }
    }

    // 将拼接后的输出传递给后续的网络层
    layers[current_layer_index]->forward(concatenated_output);
}

void NeuralNetwork::backward()
{
	log_print("NN.backward");
	computeDelta();	
	updateEquations();
}

void NeuralNetwork::computeDelta()
{
	log_print("NN.computeDelta");
	
	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	size_t size = rows*columns;
	size_t index;

	if (WITH_NORMALIZATION)
	{
		RSSVectorMyType rowSum(size, make_pair(0,0));
		RSSVectorMyType quotient(size, make_pair(0,0));

		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
				rowSum[i*columns] = rowSum[i*columns] + 
									(*(layers[NUM_LAYERS-1]->getActivation()))[i * columns + j];

		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
				rowSum[i*columns + j] = rowSum[i*columns];

		funcDivision(*(layers[NUM_LAYERS-1]->getActivation()), rowSum, quotient, size);

		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
			{
				index = i * columns + j;
				(*(layers[NUM_LAYERS-1]->getDelta()))[index] = quotient[index] - outputData[index];
			}
	}
	else
	{
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
			{
				index = i * columns + j;
				(*(layers[NUM_LAYERS-1]->getDelta()))[index] = 
				(*(layers[NUM_LAYERS-1]->getActivation()))[index] - outputData[index];
			}
	}

	if (LARGE_NETWORK)		
		cout << "Delta last layer completed." << endl;

	for (size_t i = NUM_LAYERS-1; i > 0; --i)
	{
		layers[i]->computeDelta(*(layers[i-1]->getDelta()));
		if (LARGE_NETWORK)
			cout << "Delta \t\t" << layers[i]->layerNum << " completed..." << endl;
	}
}

void NeuralNetwork::updateEquations()
{
	log_print("NN.updateEquations");

	for (size_t i = NUM_LAYERS-1; i > 0; --i)
	{
		layers[i]->updateEquations(*(layers[i-1]->getActivation()));	
		if (LARGE_NETWORK)
			cout << "Update Eq. \t" << layers[i]->layerNum << " completed..." << endl;	
	}

	layers[0]->updateEquations(inputData);
	if (LARGE_NETWORK)
		cout << "First layer update Eq. completed." << endl;		
}

void NeuralNetwork::predict(RSSVectorMyType &maxIndex)
{
	log_print("NN.predict");

	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	RSSVectorMyType max(rows);
	RSSVectorSmallType maxPrime(rows*columns);

	//funcMaxpool(*(layers[NUM_LAYERS-1]->getActivation()), max, maxPrime, rows, columns);
}

/* new implementation, may still have bug and security flaws */
void NeuralNetwork::getAccuracy(const RSSVectorMyType &maxIndex, vector<size_t> &counter)
{
	log_print("NN.getAccuracy");

	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	
	RSSVectorMyType max(rows);
	RSSVectorSmallType maxPrime(rows*columns);
	RSSVectorMyType temp_max(rows), temp_groundTruth(rows);
	RSSVectorSmallType temp_maxPrime(rows*columns);
	
	vector<myType> groundTruth(rows*columns);
	vector<smallType> prediction(rows*columns);
	
	// reconstruct ground truth from output data
	funcReconstruct(outputData, groundTruth, rows*columns, "groundTruth", false);
	// print_vector(outputData, "FLOAT", "outputData:", rows*columns);
	
	// reconstruct prediction from neural network
	//funcMaxpool((*(layers[NUM_LAYERS-1])->getActivation()), temp_max, temp_maxPrime, rows, columns);
	funcReconstructBit(temp_maxPrime, prediction, rows*columns, "prediction", false);
	
	for (int i = 0, index = 0; i < rows; ++i){
		counter[1]++;
		for (int j = 0; j < columns; j++){
			index = i * columns + j;
			if ((int) groundTruth[index] * (int) prediction[index] || 
				(!(int) groundTruth[index] && !(int) prediction[index])){
				if (j == columns - 1){
					counter[0]++;
				}
			} else {
				break;
			}
		}
	}

	cout << "Rolling accuracy: " << counter[0] << " out of " 
		 << counter[1] << " (" << (counter[0]*100/counter[1]) << " %)" << endl;
}

// original implmentation of NeuralNetwork::getAccuracy(.)
/* void NeuralNetwork::getAccuracy(const RSSVectorMyType &maxIndex, vector<size_t> &counter)
{
	log_print("NN.getAccuracy");

	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	RSSVectorMyType max(rows);
	RSSVectorSmallType maxPrime(rows*columns);

	//Needed maxIndex here
	funcMaxpool(outputData, max, maxPrime, rows, columns);

	//Reconstruct things
	RSSVectorMyType temp_max(rows), temp_groundTruth(rows);
	// if (partyNum == PARTY_B)
	// 	sendTwoVectors<RSSMyType>(max, groundTruth, PARTY_A, rows, rows);

	// if (partyNum == PARTY_A)
	// {
	// 	receiveTwoVectors<RSSMyType>(temp_max, temp_groundTruth, PARTY_B, rows, rows);
	// 	addVectors<RSSMyType>(temp_max, max, temp_max, rows);
//		dividePlain(temp_max, (1 << FLOAT_PRECISION));
	// 	addVectors<RSSMyType>(temp_groundTruth, groundTruth, temp_groundTruth, rows);	
	// }

	for (size_t i = 0; i < MINI_BATCH_SIZE; ++i)
	{
		counter[1]++;
		if (temp_max[i] == temp_groundTruth[i])
			counter[0]++;
	}		

	cout << "Rolling accuracy: " << counter[0] << " out of " 
		 << counter[1] << " (" << (counter[0]*100/counter[1]) << " %)" << endl;
} */

/*
                   _ooOoo_
                  o8888888o
                  88" . "88
                  (| -_- |)
                  O\  =  /O
               ____/`---'\____
            .'  \\|     |//  `.
            /  \\|||  :  |||//  \
           /  _||||| -:- |||||-  \
           |   | \\\  -  /// |   |
           | \_|  ''\---/''  |   |
           \  .-\__  `-`  ___/-. /
         ___`. .'  /--.--\  `. . __
      ."" '<  `.___\_<|>_/___.'  >'"".
     | | :  `- \`.;`\ _ /`;.`/ - ` : | |
     \  \ `-.   \_ __\ /__ _/   .-` /  /
======`-.____`-.___\_____/___.-`____.-'======
                   `=---='
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    佛祖保佑       永不宕机     永无BUG
*/
