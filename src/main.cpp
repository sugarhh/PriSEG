#include <iostream>
#include <string>
#include <chrono>
#pragma GCC optimize(2)
#include "AESObject.h"
#include "Precompute.h"
#include "secondary.h"
#include "connect.h"
#include "NeuralNetConfig.h"
#include "NeuralNetwork.h"
#include "unitTests.h"


int partyNum;
AESObject* aes_indep;
AESObject* aes_next;
AESObject* aes_prev;
Precompute PrecomputeObject;

int iter;

int main(int argc, char** argv)
{
/****************************** PREPROCESSING ******************************/ 
	parseInputs(argc, argv);	//解析命令行
	NeuralNetConfig* config = new NeuralNetConfig(NUM_ITERATIONS); 
	string network, dataset, security;
	bool PRELOADING = false;	//预加载功能

/****************************** SELECT NETWORK ******************************/ 
	//Network {SecureML, Sarda, MiniONN, LeNet, AlexNet, and VGG16}  ConvNet
	//Dataset {MNIST, CIFAR10, and ImageNet}
	//Security {Semi-honest or Malicious}
	if (argc == 9)
	{network = argv[6]; dataset = argv[7]; security = argv[8];}
	else
	{
		// network = "U2netNet";
		// dataset = "ImageNet";
		network = "LeNet";
		dataset = "MNIST";
		security = "Semi-honest";
		iter = atoi(argv[6]);
		cout<<iter<<endl;
	}
	selectNetwork(network, dataset, security, config);
	config->checkNetwork();
	//检查神经网络配置：最后一层必须是全连接层、最后一层的输出维度是否与预定义的LAST_LAYER_SIZE相匹配、检查numIterations变量是否与预定义的NUM_ITERATIONS值相等
	//如果第一层是全连接层，这个断言用于检查第一层的输入维度是否与预定义的INPUT_SIZE相匹配。
	//如果第一层是卷积层，这个断言用于检查第一层的输入维度是否与预定义的INPUT_SIZE相匹配。
	NeuralNetwork* net = new NeuralNetwork(config);
	
/****************************** AES SETUP and SYNC ******************************/ 
//创建和初始化AES对象，初始化通信配置，并进行同步操作，以确保各个对象在一定时间内达到同步状态。
	aes_indep = new AESObject(argv[3]);
	aes_next = new AESObject(argv[4]);
	aes_prev = new AESObject(argv[5]);

	initializeCommunication(argv[2], partyNum);
	synchronize(2000000);

/****************************** RUN NETWORK/UNIT TESTS ******************************/ 
	

	//Run these if you want a preloaded network to be tested
	//测试预加载的网络
	//assert(NUM_ITERATION == 1 and "check if readMiniBatch is false in test(net)")
	//First argument {SecureML, Sarda, MiniONN, or LeNet}
	network += " preloaded"; PRELOADING = true;
	preload_network(PRELOADING, network, net);
	auto start_time = std::chrono::high_resolution_clock::now();
	start_m();
	//Run unit tests in two modes: 
	//以两种模式运行单元测试
	//	1. Debug {Mat-Mul, DotProd, PC, Wrap, ReLUPrime, ReLU, Division, BN, SSBits, SS, and Maxpool}
	//	2. Test {Mat-Mul1, Mat-Mul2, Mat-Mul3 (and similarly) Conv*, ReLU*, ReLUPrime*, and Maxpool*} where * = {1,2,3}
	// runTest("Debug", "BN", network);
	// runTest("Test", "ReLUPrime1", network);

	// Run forward/backward for single layers
	//  1. what {F, D, U}
	// 	2. l {0,1,....NUM_LAYERS-1}
	// size_t l = 0;
	// string what = "F";
	// runOnly(net, l, what, network);

	//Run training
	// network += " train";
	// train(net);

	//Run inference (possibly with preloading a network)
	cout << "----------------------------------------------" << endl;
	network += " test";
	test(PRELOADING, network, net);

	end_m(network);

	auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

	cout << "----------------------------------------------" << endl;  	
	cout << "Run details: " << NUM_OF_PARTIES << "PC (P" << partyNum 
		 << "), " << NUM_ITERATIONS << " iterations, batch size " << MINI_BATCH_SIZE << endl 
		 << "Running " << security << " " << network << " on " << dataset << " dataset" << endl;
	cout << "----------------------------------------------" << endl << endl;  
	cout << "Total runtime: " << duration << " ms" << endl;
    cout << "----------------------------------------------" << endl << endl;  
	printNetwork(net);

/****************************** CLEAN-UP ******************************/ 
	delete aes_indep;
	delete aes_next;
	delete aes_prev;
	delete config;
	delete net;
	deleteObjects();

	return 0;
}




