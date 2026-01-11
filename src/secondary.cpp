
#include "connect.h" 
#include "secondary.h"

extern CommunicationObject commObject;
extern int partyNum;
extern string * addrs;
extern BmrNet ** communicationSenders;
extern BmrNet ** communicationReceivers;
extern void log_print(string str);
#define NANOSECONDS_PER_SEC 1E9

//For time measurements
clock_t tStart;
struct timespec requestStart, requestEnd;
bool alreadyMeasuringTime = false;
int roundComplexitySend = 0;
int roundComplexityRecv = 0;
bool alreadyMeasuringRounds = false;

//For faster modular operations
extern smallType additionModPrime[PRIME_NUMBER][PRIME_NUMBER];
extern smallType subtractModPrime[PRIME_NUMBER][PRIME_NUMBER];
extern smallType multiplicationModPrime[PRIME_NUMBER][PRIME_NUMBER];

RSSVectorMyType trainData, testData;
RSSVectorMyType trainLabels, testLabels;
size_t trainDataBatchCounter = 0;
size_t trainLabelsBatchCounter = 0;
size_t testDataBatchCounter = 0;
size_t testLabelsBatchCounter = 0;

size_t INPUT_SIZE;
size_t LAST_LAYER_SIZE;
size_t NUM_LAYERS;
bool WITH_NORMALIZATION;
bool LARGE_NETWORK;
size_t TRAINING_DATA_SIZE;
size_t TEST_DATA_SIZE;
string SECURITY_TYPE;

extern void print_linear(myType var, string type);
extern void funcReconstruct(const RSSVectorMyType &a, vector<myType> &b, size_t size, string str, bool print);

extern int iter;

/******************* Main train and test functions *******************/
void parseInputs(int argc, char* argv[])
{	
	if (argc < 6) 
		print_usage(argv[0]);

	partyNum = atoi(argv[1]);

	for (int i = 0; i < PRIME_NUMBER; ++i)
		for (int j = 0; j < PRIME_NUMBER; ++j)
		{
			additionModPrime[i][j] = ((i + j) % PRIME_NUMBER);
			subtractModPrime[i][j] = ((PRIME_NUMBER + i - j) % PRIME_NUMBER);
			multiplicationModPrime[i][j] = ((i * j) % PRIME_NUMBER); //How come you give the right answer multiplying in 8-bits??
		}
}

void train(NeuralNetwork* net)
{
	log_print("train");

	for (int i = 0; i < NUM_ITERATIONS; ++i)
	{
		// cout << "----------------------------------" << endl;  
		// cout << "Iteration " << i << endl;
		readMiniBatch(net, "TRAINING");
		net->forward();
		net->backward();
		// cout << "----------------------------------" << endl;  
	}
}


extern void print_vector(RSSVectorMyType &var, string type, string pre_text, int print_nos);
extern string which_network(string network);
void test(bool PRELOADING, string network, NeuralNetwork* net)
{
	log_print("test");

	//counter[0]: Correct samples, counter[1]: total samples
	vector<size_t> counter(2,0);
	RSSVectorMyType maxIndex(MINI_BATCH_SIZE);

	for (int i = 0; i < NUM_ITERATIONS; ++i)
	{
		if (!PRELOADING)
			readMiniBatch(net, "TESTING");

		net->forward();
		// net->predict(maxIndex);
		// net->getAccuracy(maxIndex, counter);
	}
	print_vector((*(net->layers[NUM_LAYERS-1])->getActivation()), "FLOAT", "MPC Output over uint32_t:", 3);

	// Write output to file
	// if (PRELOADING)
	// {
	// 	ofstream data_file;
	// 	data_file.open("files/preload/"+which_network(network)+"/"+which_network(network)+".txt");
		
	// 	vector<myType> b(MINI_BATCH_SIZE * LAST_LAYER_SIZE);
	// 	funcReconstruct((*(net->layers[NUM_LAYERS-1])->getActivation()), b, MINI_BATCH_SIZE * LAST_LAYER_SIZE, "anything", false);
	// 	for (int i = 0; i < MINI_BATCH_SIZE; ++i)
	// 	{
	// 		for (int j = 0; j < LAST_LAYER_SIZE; ++j)
	// 			data_file << b[i*(LAST_LAYER_SIZE) + j] << " ";
	// 		data_file << endl;
	// 	}
	// }
}


// Generate a file with 0's of appropriate size
void generate_zeros(string name, size_t number, string network)
{
	string default_path = "files/preload/"+which_network(network)+"/";
	ofstream data_file;
	data_file.open(default_path+name);

	for (int i = 0; i < number; ++i)
		data_file << (int)0 << " ";
}


extern size_t nextParty(size_t party);
#include "FCLayer.h"
#include "CNNLayer.h"
#include "BNLayer.h"
#include "SigmoidLayer.h"
#include "UpsampleLayer.h"
#include "MaxpoolLayer.h"


void preload_network(bool PRELOADING, string network, NeuralNetwork* net)
{
	log_print("preload_network");
	//assert((PRELOADING) and (NUM_ITERATIONS == 1) and (MINI_BATCH_SIZE == 128) && "Preloading conditions fail");

	float temp_next = 0, temp_prev = 0;
	string default_path = "files/preload/"+which_network(network)+"/all_input/";
	string canshu_path = "files/preload/"+which_network(network)+"/all_canshu/";
	//Set to true if you want the zeros files generated.
	bool ZEROS = false;

	if (which_network(network).compare("SecureML") == 0)
	{
		string temp = "SecureML";
		/************************** Input **********************************/
		string path_input_1 = default_path+"input_"+to_string(partyNum);
		string path_input_2 = default_path+"input_"+to_string(nextParty(partyNum));
		ifstream f_input_1(path_input_1), f_input_2(path_input_2);

		for (int i = 0; i < INPUT_SIZE * MINI_BATCH_SIZE; ++i)
		{
			f_input_1 >> temp_next; f_input_2 >> temp_prev;
			net->inputData[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_input_1.close(); f_input_2.close();
		if (ZEROS)
		{
			generate_zeros("input_1", 784*128, temp);
			generate_zeros("input_2", 784*128, temp);
		}

		// print_vector(net->inputData, "FLOAT", "inputData:", 784);

		/************************** Weight1 **********************************/
		string path_weight1_1 = default_path+"weight1_"+to_string(partyNum);
		string path_weight1_2 = default_path+"weight1_"+to_string(nextParty(partyNum));
		ifstream f_weight1_1(path_weight1_1), f_weight1_2(path_weight1_2);

		for (int column = 0; column < 128; ++column)
		{
			for (int row = 0; row < 784; ++row)
			{
				f_weight1_1 >> temp_next; f_weight1_2 >> temp_prev;
				(*((FCLayer*)net->layers[0])->getWeights())[128*row + column] = 
						std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
			}
		}
		f_weight1_1.close(); f_weight1_2.close();
		if (ZEROS)
		{
			generate_zeros("weight1_1", 784*128, temp);
			generate_zeros("weight1_2", 784*128, temp);
		}

		/************************** Weight2 **********************************/
		string path_weight2_1 = default_path+"weight2_"+to_string(partyNum);
		string path_weight2_2 = default_path+"weight2_"+to_string(nextParty(partyNum));
		ifstream f_weight2_1(path_weight2_1), f_weight2_2(path_weight2_2);

		for (int column = 0; column < 128; ++column)
		{
			for (int row = 0; row < 128; ++row)
			{
				f_weight2_1 >> temp_next; f_weight2_2 >> temp_prev;
				(*((FCLayer*)net->layers[2])->getWeights())[128*row + column] = 
						std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
			}
		}
		f_weight2_1.close(); f_weight2_2.close();
		if (ZEROS)
		{
			generate_zeros("weight2_1", 128*128, temp);
			generate_zeros("weight2_2", 128*128, temp);
		}

		/************************** Weight3 **********************************/
		string path_weight3_1 = default_path+"weight3_"+to_string(partyNum);
		string path_weight3_2 = default_path+"weight3_"+to_string(nextParty(partyNum));
		ifstream f_weight3_1(path_weight3_1), f_weight3_2(path_weight3_2);

		for (int column = 0; column < 10; ++column)
		{
			for (int row = 0; row < 128; ++row)
			{
				f_weight3_1 >> temp_next; f_weight3_2 >> temp_prev;
				(*((FCLayer*)net->layers[4])->getWeights())[10*row + column] = 
						std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
			}
		}
		f_weight3_1.close(); f_weight3_2.close();
		if (ZEROS)
		{
			generate_zeros("weight3_1", 128*10, temp);
			generate_zeros("weight3_2", 128*10, temp);
		}


		/************************** Bias1 **********************************/
		string path_bias1_1 = default_path+"bias1_"+to_string(partyNum);
		string path_bias1_2 = default_path+"bias1_"+to_string(nextParty(partyNum));
		ifstream f_bias1_1(path_bias1_1), f_bias1_2(path_bias1_2);

		for (int i = 0; i < 128; ++i)
		{
			f_bias1_1 >> temp_next; f_bias1_2 >> temp_prev;
			(*((FCLayer*)net->layers[0])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias1_1.close(); f_bias1_2.close();
		if (ZEROS)
		{
			generate_zeros("bias1_1", 128, temp);
			generate_zeros("bias1_2", 128, temp);
		}


		/************************** Bias2 **********************************/
		string path_bias2_1 = default_path+"bias2_"+to_string(partyNum);
		string path_bias2_2 = default_path+"bias2_"+to_string(nextParty(partyNum));
		ifstream f_bias2_1(path_bias2_1), f_bias2_2(path_bias2_2);

		for (int i = 0; i < 128; ++i)
		{
			f_bias2_1 >> temp_next; f_bias2_2 >> temp_prev;
			(*((FCLayer*)net->layers[2])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias2_1.close(); f_bias2_2.close();
		if (ZEROS)
		{
			generate_zeros("bias2_1", 128, temp);
			generate_zeros("bias2_2", 128, temp);
		}


		/************************** Bias3 **********************************/
		string path_bias3_1 = default_path+"bias3_"+to_string(partyNum);
		string path_bias3_2 = default_path+"bias3_"+to_string(nextParty(partyNum));
		ifstream f_bias3_1(path_bias3_1), f_bias3_2(path_bias3_2);

		for (int i = 0; i < 10; ++i)
		{
			f_bias3_1 >> temp_next; f_bias3_2 >> temp_prev;
			(*((FCLayer*)net->layers[4])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias3_1.close(); f_bias3_2.close();
		if (ZEROS)
		{
			generate_zeros("bias3_1", 10, temp);
			generate_zeros("bias3_2", 10, temp);
		}
	}
	else if (which_network(network).compare("Sarda") == 0)
	{
		string temp = "Sarda";
		/************************** Input **********************************/
		string path_input_1 = default_path+"input_"+to_string(partyNum);
		string path_input_2 = default_path+"input_"+to_string(nextParty(partyNum));
		ifstream f_input_1(path_input_1), f_input_2(path_input_2);

		for (int i = 0; i < INPUT_SIZE * MINI_BATCH_SIZE; ++i)
		{
			f_input_1 >> temp_next; f_input_2 >> temp_prev;
			net->inputData[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_input_1.close(); f_input_2.close();
		if (ZEROS)
		{
			generate_zeros("input_1", 784*128, temp);
			generate_zeros("input_2", 784*128, temp);
		}

		// print_vector(net->inputData, "FLOAT", "inputData:", 784);

		/************************** Weight1 **********************************/
		string path_weight1_1 = default_path+"weight1_"+to_string(partyNum);
		string path_weight1_2 = default_path+"weight1_"+to_string(nextParty(partyNum));
		ifstream f_weight1_1(path_weight1_1), f_weight1_2(path_weight1_2);

		for (int column = 0; column < 5; ++column)
		{
			for (int row = 0; row < 4; ++row)
			{
				f_weight1_1 >> temp_next; f_weight1_2 >> temp_prev;
				(*((CNNLayer*)net->layers[0])->getWeights())[4*column + row] = 
						std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
			}
		}
		f_weight1_1.close(); f_weight1_2.close();
		if (ZEROS)
		{
			generate_zeros("weight1_1", 2*2*1*5, temp);
			generate_zeros("weight1_2", 2*2*1*5, temp);
		}

		/************************** Weight2 **********************************/
		string path_weight2_1 = default_path+"weight2_"+to_string(partyNum);
		string path_weight2_2 = default_path+"weight2_"+to_string(nextParty(partyNum));
		ifstream f_weight2_1(path_weight2_1), f_weight2_2(path_weight2_2);

		for (int column = 0; column < 100; ++column)
		{
			for (int row = 0; row < 980; ++row)
			{
				f_weight2_1 >> temp_next; f_weight2_2 >> temp_prev;
				(*((FCLayer*)net->layers[2])->getWeights())[100*row + column] = 
						std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
			}
		}
		f_weight2_1.close(); f_weight2_2.close();
		if (ZEROS)
		{
			generate_zeros("weight2_1", 980*100, temp);
			generate_zeros("weight2_2", 980*100, temp);
		}


		/************************** Weight3 **********************************/
		string path_weight3_1 = default_path+"weight3_"+to_string(partyNum);
		string path_weight3_2 = default_path+"weight3_"+to_string(nextParty(partyNum));
		ifstream f_weight3_1(path_weight3_1), f_weight3_2(path_weight3_2);

		for (int column = 0; column < 10; ++column)
		{
			for (int row = 0; row < 100; ++row)
			{
				f_weight3_1 >> temp_next; f_weight3_2 >> temp_prev;
				(*((FCLayer*)net->layers[4])->getWeights())[10*row + column] = 
						std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
			}
		}
		f_weight3_1.close(); f_weight3_2.close();
		if (ZEROS)
		{
			generate_zeros("weight3_1", 100*10, temp);
			generate_zeros("weight3_2", 100*10, temp);
		}

		/************************** Bias1 **********************************/
		string path_bias1_1 = default_path+"bias1_"+to_string(partyNum);
		string path_bias1_2 = default_path+"bias1_"+to_string(nextParty(partyNum));
		ifstream f_bias1_1(path_bias1_1), f_bias1_2(path_bias1_2);

		for (int i = 0; i < 5; ++i)
		{
			f_bias1_1 >> temp_next; f_bias1_2 >> temp_prev;
			(*((CNNLayer*)net->layers[0])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias1_1.close(); f_bias1_2.close();
		if (ZEROS)
		{
			generate_zeros("bias1_1", 5, temp);
			generate_zeros("bias1_2", 5, temp);
		}

		/************************** Bias2 **********************************/
		string path_bias2_1 = default_path+"bias2_"+to_string(partyNum);
		string path_bias2_2 = default_path+"bias2_"+to_string(nextParty(partyNum));
		ifstream f_bias2_1(path_bias2_1), f_bias2_2(path_bias2_2);

		for (int i = 0; i < 100; ++i)
		{
			f_bias2_1 >> temp_next; f_bias2_2 >> temp_prev;
			(*((FCLayer*)net->layers[2])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias2_1.close(); f_bias2_2.close();
		if (ZEROS)
		{
			generate_zeros("bias2_1", 100, temp);
			generate_zeros("bias2_2", 100, temp);
		}

		/************************** Bias3 **********************************/
		string path_bias3_1 = default_path+"bias3_"+to_string(partyNum);
		string path_bias3_2 = default_path+"bias3_"+to_string(nextParty(partyNum));
		ifstream f_bias3_1(path_bias3_1), f_bias3_2(path_bias3_2);

		for (int i = 0; i < 10; ++i)
		{
			f_bias3_1 >> temp_next; f_bias3_2 >> temp_prev;
			(*((FCLayer*)net->layers[4])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias3_1.close(); f_bias3_2.close();
		if (ZEROS)
		{
			generate_zeros("bias3_1", 10, temp);
			generate_zeros("bias3_2", 10, temp);
		}
	}
	else if (which_network(network).compare("MiniONN") == 0)
	{
		string temp = "MiniONN";
		/************************** Input **********************************/
		string path_input_1 = default_path+"input_"+to_string(partyNum);
		string path_input_2 = default_path+"input_"+to_string(nextParty(partyNum));
		ifstream f_input_1(path_input_1), f_input_2(path_input_2);

		for (int i = 0; i < INPUT_SIZE * MINI_BATCH_SIZE; ++i)
		{
			f_input_1 >> temp_next; f_input_2 >> temp_prev;
			net->inputData[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_input_1.close(); f_input_2.close();
		if (ZEROS)
		{
			generate_zeros("input_1", 784*128, temp);
			generate_zeros("input_2", 784*128, temp);
		}

		// print_vector(net->inputData, "FLOAT", "inputData:", 784);

		/************************** Weight1 **********************************/
		string path_weight1_1 = default_path+"weight1_"+to_string(partyNum);
		string path_weight1_2 = default_path+"weight1_"+to_string(nextParty(partyNum));
		ifstream f_weight1_1(path_weight1_1), f_weight1_2(path_weight1_2);

		for (int row = 0; row < 5*5*1*16; ++row)
		{
			f_weight1_1 >> temp_next; f_weight1_2 >> temp_prev;
			(*((CNNLayer*)net->layers[0])->getWeights())[row] = 
					std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_weight1_1.close(); f_weight1_2.close();
		if (ZEROS)
		{
			generate_zeros("weight1_1", 5*5*1*16, temp);
			generate_zeros("weight1_2", 5*5*1*16, temp);
		}

		/************************** Weight2 **********************************/
		string path_weight2_1 = default_path+"weight2_"+to_string(partyNum);
		string path_weight2_2 = default_path+"weight2_"+to_string(nextParty(partyNum));
		ifstream f_weight2_1(path_weight2_1), f_weight2_2(path_weight2_2);


		for (int row = 0; row < 25*16*16; ++row)
		{
			f_weight2_1 >> temp_next; f_weight2_2 >> temp_prev;
			(*((CNNLayer*)net->layers[3])->getWeights())[row] = 
					std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_weight2_1.close(); f_weight2_2.close();
		if (ZEROS)
		{
			generate_zeros("weight2_1", 5*5*16*16, temp);
			generate_zeros("weight2_2", 5*5*16*16, temp);
		}

		/************************** Weight3 **********************************/
		string path_weight3_1 = default_path+"weight3_"+to_string(partyNum);
		string path_weight3_2 = default_path+"weight3_"+to_string(nextParty(partyNum));
		ifstream f_weight3_1(path_weight3_1), f_weight3_2(path_weight3_2);

		for (int column = 0; column < 100; ++column)
		{
			for (int row = 0; row < 256; ++row)
			{
				f_weight3_1 >> temp_next; f_weight3_2 >> temp_prev;
				(*((FCLayer*)net->layers[6])->getWeights())[100*row + column] = 
						std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
			}
		}
		f_weight3_1.close(); f_weight3_2.close();
		if (ZEROS)
		{
			generate_zeros("weight3_1", 256*100, temp);
			generate_zeros("weight3_2", 256*100, temp);
		}


		/************************** Weight4 **********************************/
		string path_weight4_1 = default_path+"weight4_"+to_string(partyNum);
		string path_weight4_2 = default_path+"weight4_"+to_string(nextParty(partyNum));
		ifstream f_weight4_1(path_weight4_1), f_weight4_2(path_weight4_2);

		for (int column = 0; column < 10; ++column)
		{
			for (int row = 0; row < 100; ++row)
			{
				f_weight4_1 >> temp_next; f_weight4_2 >> temp_prev;
				(*((FCLayer*)net->layers[8])->getWeights())[10*row + column] = 
						std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
			}
		}
		f_weight4_1.close(); f_weight4_2.close();
		if (ZEROS)
		{
			generate_zeros("weight4_1", 100*10, temp);
			generate_zeros("weight4_2", 100*10, temp);
		}

		/************************** Bias1 **********************************/
		string path_bias1_1 = default_path+"bias1_"+to_string(partyNum);
		string path_bias1_2 = default_path+"bias1_"+to_string(nextParty(partyNum));
		ifstream f_bias1_1(path_bias1_1), f_bias1_2(path_bias1_2);

		for (int i = 0; i < 16; ++i)
		{
			f_bias1_1 >> temp_next; f_bias1_2 >> temp_prev;
			(*((CNNLayer*)net->layers[0])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias1_1.close(); f_bias1_2.close();
		if (ZEROS)
		{
			generate_zeros("bias1_1", 16, temp);
			generate_zeros("bias1_2", 16, temp);
		}

		/************************** Bias2 **********************************/
		string path_bias2_1 = default_path+"bias2_"+to_string(partyNum);
		string path_bias2_2 = default_path+"bias2_"+to_string(nextParty(partyNum));
		ifstream f_bias2_1(path_bias2_1), f_bias2_2(path_bias2_2);

		for (int i = 0; i < 16; ++i)
		{
			f_bias2_1 >> temp_next; f_bias2_2 >> temp_prev;
			(*((CNNLayer*)net->layers[3])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias2_1.close(); f_bias2_2.close();
		if (ZEROS)
		{
			generate_zeros("bias2_1", 16, temp);
			generate_zeros("bias2_2", 16, temp);
		}

		/************************** Bias3 **********************************/
		string path_bias3_1 = default_path+"bias3_"+to_string(partyNum);
		string path_bias3_2 = default_path+"bias3_"+to_string(nextParty(partyNum));
		ifstream f_bias3_1(path_bias3_1), f_bias3_2(path_bias3_2);

		for (int i = 0; i < 100; ++i)
		{
			f_bias3_1 >> temp_next; f_bias3_2 >> temp_prev;
			(*((FCLayer*)net->layers[6])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias3_1.close(); f_bias3_2.close();
		if (ZEROS)
		{
			generate_zeros("bias3_1", 100, temp);
			generate_zeros("bias3_2", 100, temp);
		}

		/************************** Bias4 **********************************/
		string path_bias4_1 = default_path+"bias4_"+to_string(partyNum);
		string path_bias4_2 = default_path+"bias4_"+to_string(nextParty(partyNum));
		ifstream f_bias4_1(path_bias4_1), f_bias4_2(path_bias4_2);

		for (int i = 0; i < 10; ++i)
		{
			f_bias4_1 >> temp_next; f_bias4_2 >> temp_prev;
			(*((FCLayer*)net->layers[8])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias4_1.close(); f_bias4_2.close();
		if (ZEROS)
		{
			generate_zeros("bias4_1", 10, temp);
			generate_zeros("bias4_2", 10, temp);
		}
	}
	else if (which_network(network).compare("LeNet") == 0)
	{
		string temp = "LeNet";
		/************************** Input **********************************/
		string path_input_1 = default_path+"input_t_"+to_string(partyNum);
		string path_input_2 = default_path+"input_t_"+to_string(nextParty(partyNum));
		ifstream f_input_1(path_input_1), f_input_2(path_input_2);

		for (int i = 0; i <  320 * 320; ++i)	//测试
		{
			f_input_1 >> temp_next; f_input_2 >> temp_prev;
			net->inputData[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
			// cout<<net->inputData[i].first<<" "<<net->inputData[i].second<<endl;
		}
		f_input_1.close(); f_input_2.close();
		// if (ZEROS)
		// {
		// 	generate_zeros("input_1", 784*128, temp);
		// 	generate_zeros("input_2", 784*128, temp);
		// }
		// loadCNNWeightParams("stage1.rebnconvin", 0, 3*64*9, net);
		// loadCNNBiasParams("stage1.rebnconvin", 0, 64, net);
		// loadBNGammaParams("stage1.rebnconvin", 1, 64, net); 
		// loadBNBetaParams("stage1.rebnconvin", 1, 64, net); 
		// loadBNvarParams("stage1.rebnconvin", 1, 64, net); 
		// loadBNmeanParams("stage1.rebnconvin", 1, 64, net); 

		// loadCNNWeightParams("stage1.rebnconv1", 3, 64*32*9, net);
		// loadCNNBiasParams("stage1.rebnconv1", 3, 32, net);
		// loadBNGammaParams("stage1.rebnconv1", 4, 32, net); 
		// loadBNBetaParams("stage1.rebnconv1", 4, 32, net); 
		// loadBNvarParams("stage1.rebnconv1", 4, 32, net); 
		// loadBNmeanParams("stage1.rebnconv1", 4, 32, net); 

		// loadCNNWeightParams("stage1.rebnconv2", 7, 32*32*9, net);
		// loadCNNBiasParams("stage1.rebnconv2", 7, 32, net);
		// loadBNGammaParams("stage1.rebnconv2", 8, 32, net); 
		// loadBNBetaParams("stage1.rebnconv2", 8, 32, net); 
		// loadBNvarParams("stage1.rebnconv2", 8, 32, net); 
		// loadBNmeanParams("stage1.rebnconv2", 8, 32, net); 

		// loadCNNWeightParams("stage1.rebnconv3", 11, 32*32*9, net);
		// loadCNNBiasParams("stage1.rebnconv3", 11, 32, net);
		// loadBNGammaParams("stage1.rebnconv3", 12, 32, net); 
		// loadBNBetaParams("stage1.rebnconv3", 12, 32, net); 
		// loadBNvarParams("stage1.rebnconv3", 12, 32, net); 
		// loadBNmeanParams("stage1.rebnconv3", 12, 32, net); 

		// print_vector(net->inputData, "FLOAT", "inputData:", 784);

		// /************************** Weight1 **********************************/
		// string path_weight1_1 = default_path+"weight1_"+to_string(partyNum);
		// string path_weight1_2 = default_path+"weight1_"+to_string(nextParty(partyNum));
		// ifstream f_weight1_1(path_weight1_1), f_weight1_2(path_weight1_2);

		// for (int row = 0; row < 3*12*9; ++row)
		// {
		// 	f_weight1_1 >> temp_next; f_weight1_2 >> temp_prev;
		// 	(*((CNNLayer*)net->layers[0])->getWeights())[row] = 
		// 			std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		// }
		// f_weight1_1.close(); f_weight1_2.close();
		// // // if (ZEROS)
		// // // {
		// // // 	generate_zeros("weight1_1", 5*5*1*20, temp);
		// // // 	generate_zeros("weight1_2", 5*5*1*20, temp);
		// // // }

		// // /************************** Bias1 **********************************/
		// string path_bias1_1 = default_path+"bias1_"+to_string(partyNum);
		// string path_bias1_2 = default_path+"bias1_"+to_string(nextParty(partyNum));
		// ifstream f_bias1_1(path_bias1_1), f_bias1_2(path_bias1_2);

		// for (int i = 0; i < 12; ++i)
		// {	
		// 	//cout<<1<<endl;

		// 	f_bias1_1 >> temp_next; f_bias1_2 >> temp_prev;
		// 	(*((CNNLayer*)net->layers[0])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		// }
		// f_bias1_1.close(); f_bias1_2.close();
		// if (ZEROS)
		// {
		// 	generate_zeros("weight2_1", 5*5*20*50, temp);
		// 	generate_zeros("weight2_2", 5*5*20*50, temp);
		// }

		// /************************** Weight3 **********************************/
		// string path_weight3_1 = default_path+"weight3_"+to_string(partyNum);
		// string path_weight3_2 = default_path+"weight3_"+to_string(nextParty(partyNum));
		// ifstream f_weight3_1(path_weight3_1), f_weight3_2(path_weight3_2);

		// for (int column = 0; column < 500; ++column)
		// {
		// 	for (int row = 0; row < 800; ++row)
		// 	{
		// 		f_weight3_1 >> temp_next; f_weight3_2 >> temp_prev;
		// 		(*((FCLayer*)net->layers[6])->getWeights())[500*row + column] = 
		// 				std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		// 	}
		// }
		// f_weight3_1.close(); f_weight3_2.close();
		// if (ZEROS)
		// {
		// 	generate_zeros("weight3_1", 800*500, temp);
		// 	generate_zeros("weight3_2", 800*500, temp);
		// }


		// /************************** Weight4 **********************************/
		// string path_weight4_1 = default_path+"weight4_"+to_string(partyNum);
		// string path_weight4_2 = default_path+"weight4_"+to_string(nextParty(partyNum));
		// ifstream f_weight4_1(path_weight4_1), f_weight4_2(path_weight4_2);

		// for (int column = 0; column < 10; ++column)
		// {
		// 	for (int row = 0; row < 500; ++row)
		// 	{
		// 		f_weight4_1 >> temp_next; f_weight4_2 >> temp_prev;
		// 		(*((FCLayer*)net->layers[8])->getWeights())[10*row + column] = 
		// 				std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		// 	}
		// }
		// f_weight4_1.close(); f_weight4_2.close();
		// if (ZEROS)
		// {
		// 	generate_zeros("weight4_1", 500*10, temp);
		// 	generate_zeros("weight4_2", 500*10, temp);
		// }

		// /************************** Bias1 **********************************/
		// string path_bias1_1 = default_path+"bias1_"+to_string(partyNum);
		// string path_bias1_2 = default_path+"bias1_"+to_string(nextParty(partyNum));
		// ifstream f_bias1_1(path_bias1_1), f_bias1_2(path_bias1_2);

		// for (int i = 0; i < 20; ++i)
		// {
		// 	f_bias1_1 >> temp_next; f_bias1_2 >> temp_prev;
		// 	(*((CNNLayer*)net->layers[0])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		// }
		// f_bias1_1.close(); f_bias1_2.close();
		// if (ZEROS)
		// {
		// 	generate_zeros("bias1_1", 20, temp);
		// 	generate_zeros("bias1_2", 20, temp);
		// }

		// /************************** Bias2 **********************************/
		// string path_bias2_1 = default_path+"bias2_"+to_string(partyNum);
		// string path_bias2_2 = default_path+"bias2_"+to_string(nextParty(partyNum));
		// ifstream f_bias2_1(path_bias2_1), f_bias2_2(path_bias2_2);

		// for (int i = 0; i < 50; ++i)
		// {
		// 	f_bias2_1 >> temp_next; f_bias2_2 >> temp_prev;
		// 	(*((CNNLayer*)net->layers[3])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		// }
		// f_bias2_1.close(); f_bias2_2.close();
		// if (ZEROS)
		// {
		// 	generate_zeros("bias2_1", 50, temp);
		// 	generate_zeros("bias2_2", 50, temp);
		// }

		// /************************** Bias3 **********************************/
		// string path_bias3_1 = default_path+"bias3_"+to_string(partyNum);
		// string path_bias3_2 = default_path+"bias3_"+to_string(nextParty(partyNum));
		// ifstream f_bias3_1(path_bias3_1), f_bias3_2(path_bias3_2);

		// for (int i = 0; i < 500; ++i)
		// {
		// 	f_bias3_1 >> temp_next; f_bias3_2 >> temp_prev;
		// 	(*((FCLayer*)net->layers[6])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		// }
		// f_bias3_1.close(); f_bias3_2.close();
		// if (ZEROS)
		// {
		// 	generate_zeros("bias3_1", 500, temp);
		// 	generate_zeros("bias3_2", 500, temp);
		// }

		// /************************** Bias4 **********************************/
		// string path_bias4_1 = default_path+"bias4_"+to_string(partyNum);
		// string path_bias4_2 = default_path+"bias4_"+to_string(nextParty(partyNum));
		// ifstream f_bias4_1(path_bias4_1), f_bias4_2(path_bias4_2);

		// for (int i = 0; i < 10; ++i)
		// {
		// 	f_bias4_1 >> temp_next; f_bias4_2 >> temp_prev;
		// 	(*((FCLayer*)net->layers[8])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		// }
		// f_bias4_1.close(); f_bias4_2.close();
		// if (ZEROS)
		// {
		// 	generate_zeros("bias4_1", 10, temp);
		// 	generate_zeros("bias4_2", 10, temp);
		// }
	}
	else if (which_network(network).compare("U2netNet") == 0)
	{
		string temp = "U2netNet";

		cout<<"in:"<<iter<<endl;;

		/************************** Input **********************************/
		string path_input_1 = default_path+"input_"+to_string(iter)+"_"+to_string(partyNum);
		string path_input_2 = default_path+"input_"+to_string(iter)+"_"+to_string(nextParty(partyNum));
		ifstream f_input_1(path_input_1), f_input_2(path_input_2);

		for (int i = 0; i < INPUT_SIZE * MINI_BATCH_SIZE; ++i)
		{
			f_input_1 >> temp_next; f_input_2 >> temp_prev;
			net->inputData[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_input_1.close(); f_input_2.close();
		
		print_vector(net->inputData, "FLOAT", "inputData:", 32);
		
		/************************** RSU7L0-CNN-Weight **********************************/
		loadCNNWeightParams("stage1.rebnconvin", 0, 3*64*9, net);
		loadCNNBiasParams("stage1.rebnconvin", 0, 64, net);
		/************************** RSU7 L3-CNN-Weight、bias**********************************/
		loadCNNWeightParams("stage1.rebnconv1", 3, 64*16*9, net);
		loadCNNBiasParams("stage1.rebnconv1", 3, 16, net);
		/************************** RSU7 L7-CNN-Weight、bias**********************************/
		loadCNNWeightParams("stage1.rebnconv2", 7, 16*16*9, net);
		loadCNNBiasParams("stage1.rebnconv2", 7, 16, net);
		/************************** RSU7 L11-CNN-Weight、bias**********************************/
		loadCNNWeightParams("stage1.rebnconv3", 11, 16*16*9, net);
		loadCNNBiasParams("stage1.rebnconv3", 11, 16, net);
		/************************** RSU7 L15-CNN-Weight、bias**********************************/
		loadCNNWeightParams("stage1.rebnconv4", 15, 16*16*9, net);
		loadCNNBiasParams("stage1.rebnconv4", 15, 16, net);
		/************************** RSU7 L19-CNN-Weight、bias**********************************/
		loadCNNWeightParams("stage1.rebnconv5", 19, 16*16*9, net);
		loadCNNBiasParams("stage1.rebnconv5", 19, 16, net);
		/************************** RSU7 L23-CNN-Weight、bias**********************************/
		loadCNNWeightParams("stage1.rebnconv6", 23, 16*16*9, net);
		loadCNNBiasParams("stage1.rebnconv6", 23, 16, net);
		/************************** RSU7 L26-CNN-Weight、bias**********************************/
		loadCNNWeightParams("stage1.rebnconv7", 26, 16*16*9, net);
		loadCNNBiasParams("stage1.rebnconv7", 26, 16, net);
		/************************** RSU7 L29-CNN-Weight、bias**********************************/
		loadCNNWeightParams("stage1.rebnconv6d", 29, 16*32*9, net);
		loadCNNBiasParams("stage1.rebnconv6d", 29, 16, net);
		/************************** RSU7 L33-CNN-Weight、bias**********************************/
		loadCNNWeightParams("stage1.rebnconv5d", 33, 16*32*9, net);
		loadCNNBiasParams("stage1.rebnconv5d", 33, 16, net);
		/************************** RSU7 L37-CNN-Weight、bias**********************************/
		loadCNNWeightParams("stage1.rebnconv4d", 37, 16*32*9, net);
		loadCNNBiasParams("stage1.rebnconv4d", 37, 16, net);
		/************************** RSU7 L41-CNN-Weight、bias**********************************/
		loadCNNWeightParams("stage1.rebnconv3d", 41, 16*32*9, net);
		loadCNNBiasParams("stage1.rebnconv3d", 41, 16, net);
		/************************** RSU7 L45-CNN-Weight、bias**********************************/
		loadCNNWeightParams("stage1.rebnconv2d", 45, 16*32*9, net);
		loadCNNBiasParams("stage1.rebnconv2d", 45, 16, net);
		/************************** RSU7 L49-CNN-Weight、bias**********************************/
		loadCNNWeightParams("stage1.rebnconv1d", 49, 32*64*9, net);
		loadCNNBiasParams("stage1.rebnconv1d", 49, 64, net);
		
		/************************** RSU7 L1-BN-gamma、beta**********************************/
		loadBNGammaParams("stage1.rebnconvin", 1, 64, net); 
		loadBNBetaParams("stage1.rebnconvin", 1, 64, net); 
		loadBNvarParams("stage1.rebnconvin", 1, 64, net); 
		loadBNmeanParams("stage1.rebnconvin", 1, 64, net); 
		/************************** RSU7 L4-BN-gamma、beta**********************************/
		loadBNGammaParams("stage1.rebnconv1", 4, 16, net); 
		loadBNBetaParams("stage1.rebnconv1", 4, 16, net); 
		loadBNvarParams("stage1.rebnconv1", 4, 16, net); 
		loadBNmeanParams("stage1.rebnconv1", 4, 16, net); 
		/************************** RSU7 L8-BN-gamma、beta **********************************/
		loadBNGammaParams("stage1.rebnconv2", 8, 16, net); 
		loadBNBetaParams("stage1.rebnconv2", 8, 16, net); 
		loadBNvarParams("stage1.rebnconv2", 8, 16, net); 
		loadBNmeanParams("stage1.rebnconv2", 8, 16, net); 
		/************************** RSU7 L12-BN-gamma、beta **********************************/
		loadBNGammaParams("stage1.rebnconv3", 12, 16, net); 
		loadBNBetaParams("stage1.rebnconv3", 12, 16, net); 
		loadBNvarParams("stage1.rebnconv3", 12, 16, net); 
		loadBNmeanParams("stage1.rebnconv3", 12, 16, net); 
		/************************** RSU7 L16-BN-gamma、beta **********************************/
		loadBNGammaParams("stage1.rebnconv4", 16, 16, net); 
		loadBNBetaParams("stage1.rebnconv4", 16, 16, net); 
		loadBNvarParams("stage1.rebnconv4", 16, 16, net); 
		loadBNmeanParams("stage1.rebnconv4", 16, 16, net); 
		/************************** RSU7 L20-BN-gamma、beta **********************************/
		loadBNGammaParams("stage1.rebnconv5", 20, 16, net); 
		loadBNBetaParams("stage1.rebnconv5", 20, 16, net); 
		loadBNvarParams("stage1.rebnconv5", 20, 16, net); 
		loadBNmeanParams("stage1.rebnconv5", 20, 16, net); 
		/************************** RSU7 L24-BN-gamma、beta **********************************/
		loadBNGammaParams("stage1.rebnconv6", 24, 16, net); 
		loadBNBetaParams("stage1.rebnconv6", 24, 16, net); 
		loadBNvarParams("stage1.rebnconv6", 24, 16, net); 
		loadBNmeanParams("stage1.rebnconv6", 24, 16, net); 
		/************************** RSU7 L27-BN-gamma、beta **********************************/
		loadBNGammaParams("stage1.rebnconv7", 27, 16, net); 
		loadBNBetaParams("stage1.rebnconv7", 27, 16, net); 
		loadBNvarParams("stage1.rebnconv7", 27, 16, net); 
		loadBNmeanParams("stage1.rebnconv7", 27, 16, net); 
		/************************** RSU7 L30-BN-gamma、beta **********************************/
		loadBNGammaParams("stage1.rebnconv6d", 30, 16, net); 
		loadBNBetaParams("stage1.rebnconv6d", 30, 16, net);
		loadBNvarParams("stage1.rebnconv6d", 30, 16, net); 
		loadBNmeanParams("stage1.rebnconv6d", 30, 16, net);
		/************************** RSU7 L34-BN-gamma、beta **********************************/
		loadBNGammaParams("stage1.rebnconv5d", 34, 16, net); 
		loadBNBetaParams("stage1.rebnconv5d", 34, 16, net);
		loadBNvarParams("stage1.rebnconv5d", 34, 16, net); 
		loadBNmeanParams("stage1.rebnconv5d", 34, 16, net);
		/************************** RSU7 L38-BN-gamma、beta **********************************/
		loadBNGammaParams("stage1.rebnconv4d", 38, 16, net); 
		loadBNBetaParams("stage1.rebnconv4d", 38, 16, net);
		loadBNvarParams("stage1.rebnconv4d", 38, 16, net); 
		loadBNmeanParams("stage1.rebnconv4d", 38, 16, net);
		/************************** RSU7 L42-BN-gamma、beta **********************************/
		loadBNGammaParams("stage1.rebnconv3d", 42, 16, net); 
		loadBNBetaParams("stage1.rebnconv3d", 42, 16, net);
		loadBNvarParams("stage1.rebnconv3d", 42, 16, net); 
		loadBNmeanParams("stage1.rebnconv3d", 42, 16, net);
		/************************** RSU7 L46-BN-gamma、beta **********************************/
		loadBNGammaParams("stage1.rebnconv2d", 46, 16, net); 
		loadBNBetaParams("stage1.rebnconv2d", 46, 16, net);
		loadBNvarParams("stage1.rebnconv2d", 46, 16, net); 
		loadBNmeanParams("stage1.rebnconv2d", 46, 16, net);
		/************************** RSU7 L50-BN-gamma、beta **********************************/
		loadBNGammaParams("stage1.rebnconv1d", 50, 64, net); 
		loadBNBetaParams("stage1.rebnconv1d", 50, 64, net);
		loadBNvarParams("stage1.rebnconv1d", 50, 64, net); 
		loadBNmeanParams("stage1.rebnconv1d", 50, 64, net);

		
		/************************** RSU6 convin**********************************/
		loadCNNWeightParams("stage2.rebnconvin", 53, 64*64*9, net);
		loadCNNBiasParams("stage2.rebnconvin", 53, 64, net);
		loadBNGammaParams("stage2.rebnconvin", 54, 64, net); 
		loadBNBetaParams("stage2.rebnconvin", 54, 64, net);
		loadBNvarParams("stage2.rebnconvin", 54, 64, net); 
		loadBNmeanParams("stage2.rebnconvin", 54, 64, net);

		/************************** RSU6 conv1**********************************/
		loadCNNWeightParams("stage2.rebnconv1", 56, 64*16*9, net);
		loadCNNBiasParams("stage2.rebnconv1", 56, 16, net);
		loadBNGammaParams("stage2.rebnconv1", 57, 16, net); 
		loadBNBetaParams("stage2.rebnconv1", 57, 16, net);
		loadBNvarParams("stage2.rebnconv1", 57, 16, net); 
		loadBNmeanParams("stage2.rebnconv1", 57, 16, net);

		/************************** RSU6 conv2**********************************/
		loadCNNWeightParams("stage2.rebnconv2", 60, 16*16*9, net);
		loadCNNBiasParams("stage2.rebnconv2", 60, 16, net);
		loadBNGammaParams("stage2.rebnconv2", 61, 16, net); 
		loadBNBetaParams("stage2.rebnconv2", 61, 16, net);
		loadBNvarParams("stage2.rebnconv2", 61, 16, net); 
		loadBNmeanParams("stage2.rebnconv2", 61, 16, net);

		/************************** RSU6 conv3**********************************/
		loadCNNWeightParams("stage2.rebnconv3", 64, 16*16*9, net);
		loadCNNBiasParams("stage2.rebnconv3", 64, 16, net);
		loadBNGammaParams("stage2.rebnconv3", 65, 16, net); 
		loadBNBetaParams("stage2.rebnconv3", 65, 16, net);
		loadBNvarParams("stage2.rebnconv3", 65, 16, net); 
		loadBNmeanParams("stage2.rebnconv3", 65, 16, net);

		/************************** RSU6 conv4**********************************/
		loadCNNWeightParams("stage2.rebnconv4", 68, 16*16*9, net);
		loadCNNBiasParams("stage2.rebnconv4", 68, 16, net);
		loadBNGammaParams("stage2.rebnconv4", 69, 16, net); 
		loadBNBetaParams("stage2.rebnconv4", 69, 16, net);
		loadBNvarParams("stage2.rebnconv4", 69, 16, net); 
		loadBNmeanParams("stage2.rebnconv4", 69, 16, net);

		/************************** RSU6 conv5**********************************/
		loadCNNWeightParams("stage2.rebnconv5", 72, 16*16*9, net);
		loadCNNBiasParams("stage2.rebnconv5", 72, 16, net);
		loadBNGammaParams("stage2.rebnconv5", 73, 16, net); 
		loadBNBetaParams("stage2.rebnconv5", 73, 16, net);
		loadBNvarParams("stage2.rebnconv5", 73, 16, net); 
		loadBNmeanParams("stage2.rebnconv5", 73, 16, net);

		/************************** RSU6 conv6**********************************/
		loadCNNWeightParams("stage2.rebnconv6", 75, 16*16*9, net);
		loadCNNBiasParams("stage2.rebnconv6", 75, 16, net);
		loadBNGammaParams("stage2.rebnconv6", 76, 16, net); 
		loadBNBetaParams("stage2.rebnconv6", 76, 16, net);
		loadBNvarParams("stage2.rebnconv6", 76, 16, net); 
		loadBNmeanParams("stage2.rebnconv6", 76, 16, net);

		/************************** RSU6 conv5d**********************************/
		loadCNNWeightParams("stage2.rebnconv5d", 78, 32*16*9, net);
		loadCNNBiasParams("stage2.rebnconv5d", 78, 16, net);
		loadBNGammaParams("stage2.rebnconv5d", 79, 16, net); 
		loadBNBetaParams("stage2.rebnconv5d", 79, 16, net);
		loadBNvarParams("stage2.rebnconv5d", 79, 16, net); 
		loadBNmeanParams("stage2.rebnconv5d", 79, 16, net);

		/************************** RSU6 conv4d**********************************/
		loadCNNWeightParams("stage2.rebnconv4d", 82, 32*16*9, net);
		loadCNNBiasParams("stage2.rebnconv4d", 82, 16, net);
		loadBNGammaParams("stage2.rebnconv4d", 83, 16, net); 
		loadBNBetaParams("stage2.rebnconv4d", 83, 16, net);
		loadBNvarParams("stage2.rebnconv4d", 83, 16, net); 
		loadBNmeanParams("stage2.rebnconv4d", 83, 16, net);

		/************************** RSU6 conv3d**********************************/
		loadCNNWeightParams("stage2.rebnconv3d", 86, 32*16*9, net);
		loadCNNBiasParams("stage2.rebnconv3d", 86, 16, net);
		loadBNGammaParams("stage2.rebnconv3d", 87, 16, net); 
		loadBNBetaParams("stage2.rebnconv3d", 87, 16, net);
		loadBNvarParams("stage2.rebnconv3d", 87, 16, net); 
		loadBNmeanParams("stage2.rebnconv3d", 87, 16, net);

		/************************** RSU6 conv2d**********************************/
		loadCNNWeightParams("stage2.rebnconv2d", 90, 32*16*9, net);
		loadCNNBiasParams("stage2.rebnconv2d", 90, 16, net);
		loadBNGammaParams("stage2.rebnconv2d", 91, 16, net); 
		loadBNBetaParams("stage2.rebnconv2d", 91, 16, net);
		loadBNvarParams("stage2.rebnconv2d", 91, 16, net); 
		loadBNmeanParams("stage2.rebnconv2d", 91, 16, net);

		/************************** RSU6 conv1d**********************************/
		loadCNNWeightParams("stage2.rebnconv1d", 94, 32*64*9, net);
		loadCNNBiasParams("stage2.rebnconv1d", 94, 64, net);
		loadBNGammaParams("stage2.rebnconv1d", 95, 64, net); 
		loadBNBetaParams("stage2.rebnconv1d", 95, 64, net);
		loadBNvarParams("stage2.rebnconv1d", 95, 64, net); 
		loadBNmeanParams("stage2.rebnconv1d", 95, 64, net);

		/************************** RSU5 convin**********************************/
		loadCNNWeightParams("stage3.rebnconvin", 98, 64*64*9, net);
		loadCNNBiasParams("stage3.rebnconvin", 98, 64, net);
		loadBNGammaParams("stage3.rebnconvin", 99, 64, net); 
		loadBNBetaParams("stage3.rebnconvin", 99, 64, net);
		loadBNvarParams("stage3.rebnconvin", 99, 64, net); 
		loadBNmeanParams("stage3.rebnconvin", 99, 64, net);

		/************************** RSU5 conv1**********************************/
		loadCNNWeightParams("stage3.rebnconv1", 101, 64*16*9, net);
		loadCNNBiasParams("stage3.rebnconv1", 101, 16, net);
		loadBNGammaParams("stage3.rebnconv1", 102, 16, net); 
		loadBNBetaParams("stage3.rebnconv1", 102, 16, net);
		loadBNvarParams("stage3.rebnconv1", 102, 16, net); 
		loadBNmeanParams("stage3.rebnconv1", 102, 16, net);

		/************************** RSU5 conv2**********************************/
		loadCNNWeightParams("stage3.rebnconv2", 105, 16*16*9, net);
		loadCNNBiasParams("stage3.rebnconv2", 105, 16, net);
		loadBNGammaParams("stage3.rebnconv2", 106, 16, net); 
		loadBNBetaParams("stage3.rebnconv2", 106, 16, net);
		loadBNvarParams("stage3.rebnconv2", 106, 16, net); 
		loadBNmeanParams("stage3.rebnconv2", 106, 16, net);

		/************************** RSU5 conv3**********************************/
		loadCNNWeightParams("stage3.rebnconv3", 109, 16*16*9, net);
		loadCNNBiasParams("stage3.rebnconv3", 109, 16, net);
		loadBNGammaParams("stage3.rebnconv3", 110, 16, net); 
		loadBNBetaParams("stage3.rebnconv3", 110, 16, net);
		loadBNvarParams("stage3.rebnconv3", 110, 16, net); 
		loadBNmeanParams("stage3.rebnconv3", 110, 16, net);

		/************************** RSU5 conv4**********************************/
		loadCNNWeightParams("stage3.rebnconv4", 113, 16*16*9, net);
		loadCNNBiasParams("stage3.rebnconv4", 113, 16, net);
		loadBNGammaParams("stage3.rebnconv4", 114, 16, net); 
		loadBNBetaParams("stage3.rebnconv4", 114, 16, net);
		loadBNvarParams("stage3.rebnconv4", 114, 16, net); 
		loadBNmeanParams("stage3.rebnconv4", 114, 16, net);

		/************************** RSU5 conv5**********************************/
		loadCNNWeightParams("stage3.rebnconv5", 116, 16*16*9, net);
		loadCNNBiasParams("stage3.rebnconv5", 116, 16, net);
		loadBNGammaParams("stage3.rebnconv5", 117, 16, net); 
		loadBNBetaParams("stage3.rebnconv5", 117, 16, net);
		loadBNvarParams("stage3.rebnconv5", 117, 16, net); 
		loadBNmeanParams("stage3.rebnconv5", 117, 16, net);

		/************************** RSU5 conv4d**********************************/
		loadCNNWeightParams("stage3.rebnconv4d", 119, 32*16*9, net);
		loadCNNBiasParams("stage3.rebnconv4d", 119, 16, net);
		loadBNGammaParams("stage3.rebnconv4d", 120, 16, net); 
		loadBNBetaParams("stage3.rebnconv4d", 120, 16, net);
		loadBNvarParams("stage3.rebnconv4d", 120, 16, net); 
		loadBNmeanParams("stage3.rebnconv4d", 120, 16, net);

		/************************** RSU5 conv3d**********************************/
		loadCNNWeightParams("stage3.rebnconv3d", 123, 32*16*9, net);
		loadCNNBiasParams("stage3.rebnconv3d", 123, 16, net);
		loadBNGammaParams("stage3.rebnconv3d", 124, 16, net); 
		loadBNBetaParams("stage3.rebnconv3d", 124, 16, net);
		loadBNvarParams("stage3.rebnconv3d", 124, 16, net); 
		loadBNmeanParams("stage3.rebnconv3d", 124, 16, net);

		/************************** RSU5 conv2d**********************************/
		loadCNNWeightParams("stage3.rebnconv2d", 127, 32*16*9, net);
		loadCNNBiasParams("stage3.rebnconv2d", 127, 16, net);
		loadBNGammaParams("stage3.rebnconv2d", 128, 16, net); 
		loadBNBetaParams("stage3.rebnconv2d", 128, 16, net);
		loadBNvarParams("stage3.rebnconv2d", 128, 16, net); 
		loadBNmeanParams("stage3.rebnconv2d", 128, 16, net);

		/************************** RSU5 conv1d**********************************/
		loadCNNWeightParams("stage3.rebnconv1d", 131, 32*64*9, net);
		loadCNNBiasParams("stage3.rebnconv1d", 131, 64, net);
		loadBNGammaParams("stage3.rebnconv1d", 132, 64, net); 
		loadBNBetaParams("stage3.rebnconv1d", 132, 64, net);
		loadBNvarParams("stage3.rebnconv1d", 132, 64, net); 
		loadBNmeanParams("stage3.rebnconv1d", 132, 64, net);
		
		/************************** RSU4 convin**********************************/
		loadCNNWeightParams("stage4.rebnconvin", 135, 64*64*9, net);
		loadCNNBiasParams("stage4.rebnconvin", 135, 64, net);
		loadBNGammaParams("stage4.rebnconvin", 136, 64, net); 
		loadBNBetaParams("stage4.rebnconvin", 136, 64, net);
		loadBNvarParams("stage4.rebnconvin", 136, 64, net); 
		loadBNmeanParams("stage4.rebnconvin", 136, 64, net);

		/************************** RSU4 conv1**********************************/
		loadCNNWeightParams("stage4.rebnconv1", 138, 64*16*9, net);
		loadCNNBiasParams("stage4.rebnconv1", 138, 16, net);
		loadBNGammaParams("stage4.rebnconv1", 139, 16, net); 
		loadBNBetaParams("stage4.rebnconv1", 139, 16, net);
		loadBNvarParams("stage4.rebnconv1", 139, 16, net); 
		loadBNmeanParams("stage4.rebnconv1", 139, 16, net);

		/************************** RSU4 conv2**********************************/
		loadCNNWeightParams("stage4.rebnconv2", 142, 16*16*9, net);
		loadCNNBiasParams("stage4.rebnconv2", 142, 16, net);
		loadBNGammaParams("stage4.rebnconv2", 143, 16, net); 
		loadBNBetaParams("stage4.rebnconv2", 143, 16, net);
		loadBNvarParams("stage4.rebnconv2", 143, 16, net); 
		loadBNmeanParams("stage4.rebnconv2", 143, 16, net);

		/************************** RSU4 conv3**********************************/
		loadCNNWeightParams("stage4.rebnconv3", 146, 16*16*9, net);
		loadCNNBiasParams("stage4.rebnconv3", 146, 16, net);
		loadBNGammaParams("stage4.rebnconv3", 147, 16, net); 
		loadBNBetaParams("stage4.rebnconv3", 147, 16, net);
		loadBNvarParams("stage4.rebnconv3", 147, 16, net); 
		loadBNmeanParams("stage4.rebnconv3", 147, 16, net);

		/************************** RSU4 conv4**********************************/
		loadCNNWeightParams("stage4.rebnconv4", 149, 16*16*9, net);
		loadCNNBiasParams("stage4.rebnconv4", 149, 16, net);
		loadBNGammaParams("stage4.rebnconv4", 150, 16, net); 
		loadBNBetaParams("stage4.rebnconv4", 150, 16, net);
		loadBNvarParams("stage4.rebnconv4", 150, 16, net); 
		loadBNmeanParams("stage4.rebnconv4", 150, 16, net);

		/************************** RSU4 conv3d**********************************/
		loadCNNWeightParams("stage4.rebnconv3d", 152, 32*16*9, net);
		loadCNNBiasParams("stage4.rebnconv3d", 152, 16, net);
		loadBNGammaParams("stage4.rebnconv3d", 153, 16, net); 
		loadBNBetaParams("stage4.rebnconv3d", 153, 16, net);
		loadBNvarParams("stage4.rebnconv3d", 153, 16, net); 
		loadBNmeanParams("stage4.rebnconv3d", 153, 16, net);

		/************************** RSU4 conv2d**********************************/
		loadCNNWeightParams("stage4.rebnconv2d", 156, 32*16*9, net);
		loadCNNBiasParams("stage4.rebnconv2d", 156, 16, net);
		loadBNGammaParams("stage4.rebnconv2d", 157, 16, net); 
		loadBNBetaParams("stage4.rebnconv2d", 157, 16, net);
		loadBNvarParams("stage4.rebnconv2d", 157, 16, net); 
		loadBNmeanParams("stage4.rebnconv2d", 157, 16, net);

		/************************** RSU4 conv1d**********************************/
		loadCNNWeightParams("stage4.rebnconv1d", 160, 32*64*9, net);
		loadCNNBiasParams("stage4.rebnconv1d", 160, 64, net);
		loadBNGammaParams("stage4.rebnconv1d", 161, 64, net); 
		loadBNBetaParams("stage4.rebnconv1d", 161, 64, net);
		loadBNvarParams("stage4.rebnconv1d", 161, 64, net); 
		loadBNmeanParams("stage4.rebnconv1d", 161, 64, net);
		
		
		/************************** RSU4f convin**********************************/
		loadCNNWeightParams("stage5.rebnconvin", 164, 64*64*9, net);
		loadCNNBiasParams("stage5.rebnconvin", 164, 64, net);
		loadBNGammaParams("stage5.rebnconvin", 165, 64, net); 
		loadBNBetaParams("stage5.rebnconvin", 165, 64, net);
		loadBNvarParams("stage5.rebnconvin", 165, 64, net); 
		loadBNmeanParams("stage5.rebnconvin", 165, 64, net);
		/************************** RSU4f conv1**********************************/
		loadCNNWeightParams("stage5.rebnconv1", 167, 64*16*9, net);
		loadCNNBiasParams("stage5.rebnconv1", 167, 16, net);
		loadBNGammaParams("stage5.rebnconv1", 168, 16, net); 
		loadBNBetaParams("stage5.rebnconv1", 168, 16, net);
		loadBNvarParams("stage5.rebnconv1", 168, 16, net); 
		loadBNmeanParams("stage5.rebnconv1", 168, 16, net);
		/************************** RSU4f conv2**********************************/
		loadCNNWeightParams("stage5.rebnconv2", 170, 16*16*9, net);
		loadCNNBiasParams("stage5.rebnconv2", 170, 16, net);
		loadBNGammaParams("stage5.rebnconv2", 171, 16, net); 
		loadBNBetaParams("stage5.rebnconv2", 171, 16, net);
		loadBNvarParams("stage5.rebnconv2", 171, 16, net); 
		loadBNmeanParams("stage5.rebnconv2", 171, 16, net);
		/************************** RSU4f conv3**********************************/
		loadCNNWeightParams("stage5.rebnconv3", 173, 16*16*9, net);
		loadCNNBiasParams("stage5.rebnconv3", 173, 16, net);
		loadBNGammaParams("stage5.rebnconv3", 174, 16, net); 
		loadBNBetaParams("stage5.rebnconv3", 174, 16, net);
		loadBNvarParams("stage5.rebnconv3", 174, 16, net); 
		loadBNmeanParams("stage5.rebnconv3", 174, 16, net);
		/************************** RSU4f conv4**********************************/
		loadCNNWeightParams("stage5.rebnconv4", 176, 16*16*9, net);
		loadCNNBiasParams("stage5.rebnconv4", 176, 16, net);
		loadBNGammaParams("stage5.rebnconv4", 177, 16, net); 
		loadBNBetaParams("stage5.rebnconv4", 177, 16, net);
		loadBNvarParams("stage5.rebnconv4", 177, 16, net); 
		loadBNmeanParams("stage5.rebnconv4", 177, 16, net);
		/************************** RSU4f conv3d**********************************/
		loadCNNWeightParams("stage5.rebnconv3d", 179, 32*16*9, net);
		loadCNNBiasParams("stage5.rebnconv3d", 179, 16, net);
		loadBNGammaParams("stage5.rebnconv3d", 180, 16, net); 
		loadBNBetaParams("stage5.rebnconv3d", 180, 16, net);
		loadBNvarParams("stage5.rebnconv3d", 180, 16, net); 
		loadBNmeanParams("stage5.rebnconv3d", 180, 16, net);
		/************************** RSU4f conv2d**********************************/
		loadCNNWeightParams("stage5.rebnconv2d", 182, 32*16*9, net);
		loadCNNBiasParams("stage5.rebnconv2d", 182, 16, net);
		loadBNGammaParams("stage5.rebnconv2d", 183, 16, net); 
		loadBNBetaParams("stage5.rebnconv2d", 183, 16, net);
		loadBNvarParams("stage5.rebnconv2d", 183, 16, net); 
		loadBNmeanParams("stage5.rebnconv2d", 183, 16, net);
		/************************** RSU4f conv1d**********************************/
		loadCNNWeightParams("stage5.rebnconv1d", 185, 32*64*9, net);
		loadCNNBiasParams("stage5.rebnconv1d", 185, 64, net);
		loadBNGammaParams("stage5.rebnconv1d", 186, 64, net); 
		loadBNBetaParams("stage5.rebnconv1d", 186, 64, net);
		loadBNvarParams("stage5.rebnconv1d", 186, 64, net); 
		loadBNmeanParams("stage5.rebnconv1d", 186, 64, net);

		/************************** RSU4f convin**********************************/
		loadCNNWeightParams("stage6.rebnconvin", 189, 64*64*9, net);
		loadCNNBiasParams("stage6.rebnconvin", 189, 64, net);
		loadBNGammaParams("stage6.rebnconvin", 190, 64, net); 
		loadBNBetaParams("stage6.rebnconvin", 190, 64, net);
		loadBNvarParams("stage6.rebnconvin", 190, 64, net); 
		loadBNmeanParams("stage6.rebnconvin", 190, 64, net);
		/************************** RSU4f conv1**********************************/
		loadCNNWeightParams("stage6.rebnconv1", 192, 64*16*9, net);
		loadCNNBiasParams("stage6.rebnconv1", 192, 16, net);
		loadBNGammaParams("stage6.rebnconv1", 193, 16, net); 
		loadBNBetaParams("stage6.rebnconv1", 193, 16, net);
		loadBNvarParams("stage6.rebnconv1", 193, 16, net); 
		loadBNmeanParams("stage6.rebnconv1", 193, 16, net);
		/************************** RSU4f conv2**********************************/
		loadCNNWeightParams("stage6.rebnconv2", 195, 16*16*9, net);
		loadCNNBiasParams("stage6.rebnconv2", 195, 16, net);
		loadBNGammaParams("stage6.rebnconv2", 196, 16, net); 
		loadBNBetaParams("stage6.rebnconv2", 196, 16, net);
		loadBNvarParams("stage6.rebnconv2", 196, 16, net); 
		loadBNmeanParams("stage6.rebnconv2", 196, 16, net);
		/************************** RSU4f conv3**********************************/
		loadCNNWeightParams("stage6.rebnconv3", 198, 16*16*9, net);
		loadCNNBiasParams("stage6.rebnconv3", 198, 16, net);
		loadBNGammaParams("stage6.rebnconv3", 199, 16, net); 
		loadBNBetaParams("stage6.rebnconv3", 199, 16, net);
		loadBNvarParams("stage6.rebnconv3", 199, 16, net); 
		loadBNmeanParams("stage6.rebnconv3", 199, 16, net);
		/************************** RSU4f conv4**********************************/
		loadCNNWeightParams("stage6.rebnconv4", 201, 16*16*9, net);
		loadCNNBiasParams("stage6.rebnconv4", 201, 16, net);
		loadBNGammaParams("stage6.rebnconv4", 202, 16, net); 
		loadBNBetaParams("stage6.rebnconv4", 202, 16, net);
		loadBNvarParams("stage6.rebnconv4", 202, 16, net); 
		loadBNmeanParams("stage6.rebnconv4", 202, 16, net);
		/************************** RSU4f conv3d**********************************/
		loadCNNWeightParams("stage6.rebnconv3d", 204, 32*16*9, net);
		loadCNNBiasParams("stage6.rebnconv3d", 204, 16, net);
		loadBNGammaParams("stage6.rebnconv3d", 205, 16, net); 
		loadBNBetaParams("stage6.rebnconv3d", 205, 16, net);
		loadBNvarParams("stage6.rebnconv3d", 205, 16, net); 
		loadBNmeanParams("stage6.rebnconv3d", 205, 16, net);
		/************************** RSU4f conv2d**********************************/
		loadCNNWeightParams("stage6.rebnconv2d", 207, 32*16*9, net);
		loadCNNBiasParams("stage6.rebnconv2d", 207, 16, net);
		loadBNGammaParams("stage6.rebnconv2d", 208, 16, net); 
		loadBNBetaParams("stage6.rebnconv2d", 208, 16, net);
		loadBNvarParams("stage6.rebnconv2d", 208, 16, net); 
		loadBNmeanParams("stage6.rebnconv2d", 208, 16, net);
		/************************** RSU4f conv1d**********************************/
		loadCNNWeightParams("stage6.rebnconv1d", 210, 32*64*9, net);
		loadCNNBiasParams("stage6.rebnconv1d", 210, 64, net);
		loadBNGammaParams("stage6.rebnconv1d", 211, 64, net); 
		loadBNBetaParams("stage6.rebnconv1d", 211, 64, net);
		loadBNvarParams("stage6.rebnconv1d", 211, 64, net); 
		loadBNmeanParams("stage6.rebnconv1d", 211, 64, net);
		
		/************************** RSU4f convin**********************************/
		loadCNNWeightParams("stage5d.rebnconvin", 214, 128*64*9, net);
		loadCNNBiasParams("stage5d.rebnconvin", 214, 64, net);
		loadBNGammaParams("stage5d.rebnconvin", 215, 64, net); 
		loadBNBetaParams("stage5d.rebnconvin", 215, 64, net);
		loadBNvarParams("stage5d.rebnconvin", 215, 64, net); 
		loadBNmeanParams("stage5d.rebnconvin", 215, 64, net);

		/************************** RSU4f conv1**********************************/
		loadCNNWeightParams("stage5d.rebnconv1", 217, 64*16*9, net);
		loadCNNBiasParams("stage5d.rebnconv1", 217, 16, net);
		loadBNGammaParams("stage5d.rebnconv1", 218, 16, net); 
		loadBNBetaParams("stage5d.rebnconv1", 218, 16, net);
		loadBNvarParams("stage5d.rebnconv1", 218, 16, net); 
		loadBNmeanParams("stage5d.rebnconv1", 218, 16, net);

		/************************** RSU4f conv2**********************************/
		loadCNNWeightParams("stage5d.rebnconv2", 220, 16*16*9, net);
		loadCNNBiasParams("stage5d.rebnconv2", 220, 16, net);
		loadBNGammaParams("stage5d.rebnconv2", 221, 16, net); 
		loadBNBetaParams("stage5d.rebnconv2", 221, 16, net);
		loadBNvarParams("stage5d.rebnconv2", 221, 16, net); 
		loadBNmeanParams("stage5d.rebnconv2", 221, 16, net);

		/************************** RSU4f conv3**********************************/
		loadCNNWeightParams("stage5d.rebnconv3", 223, 16*16*9, net);
		loadCNNBiasParams("stage5d.rebnconv3", 223, 16, net);
		loadBNGammaParams("stage5d.rebnconv3", 224, 16, net); 
		loadBNBetaParams("stage5d.rebnconv3", 224, 16, net);
		loadBNvarParams("stage5d.rebnconv3", 224, 16, net); 
		loadBNmeanParams("stage5d.rebnconv3", 224, 16, net);

		/************************** RSU4f conv4**********************************/
		loadCNNWeightParams("stage5d.rebnconv4", 226, 16*16*9, net);
		loadCNNBiasParams("stage5d.rebnconv4", 226, 16, net);
		loadBNGammaParams("stage5d.rebnconv4", 227, 16, net); 
		loadBNBetaParams("stage5d.rebnconv4", 227, 16, net);
		loadBNvarParams("stage5d.rebnconv4", 227, 16, net); 
		loadBNmeanParams("stage5d.rebnconv4", 227, 16, net);

		/************************** RSU4f conv3d**********************************/
		loadCNNWeightParams("stage5d.rebnconv3d", 229, 32*16*9, net);
		loadCNNBiasParams("stage5d.rebnconv3d", 229, 16, net);
		loadBNGammaParams("stage5d.rebnconv3d", 230, 16, net); 
		loadBNBetaParams("stage5d.rebnconv3d", 230, 16, net);
		loadBNvarParams("stage5d.rebnconv3d", 230, 16, net); 
		loadBNmeanParams("stage5d.rebnconv3d", 230, 16, net);

		/************************** RSU4f conv2d**********************************/
		loadCNNWeightParams("stage5d.rebnconv2d", 232, 32*16*9, net);
		loadCNNBiasParams("stage5d.rebnconv2d", 232, 16, net);
		loadBNGammaParams("stage5d.rebnconv2d", 233, 16, net); 
		loadBNBetaParams("stage5d.rebnconv2d", 233, 16, net);
		loadBNvarParams("stage5d.rebnconv2d", 233, 16, net); 
		loadBNmeanParams("stage5d.rebnconv2d", 233, 16, net);

		/************************** RSU4f conv1d**********************************/
		loadCNNWeightParams("stage5d.rebnconv1d", 235, 32*64*9, net);
		loadCNNBiasParams("stage5d.rebnconv1d", 235, 64, net);
		loadBNGammaParams("stage5d.rebnconv1d", 236, 64, net); 
		loadBNBetaParams("stage5d.rebnconv1d", 236, 64, net);
		loadBNvarParams("stage5d.rebnconv1d", 236, 64, net); 
		loadBNmeanParams("stage5d.rebnconv1d", 236, 64, net);

		/************************** RSU4 convin**********************************/
		loadCNNWeightParams("stage4d.rebnconvin", 239, 128*64*9, net);
		loadCNNBiasParams("stage4d.rebnconvin", 239, 64, net);
		loadBNGammaParams("stage4d.rebnconvin", 240, 64, net); 
		loadBNBetaParams("stage4d.rebnconvin", 240, 64, net);
		loadBNvarParams("stage4d.rebnconvin", 240, 64, net); 
		loadBNmeanParams("stage4d.rebnconvin", 240, 64, net);

		/************************** RSU4 conv1**********************************/
		loadCNNWeightParams("stage4d.rebnconv1", 242, 64*16*9, net);
		loadCNNBiasParams("stage4d.rebnconv1", 242, 16, net);
		loadBNGammaParams("stage4d.rebnconv1", 243, 16, net); 
		loadBNBetaParams("stage4d.rebnconv1", 243, 16, net);
		loadBNvarParams("stage4d.rebnconv1", 243, 16, net); 
		loadBNmeanParams("stage4d.rebnconv1", 243, 16, net);

		/************************** RSU4 conv2**********************************/
		loadCNNWeightParams("stage4d.rebnconv2", 246, 16*16*9, net);
		loadCNNBiasParams("stage4d.rebnconv2", 246, 16, net);
		loadBNGammaParams("stage4d.rebnconv2", 247, 16, net); 
		loadBNBetaParams("stage4d.rebnconv2", 247, 16, net);
		loadBNvarParams("stage4d.rebnconv2", 247, 16, net); 
		loadBNmeanParams("stage4d.rebnconv2", 247, 16, net);

		/************************** RSU4 conv3**********************************/
		loadCNNWeightParams("stage4d.rebnconv3", 250, 16*16*9, net);
		loadCNNBiasParams("stage4d.rebnconv3", 250, 16, net);
		loadBNGammaParams("stage4d.rebnconv3", 251, 16, net); 
		loadBNBetaParams("stage4d.rebnconv3", 251, 16, net);
		loadBNvarParams("stage4d.rebnconv3", 251, 16, net); 
		loadBNmeanParams("stage4d.rebnconv3", 251, 16, net);

		/************************** RSU4 conv4**********************************/
		loadCNNWeightParams("stage4d.rebnconv4", 253, 16*16*9, net);
		loadCNNBiasParams("stage4d.rebnconv4", 253, 16, net);
		loadBNGammaParams("stage4d.rebnconv4", 254, 16, net); 
		loadBNBetaParams("stage4d.rebnconv4", 254, 16, net);
		loadBNvarParams("stage4d.rebnconv4", 254, 16, net); 
		loadBNmeanParams("stage4d.rebnconv4", 254, 16, net);

		/************************** RSU4 conv3d**********************************/
		loadCNNWeightParams("stage4d.rebnconv3d", 256, 32*16*9, net);
		loadCNNBiasParams("stage4d.rebnconv3d", 256, 16, net);
		loadBNGammaParams("stage4d.rebnconv3d", 257, 16, net); 
		loadBNBetaParams("stage4d.rebnconv3d", 257, 16, net);
		loadBNvarParams("stage4d.rebnconv3d", 257, 16, net); 
		loadBNmeanParams("stage4d.rebnconv3d", 257, 16, net);

		/************************** RSU4 conv2d**********************************/
		loadCNNWeightParams("stage4d.rebnconv2d", 260, 32*16*9, net);
		loadCNNBiasParams("stage4d.rebnconv2d", 260, 16, net);
		loadBNGammaParams("stage4d.rebnconv2d", 261, 16, net); 
		loadBNBetaParams("stage4d.rebnconv2d", 261, 16, net);
		loadBNvarParams("stage4d.rebnconv2d", 261, 16, net); 
		loadBNmeanParams("stage4d.rebnconv2d", 261, 16, net);

		/************************** RSU4 conv1d**********************************/
		loadCNNWeightParams("stage4d.rebnconv1d", 264, 32*64*9, net);
		loadCNNBiasParams("stage4d.rebnconv1d", 264, 64, net);
		loadBNGammaParams("stage4d.rebnconv1d", 265, 64, net); 
		loadBNBetaParams("stage4d.rebnconv1d", 265, 64, net);
		loadBNvarParams("stage4d.rebnconv1d", 265, 64, net); 
		loadBNmeanParams("stage4d.rebnconv1d", 265, 64, net);

		/************************** RSU5 convin**********************************/
		loadCNNWeightParams("stage3d.rebnconvin", 268, 128*64*9, net);
		loadCNNBiasParams("stage3d.rebnconvin", 268, 64, net);
		loadBNGammaParams("stage3d.rebnconvin", 269, 64, net); 
		loadBNBetaParams("stage3d.rebnconvin", 269, 64, net);
		loadBNvarParams("stage3d.rebnconvin", 269, 64, net); 
		loadBNmeanParams("stage3d.rebnconvin", 269, 64, net);
		
		/************************** RSU5 conv1**********************************/
		loadCNNWeightParams("stage3d.rebnconv1", 271, 64*16*9, net);
		loadCNNBiasParams("stage3d.rebnconv1", 271, 16, net);
		loadBNGammaParams("stage3d.rebnconv1", 272, 16, net); 
		loadBNBetaParams("stage3d.rebnconv1", 272, 16, net);
		loadBNvarParams("stage3d.rebnconv1", 272, 16, net); 
		loadBNmeanParams("stage3d.rebnconv1", 272, 16, net);
		
		/************************** RSU5 conv2**********************************/
		loadCNNWeightParams("stage3d.rebnconv2", 275, 16*16*9, net);
		loadCNNBiasParams("stage3d.rebnconv2", 275, 16, net);
		loadBNGammaParams("stage3d.rebnconv2", 276, 16, net); 
		loadBNBetaParams("stage3d.rebnconv2", 276, 16, net);
		loadBNvarParams("stage3d.rebnconv2", 276, 16, net); 
		loadBNmeanParams("stage3d.rebnconv2", 276, 16, net);
		
		/************************** RSU5 conv3**********************************/
		loadCNNWeightParams("stage3d.rebnconv3", 279, 16*16*9, net);
		loadCNNBiasParams("stage3d.rebnconv3", 279, 16, net);
		loadBNGammaParams("stage3d.rebnconv3", 280, 16, net); 
		loadBNBetaParams("stage3d.rebnconv3", 280, 16, net);
		loadBNvarParams("stage3d.rebnconv3", 280, 16, net); 
		loadBNmeanParams("stage3d.rebnconv3", 280, 16, net);
		
		/************************** RSU5 conv4**********************************/
		loadCNNWeightParams("stage3d.rebnconv4", 283, 16*16*9, net);
		loadCNNBiasParams("stage3d.rebnconv4", 283, 16, net);
		loadBNGammaParams("stage3d.rebnconv4", 284, 16, net); 
		loadBNBetaParams("stage3d.rebnconv4", 284, 16, net);
		loadBNvarParams("stage3d.rebnconv4", 284, 16, net); 
		loadBNmeanParams("stage3d.rebnconv4", 284, 16, net);
		
		/************************** RSU5 conv5**********************************/
		loadCNNWeightParams("stage3d.rebnconv5", 286, 16*16*9, net);
		loadCNNBiasParams("stage3d.rebnconv5", 286, 16, net);
		loadBNGammaParams("stage3d.rebnconv5", 287, 16, net); 
		loadBNBetaParams("stage3d.rebnconv5", 287, 16, net);
		loadBNvarParams("stage3d.rebnconv5", 287, 16, net); 
		loadBNmeanParams("stage3d.rebnconv5", 287, 16, net);
		
		/************************** RSU5 conv4d**********************************/
		loadCNNWeightParams("stage3d.rebnconv4d", 289, 32*16*9, net);
		loadCNNBiasParams("stage3d.rebnconv4d", 289, 16, net);
		loadBNGammaParams("stage3d.rebnconv4d", 290, 16, net); 
		loadBNBetaParams("stage3d.rebnconv4d", 290, 16, net);
		loadBNvarParams("stage3d.rebnconv4d", 290, 16, net); 
		loadBNmeanParams("stage3d.rebnconv4d", 290, 16, net);
		
		/************************** RSU5 conv3d**********************************/
		loadCNNWeightParams("stage3d.rebnconv3d", 293, 32*16*9, net);
		loadCNNBiasParams("stage3d.rebnconv3d", 293, 16, net);
		loadBNGammaParams("stage3d.rebnconv3d", 294, 16, net); 
		loadBNBetaParams("stage3d.rebnconv3d", 294, 16, net);
		loadBNvarParams("stage3d.rebnconv3d", 294, 16, net); 
		loadBNmeanParams("stage3d.rebnconv3d", 294, 16, net);
		
		/************************** RSU5 conv2d**********************************/
		loadCNNWeightParams("stage3d.rebnconv2d", 297, 32*16*9, net);
		loadCNNBiasParams("stage3d.rebnconv2d", 297, 16, net);
		loadBNGammaParams("stage3d.rebnconv2d", 298, 16, net); 
		loadBNBetaParams("stage3d.rebnconv2d", 298, 16, net);
		loadBNvarParams("stage3d.rebnconv2d", 298, 16, net); 
		loadBNmeanParams("stage3d.rebnconv2d", 298, 16, net);
		
		/************************** RSU5 conv1d**********************************/
		loadCNNWeightParams("stage3d.rebnconv1d", 301, 32*64*9, net);
		loadCNNBiasParams("stage3d.rebnconv1d", 301, 64, net);
		loadBNGammaParams("stage3d.rebnconv1d", 302, 64, net); 
		loadBNBetaParams("stage3d.rebnconv1d", 302, 64, net);
		loadBNvarParams("stage3d.rebnconv1d", 302, 64, net); 
		loadBNmeanParams("stage3d.rebnconv1d", 302, 64, net);

		/************************** RSU6 convin**********************************/
		loadCNNWeightParams("stage2d.rebnconvin", 305, 128*64*9, net);
		loadCNNBiasParams("stage2d.rebnconvin", 305, 64, net);
		loadBNGammaParams("stage2d.rebnconvin", 306, 64, net); 
		loadBNBetaParams("stage2d.rebnconvin", 306, 64, net);
		loadBNvarParams("stage2d.rebnconvin", 306, 64, net); 
		loadBNmeanParams("stage2d.rebnconvin", 306, 64, net);

		/************************** RSU6 conv1**********************************/
		loadCNNWeightParams("stage2d.rebnconv1", 308, 64*16*9, net);
		loadCNNBiasParams("stage2d.rebnconv1", 308, 16, net);
		loadBNGammaParams("stage2d.rebnconv1", 309, 16, net); 
		loadBNBetaParams("stage2d.rebnconv1", 309, 16, net);
		loadBNvarParams("stage2d.rebnconv1", 309, 16, net); 
		loadBNmeanParams("stage2d.rebnconv1", 309, 16, net);

		/************************** RSU6 conv2**********************************/
		loadCNNWeightParams("stage2d.rebnconv2", 312, 16*16*9, net);
		loadCNNBiasParams("stage2d.rebnconv2", 312, 16, net);
		loadBNGammaParams("stage2d.rebnconv2", 313, 16, net); 
		loadBNBetaParams("stage2d.rebnconv2", 313, 16, net);
		loadBNvarParams("stage2d.rebnconv2", 313, 16, net); 
		loadBNmeanParams("stage2d.rebnconv2", 313, 16, net);

		/************************** RSU6 conv3**********************************/
		loadCNNWeightParams("stage2d.rebnconv3", 316, 16*16*9, net);
		loadCNNBiasParams("stage2d.rebnconv3", 316, 16, net);
		loadBNGammaParams("stage2d.rebnconv3", 317, 16, net); 
		loadBNBetaParams("stage2d.rebnconv3", 317, 16, net);
		loadBNvarParams("stage2d.rebnconv3", 317, 16, net); 
		loadBNmeanParams("stage2d.rebnconv3", 317, 16, net);

		/************************** RSU6 conv4**********************************/
		loadCNNWeightParams("stage2d.rebnconv4", 320, 16*16*9, net);
		loadCNNBiasParams("stage2d.rebnconv4", 320, 16, net);
		loadBNGammaParams("stage2d.rebnconv4", 321, 16, net); 
		loadBNBetaParams("stage2d.rebnconv4", 321, 16, net);
		loadBNvarParams("stage2d.rebnconv4", 321, 16, net); 
		loadBNmeanParams("stage2d.rebnconv4", 321, 16, net);

		/************************** RSU6 conv5**********************************/
		loadCNNWeightParams("stage2d.rebnconv5", 324, 16*16*9, net);
		loadCNNBiasParams("stage2d.rebnconv5", 324, 16, net);
		loadBNGammaParams("stage2d.rebnconv5", 325, 16, net); 
		loadBNBetaParams("stage2d.rebnconv5", 325, 16, net);
		loadBNvarParams("stage2d.rebnconv5", 325, 16, net); 
		loadBNmeanParams("stage2d.rebnconv5", 325, 16, net);

		/************************** RSU6 conv6**********************************/
		loadCNNWeightParams("stage2d.rebnconv6", 327, 16*16*9, net);
		loadCNNBiasParams("stage2d.rebnconv6", 327, 16, net);
		loadBNGammaParams("stage2d.rebnconv6", 328, 16, net); 
		loadBNBetaParams("stage2d.rebnconv6", 328, 16, net);
		loadBNvarParams("stage2d.rebnconv6", 328, 16, net); 
		loadBNmeanParams("stage2d.rebnconv6", 328, 16, net);

		/************************** RSU6 conv5d**********************************/
		loadCNNWeightParams("stage2d.rebnconv5d", 330, 32*16*9, net);
		loadCNNBiasParams("stage2d.rebnconv5d", 330, 16, net);
		loadBNGammaParams("stage2d.rebnconv5d", 331, 16, net); 
		loadBNBetaParams("stage2d.rebnconv5d", 331, 16, net);
		loadBNvarParams("stage2d.rebnconv5d", 331, 16, net); 
		loadBNmeanParams("stage2d.rebnconv5d", 331, 16, net);

		/************************** RSU6 conv4d**********************************/
		loadCNNWeightParams("stage2d.rebnconv4d", 334, 32*16*9, net);
		loadCNNBiasParams("stage2d.rebnconv4d", 334, 16, net);
		loadBNGammaParams("stage2d.rebnconv4d", 335, 16, net); 
		loadBNBetaParams("stage2d.rebnconv4d", 335, 16, net);
		loadBNvarParams("stage2d.rebnconv4d", 335, 16, net); 
		loadBNmeanParams("stage2d.rebnconv4d", 335, 16, net);

		/************************** RSU6 conv3d**********************************/
		loadCNNWeightParams("stage2d.rebnconv3d", 338, 32*16*9, net);
		loadCNNBiasParams("stage2d.rebnconv3d", 338, 16, net);
		loadBNGammaParams("stage2d.rebnconv3d", 339, 16, net); 
		loadBNBetaParams("stage2d.rebnconv3d", 339, 16, net);
		loadBNvarParams("stage2d.rebnconv3d", 339, 16, net); 
		loadBNmeanParams("stage2d.rebnconv3d", 339, 16, net);

		/************************** RSU6 conv2d**********************************/
		loadCNNWeightParams("stage2d.rebnconv2d", 342, 32*16*9, net);
		loadCNNBiasParams("stage2d.rebnconv2d", 342, 16, net);
		loadBNGammaParams("stage2d.rebnconv2d", 343, 16, net);
		loadBNBetaParams("stage2d.rebnconv2d", 343, 16, net);
		loadBNvarParams("stage2d.rebnconv2d", 343, 16, net);
		loadBNmeanParams("stage2d.rebnconv2d", 343, 16, net);

		/************************** RSU6 conv1d**********************************/
		loadCNNWeightParams("stage2d.rebnconv1d", 346, 32*64*9, net);
		loadCNNBiasParams("stage2d.rebnconv1d", 346, 64, net);
		loadBNGammaParams("stage2d.rebnconv1d", 347, 64, net);
		loadBNBetaParams("stage2d.rebnconv1d", 347, 64, net);
		loadBNvarParams("stage2d.rebnconv1d", 347, 64, net);
		loadBNmeanParams("stage2d.rebnconv1d", 347, 64, net);

		
		/************************** RSU7 convin**********************************/
		loadCNNWeightParams("stage1d.rebnconvin", 350, 128*64*9, net);
		loadCNNBiasParams("stage1d.rebnconvin", 350, 64, net);
		loadBNGammaParams("stage1d.rebnconvin", 351, 64, net); 
		loadBNBetaParams("stage1d.rebnconvin", 351, 64, net);
		loadBNvarParams("stage1d.rebnconvin", 351, 64, net); 
		loadBNmeanParams("stage1d.rebnconvin", 351, 64, net);
		/************************** RSU7 conv1**********************************/
		loadCNNWeightParams("stage1d.rebnconv1", 353, 64*16*9, net);
		loadCNNBiasParams("stage1d.rebnconv1", 353, 16, net);
		loadBNGammaParams("stage1d.rebnconv1", 354, 16, net); 
		loadBNBetaParams("stage1d.rebnconv1", 354, 16, net);
		loadBNvarParams("stage1d.rebnconv1", 354, 16, net); 
		loadBNmeanParams("stage1d.rebnconv1", 354, 16, net);
		/************************** RSU7 conv2**********************************/
		loadCNNWeightParams("stage1d.rebnconv2", 357, 16*16*9, net);
		loadCNNBiasParams("stage1d.rebnconv2", 357, 16, net);
		loadBNGammaParams("stage1d.rebnconv2", 358, 16, net); 
		loadBNBetaParams("stage1d.rebnconv2", 358, 16, net);
		loadBNvarParams("stage1d.rebnconv2", 358, 16, net); 
		loadBNmeanParams("stage1d.rebnconv2", 358, 16, net);
		/************************** RSU7 conv3**********************************/
		loadCNNWeightParams("stage1d.rebnconv3", 361, 16*16*9, net);
		loadCNNBiasParams("stage1d.rebnconv3", 361, 16, net);
		loadBNGammaParams("stage1d.rebnconv3", 362, 16, net); 
		loadBNBetaParams("stage1d.rebnconv3", 362, 16, net);
		loadBNvarParams("stage1d.rebnconv3", 362, 16, net); 
		loadBNmeanParams("stage1d.rebnconv3", 362, 16, net);
		/************************** RSU7 conv4**********************************/
		loadCNNWeightParams("stage1d.rebnconv4", 365, 16*16*9, net);
		loadCNNBiasParams("stage1d.rebnconv4", 365, 16, net);
		loadBNGammaParams("stage1d.rebnconv4", 366, 16, net); 
		loadBNBetaParams("stage1d.rebnconv4", 366, 16, net);
		loadBNvarParams("stage1d.rebnconv4", 366, 16, net); 
		loadBNmeanParams("stage1d.rebnconv4", 366, 16, net);
		/************************** RSU7 conv5**********************************/
		loadCNNWeightParams("stage1d.rebnconv5", 369, 16*16*9, net);
		loadCNNBiasParams("stage1d.rebnconv5", 369, 16, net);
		loadBNGammaParams("stage1d.rebnconv5", 370, 16, net); 
		loadBNBetaParams("stage1d.rebnconv5", 370, 16, net);
		loadBNvarParams("stage1d.rebnconv5", 370, 16, net); 
		loadBNmeanParams("stage1d.rebnconv5", 370, 16, net);
		/************************** RSU7 conv6**********************************/
		loadCNNWeightParams("stage1d.rebnconv6", 373, 16*16*9, net);
		loadCNNBiasParams("stage1d.rebnconv6", 373, 16, net);
		loadBNGammaParams("stage1d.rebnconv6", 374, 16, net); 
		loadBNBetaParams("stage1d.rebnconv6", 374, 16, net);
		loadBNvarParams("stage1d.rebnconv6", 374, 16, net); 
		loadBNmeanParams("stage1d.rebnconv6", 374, 16, net);
		/************************** RSU7 conv7**********************************/
		loadCNNWeightParams("stage1d.rebnconv7", 376, 16*16*9, net);
		loadCNNBiasParams("stage1d.rebnconv7", 376, 16, net);
		loadBNGammaParams("stage1d.rebnconv7", 377, 16, net); 
		loadBNBetaParams("stage1d.rebnconv7", 377, 16, net);
		loadBNvarParams("stage1d.rebnconv7", 377, 16, net); 
		loadBNmeanParams("stage1d.rebnconv7", 377, 16, net);
		/************************** RSU7 conv6d**********************************/
		loadCNNWeightParams("stage1d.rebnconv6d", 379, 32*16*9, net);
		loadCNNBiasParams("stage1d.rebnconv6d", 379, 16, net);
		loadBNGammaParams("stage1d.rebnconv6d", 380, 16, net); 
		loadBNBetaParams("stage1d.rebnconv6d", 380, 16, net);
		loadBNvarParams("stage1d.rebnconv6d", 380, 16, net); 
		loadBNmeanParams("stage1d.rebnconv6d", 380, 16, net);
		/************************** RSU7 conv5d**********************************/
		loadCNNWeightParams("stage1d.rebnconv5d", 383, 32*16*9, net);
		loadCNNBiasParams("stage1d.rebnconv5d", 383, 16, net);
		loadBNGammaParams("stage1d.rebnconv5d", 384, 16, net); 
		loadBNBetaParams("stage1d.rebnconv5d", 384, 16, net);
		loadBNvarParams("stage1d.rebnconv5d", 384, 16, net); 
		loadBNmeanParams("stage1d.rebnconv5d", 384, 16, net);
		/************************** RSU7 conv4d**********************************/
		loadCNNWeightParams("stage1d.rebnconv4d", 387, 32*16*9, net);
		loadCNNBiasParams("stage1d.rebnconv4d", 387, 16, net);
		loadBNGammaParams("stage1d.rebnconv4d", 388, 16, net); 
		loadBNBetaParams("stage1d.rebnconv4d", 388, 16, net);
		loadBNvarParams("stage1d.rebnconv4d", 388, 16, net); 
		loadBNmeanParams("stage1d.rebnconv4d", 388, 16, net);
		/************************** RSU7 conv3d**********************************/
		loadCNNWeightParams("stage1d.rebnconv3d", 391, 32*16*9, net);
		loadCNNBiasParams("stage1d.rebnconv3d", 391, 16, net);
		loadBNGammaParams("stage1d.rebnconv3d", 392, 16, net); 
		loadBNBetaParams("stage1d.rebnconv3d", 392, 16, net);
		loadBNvarParams("stage1d.rebnconv3d", 392, 16, net); 
		loadBNmeanParams("stage1d.rebnconv3d", 392, 16, net);
		/************************** RSU7 conv2d**********************************/
		loadCNNWeightParams("stage1d.rebnconv2d", 395, 32*16*9, net);
		loadCNNBiasParams("stage1d.rebnconv2d", 395, 16, net);
		loadBNGammaParams("stage1d.rebnconv2d", 396, 16, net); 
		loadBNBetaParams("stage1d.rebnconv2d", 396, 16, net);
		loadBNvarParams("stage1d.rebnconv2d", 396, 16, net); 
		loadBNmeanParams("stage1d.rebnconv2d", 396, 16, net);
		/************************** RSU7 conv1d**********************************/
		loadCNNWeightParams("stage1d.rebnconv1d", 399, 32*64*9, net);
		loadCNNBiasParams("stage1d.rebnconv1d", 399, 64, net);
		loadBNGammaParams("stage1d.rebnconv1d", 400, 64, net); 
		loadBNBetaParams("stage1d.rebnconv1d", 400, 64, net);
		loadBNvarParams("stage1d.rebnconv1d", 400, 64, net); 
		loadBNmeanParams("stage1d.rebnconv1d", 400, 64, net);

		// /**************************SIDE**********************************/
		// loadCNNWeightParams("side1", 402, 64*1*9, net);
		// loadCNNBiasParams("side1", 402, 1, net);
		// loadCNNWeightParams("side2", 403, 64*1*9, net);
		// loadCNNBiasParams("side2", 403, 1, net);
		loadCNNWeightParams("side3", 305, 64*1*9, net);   //提前计算
		loadCNNBiasParams("side3", 305, 1, net);
		// loadCNNWeightParams("side3", 405, 128*1*9, net);
		// loadCNNBiasParams("side3", 405, 1, net);
		// loadCNNWeightParams("side3", 407, 256*1*9, net);
		// loadCNNBiasParams("side3", 407, 1, net);
		// loadCNNWeightParams("side4", 409, 512*1*9, net);
		// loadCNNBiasParams("side4", 409, 1, net);
		// loadCNNWeightParams("side5", 411, 512*1*9, net);
		// loadCNNBiasParams("side5", 411, 1, net);
		// loadCNNWeightParams("outconv", 413, 6*1*1, net);
		// loadCNNBiasParams("outconv", 413, 1, net);
	}
	else 
		error("Preloading network error");

	cout << "Preloading completed..." << endl;
}
void loadCNNWeightParams(const std::string& weight_name, int layer_num, int param_size, NeuralNetwork* net)
{
    std::string canshu_path = "files/preload/U2netNet/all_canshup/";
    std::string path_weight1 = canshu_path + weight_name + ".conv_s1.weight_" + std::to_string(partyNum);
    std::string path_weight2 = canshu_path + weight_name + ".conv_s1.weight_" + std::to_string(nextParty(partyNum));

	if (weight_name.substr(0, 4) == "side" || weight_name.substr(0, 2) == "ou")
	{
		path_weight1 = canshu_path + weight_name + ".weight_"+ std::to_string(partyNum);
		path_weight2 = canshu_path + weight_name + ".weight_"+ std::to_string(nextParty(partyNum));
		//cout<<layer_num<<endl;
	}
	std::ifstream f_weight1(path_weight1), f_weight2(path_weight2);
	// 检查文件是否存在
    if (!f_weight1.good() || !f_weight2.good())
    {
        std::cout << "错误: 无法打开权重文件: " << path_weight1 << " 和 " << path_weight2 << std::endl;
        return;
    }
    for (int i = 0; i < param_size; ++i)
    {
        float temp_next, temp_prev;
        f_weight1 >> temp_next;f_weight2 >> temp_prev;
        (*((CNNLayer*)net->layers[layer_num])->getWeights())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
    }

    f_weight1.close();f_weight2.close();
}
void loadCNNBiasParams(const std::string& cnnbias_name, int layer_num, int param_size, NeuralNetwork* net)
{
    std::string canshu_path = "files/preload/U2netNet/all_canshup/";
    std::string path_cnnbias1 = canshu_path + cnnbias_name + ".conv_s1.bias_" + std::to_string(partyNum);
    std::string path_cnnbias2 = canshu_path + cnnbias_name + ".conv_s1.bias_" + std::to_string(nextParty(partyNum));

	if (cnnbias_name.substr(0, 4) == "side" || cnnbias_name.substr(0, 2) == "ou")
	{
		path_cnnbias1 = canshu_path + cnnbias_name + ".bias_"+ std::to_string(partyNum);
		path_cnnbias2 = canshu_path + cnnbias_name + ".bias_"+ std::to_string(nextParty(partyNum));
		//cout<<layer_num<<endl;
	}
	
	std::ifstream f_cnnbias1(path_cnnbias1), f_cnnbias2(path_cnnbias2);
	// 检查文件是否存在
    if (!f_cnnbias1.good() || !f_cnnbias2.good())
    {
        std::cout << "错误: 无法打开权重文件: " <<path_cnnbias1 << " 和 " << path_cnnbias2 << std::endl;
        return;
    }
    for (int i = 0; i < param_size; ++i)
    {
        float temp_next, temp_prev;
        f_cnnbias1 >> temp_next; f_cnnbias2 >> temp_prev;
        (*((CNNLayer*)net->layers[layer_num])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
    }

    f_cnnbias1.close(); f_cnnbias2.close();
}
void loadBNGammaParams(const std::string& gamma_name, int layer_num, int param_size, NeuralNetwork* net)
{
    std::string canshu_path = "files/preload/U2netNet/all_canshup/";
    std::string path_gamma1 = canshu_path + gamma_name + ".bn_s1.weight_" + std::to_string(partyNum);
    std::string path_gamma2 = canshu_path + gamma_name + ".bn_s1.weight_" + std::to_string(nextParty(partyNum));
    std::ifstream f_gamma1(path_gamma1), f_gamma2(path_gamma2);

	// 检查文件是否存在
    if (!f_gamma1.good() || !f_gamma2.good())
    {
        std::cout << "错误: 无法打开权重文件: " <<path_gamma1 << " 和 " << path_gamma2 << std::endl;
        return;
    }
    for (int i = 0; i < param_size; ++i)
    {
        float temp_next, temp_prev;
        f_gamma1 >> temp_next; f_gamma2 >> temp_prev;
        (*((BNLayer*)net->layers[layer_num])->getgamma())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
    }

    f_gamma1.close(); f_gamma2.close();
}
void loadBNBetaParams(const std::string& bias_name, int layer_num, int param_size, NeuralNetwork* net)
{
	string canshu_path = "files/preload/U2netNet/all_canshup/";
    std::string path_beta1 = canshu_path + bias_name + ".bn_s1.bias_" + to_string(partyNum);
    std::string path_beta2 = canshu_path + bias_name + ".bn_s1.bias_" + to_string(nextParty(partyNum));
    std::ifstream f_beta1(path_beta1), f_beta2(path_beta2);

	// 检查文件是否存在
    if (!f_beta1.good() || !f_beta2.good())
    {
        std::cout << "错误: 无法打开权重文件: " <<path_beta1 << " 和 " << path_beta2 << std::endl;
        return;
    }
    for (int i = 0; i < param_size; ++i)
    {
        float temp_next, temp_prev;
        f_beta1 >> temp_next; f_beta2 >> temp_prev;
        (*((BNLayer*)net->layers[layer_num])->getbeta())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
    }

    f_beta1.close(); f_beta2.close();
}
void loadBNvarParams(const std::string& var_name, int layer_num, int param_size, NeuralNetwork* net)
{
    std::string canshu_path = "files/preload/U2netNet/all_canshup1/";
    std::string path_var1 = canshu_path + var_name + ".bn_s1.running_var_" + std::to_string(partyNum);
    std::string path_var2 = canshu_path + var_name + ".bn_s1.running_var_" + std::to_string(nextParty(partyNum));
    std::ifstream f_var1(path_var1), f_var2(path_var2);

	// 检查文件是否存在
    if (!f_var1.good() || !f_var2.good())
    {
        std::cout << "错误: 无法打开权重文件: " <<path_var1 << " 和 " << path_var2 << std::endl;
        return;
    }
    for (int i = 0; i < param_size; ++i)
    {
        float temp_next, temp_prev;
        f_var1 >> temp_next; f_var2 >> temp_prev;
        (*((BNLayer*)net->layers[layer_num])->getvar())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
    }

    f_var1.close(); f_var2.close();
}
void loadBNmeanParams(const std::string& mean_name, int layer_num, int param_size, NeuralNetwork* net)
{
    std::string canshu_path = "files/preload/U2netNet/all_canshup/";
    std::string path_mean1 = canshu_path + mean_name + ".bn_s1.running_mean_" + std::to_string(partyNum);
    std::string path_mean2 = canshu_path + mean_name + ".bn_s1.running_mean_" + std::to_string(nextParty(partyNum));
    std::ifstream f_mean1(path_mean1), f_mean2(path_mean2);

	// 检查文件是否存在
    if (!f_mean1.good() || !f_mean2.good())
    {
        std::cout << "错误: 无法打开权重文件: " <<path_mean1 << " 和 " << path_mean2 << std::endl;
        return;
    }
    for (int i = 0; i < param_size; ++i)
    {
        float temp_next, temp_prev;
        f_mean1 >> temp_next; f_mean2 >> temp_prev;
        (*((BNLayer*)net->layers[layer_num])->getmean())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
    }

    f_mean1.close(); f_mean2.close();
}


void loadData(string net, string dataset)
{
	if (dataset.compare("MNIST") == 0)
	{	
		LARGE_NETWORK = true;
		INPUT_SIZE = 320 * 320;	//改测试
		LAST_LAYER_SIZE = INPUT_SIZE*4;
		TRAINING_DATA_SIZE = 1;
		TEST_DATA_SIZE = 1;
		
	}
	else if (dataset.compare("CIFAR10") == 0)
	{
		LARGE_NETWORK = false;
		if (net.compare("AlexNet") == 0)
		{
			INPUT_SIZE = 33*33*3;
			LAST_LAYER_SIZE = 10;
			TRAINING_DATA_SIZE = 8;
			TEST_DATA_SIZE = 8;			
		}
		else if (net.compare("VGG16") == 0)
		{
			INPUT_SIZE = 32*32*3;
			LAST_LAYER_SIZE = 10;
			TRAINING_DATA_SIZE = 8;
			TEST_DATA_SIZE = 8;	
		}
		else
			assert(false && "Only AlexNet and VGG16 supported on CIFAR10");
	}
	else if (dataset.compare("ImageNet") == 0)
	{
		LARGE_NETWORK = true;
		//大型网络

		//https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637
		//https://medium.com/@RaghavPrabhu/cnn-architectures-lenet-alexnet-vgg-googlenet-and-resnet-7c81c017b848
		//https://neurohive.io/en/popular-networks/vgg16/

		//Tiny ImageNet
		//http://cs231n.stanford.edu/reports/2017/pdfs/930.pdf
		//http://cs231n.stanford.edu/reports/2017/pdfs/931.pdf
		if (net.compare("AlexNet") == 0)
		{
			INPUT_SIZE = 3*4*4;
			LAST_LAYER_SIZE = 200;
			TRAINING_DATA_SIZE = 8;
			TEST_DATA_SIZE = 8;			
		}
		else if (net.compare("VGG16") == 0)
		{
			INPUT_SIZE = 64*64*3;
			LAST_LAYER_SIZE = 200;
			TRAINING_DATA_SIZE = 8;
			TEST_DATA_SIZE = 8;			
		}
		else if (net.compare("U2netNet") == 0)
		{
			//INPUT_SIZE = 56*56*3;
			INPUT_SIZE = 320*320*3;
			LAST_LAYER_SIZE = INPUT_SIZE *4;
			TRAINING_DATA_SIZE = 1;
			TEST_DATA_SIZE = 1;	
		}
		else
			assert(false && "Only AlexNet and VGG16 supported on ImageNet");
	}
	else
		assert(false && "Only MNIST, CIFAR10, and ImageNet supported");


	string filename_train_data_next, filename_train_data_prev;
	string filename_test_data_next, filename_test_data_prev;
	string filename_train_labels_next, filename_train_labels_prev;
	string filename_test_labels_next, filename_test_labels_prev;
	
	// modified to let each party holding a share of data
	if (partyNum == PARTY_A)
	{
		filename_train_data_next = "files/train_data_A";
		filename_train_data_prev = "files/train_data_B";
		filename_test_data_next = "files/test_data_A";
		filename_test_data_prev = "files/test_data_B";
		filename_train_labels_next = "files/train_labels_A";
		filename_train_labels_prev = "files/train_labels_B";
		filename_test_labels_next = "files/test_labels_A";
		filename_test_labels_prev = "files/test_labels_B";
	}

	if (partyNum == PARTY_B)
	{
		filename_train_data_next = "files/train_data_B";
		filename_train_data_prev = "files/train_data_C";
		filename_test_data_next = "files/test_data_B";
		filename_test_data_prev = "files/test_data_C";
		filename_train_labels_next = "files/train_labels_B";
		filename_train_labels_prev = "files/train_labels_C";
		filename_test_labels_next = "files/test_labels_B";
		filename_test_labels_prev = "files/test_labels_C";
	}

	if (partyNum == PARTY_C)
	{
		filename_train_data_next = "files/train_data_C";
		filename_train_data_prev = "files/train_data_A";
		filename_test_data_next = "files/test_data_C";
		filename_test_data_prev = "files/test_data_A";
		filename_train_labels_next = "files/train_labels_C";
		filename_train_labels_prev = "files/train_labels_A";
		filename_test_labels_next = "files/test_labels_C";
		filename_test_labels_prev = "files/test_labels_A";
	}	

	float temp_next = 0, temp_prev = 0;
	ifstream f_next(filename_train_data_next);
	ifstream f_prev(filename_train_data_prev);
	for (int i = 0; i < TRAINING_DATA_SIZE * INPUT_SIZE; ++i)
	{
		f_next >> temp_next; f_prev >> temp_prev;
		trainData.push_back(std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev)));
	}
	f_next.close(); f_prev.close();

	ifstream g_next(filename_train_labels_next);
	ifstream g_prev(filename_train_labels_prev);
	for (int i = 0; i < TRAINING_DATA_SIZE * LAST_LAYER_SIZE; ++i)
	{
		g_next >> temp_next; g_prev >> temp_prev;
		trainLabels.push_back(std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev)));
	}
	g_next.close(); g_prev.close();

	ifstream h_next(filename_test_data_next);
	ifstream h_prev(filename_test_data_prev);
	for (int i = 0; i < TEST_DATA_SIZE * INPUT_SIZE; ++i)
	{
		h_next >> temp_next; h_prev >> temp_prev;
		//std::cout << "temp_next: " << temp_next << ", temp_prev: " << temp_prev << std::endl;
		testData.push_back(std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev)));
	}
	h_next.close(); h_prev.close();

	ifstream k_next(filename_test_labels_next);
	ifstream k_prev(filename_test_labels_prev);
	for (int i = 0; i < TEST_DATA_SIZE * LAST_LAYER_SIZE; ++i)
	{
		k_next >> temp_next; k_prev >> temp_prev;
		testLabels.push_back(std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev)));
	}
	k_next.close(); k_prev.close();		

	cout << "Loading data done....." << endl;
}


void readMiniBatch(NeuralNetwork* net, string phase)
{
	size_t s = trainData.size();
	size_t t = trainLabels.size();

	if (phase == "TRAINING")
	{
		for (int i = 0; i < INPUT_SIZE * MINI_BATCH_SIZE; ++i)
			net->inputData[i] = trainData[(trainDataBatchCounter + i)%s];

		for (int i = 0; i < LAST_LAYER_SIZE * MINI_BATCH_SIZE; ++i)
			net->outputData[i] = trainLabels[(trainLabelsBatchCounter + i)%t];

		trainDataBatchCounter += INPUT_SIZE * MINI_BATCH_SIZE;
		trainLabelsBatchCounter += LAST_LAYER_SIZE * MINI_BATCH_SIZE;
	}

	if (trainDataBatchCounter > s)
		trainDataBatchCounter -= s;

	if (trainLabelsBatchCounter > t)
		trainLabelsBatchCounter -= t;



	size_t p = testData.size();
	size_t q = testLabels.size();

	if (phase == "TESTING")
	{
		for (int i = 0; i < INPUT_SIZE * MINI_BATCH_SIZE; ++i)
			net->inputData[i] = testData[(testDataBatchCounter + i)%p];

		for (int i = 0; i < LAST_LAYER_SIZE * MINI_BATCH_SIZE; ++i)
			net->outputData[i] = testLabels[(testLabelsBatchCounter + i)%q];

		testDataBatchCounter += INPUT_SIZE * MINI_BATCH_SIZE;
		testLabelsBatchCounter += LAST_LAYER_SIZE * MINI_BATCH_SIZE;
	}

	if (testDataBatchCounter > p)
		testDataBatchCounter -= p;

	if (testLabelsBatchCounter > q)
		testLabelsBatchCounter -= q;
}

void printNetwork(NeuralNetwork* net)
{
	for (int i = net->layers.size()-5; i < net->layers.size(); ++i)
		net->layers[i]->printLayer();
	cout << "----------------------------------------------" << endl;  	
}


void selectNetwork(string network, string dataset, string security, NeuralNetConfig* config)
{
	assert(((security.compare("Semi-honest") == 0) or (security.compare("Malicious") == 0)) && 
			"Only Semi-honest or Malicious security allowed");
	SECURITY_TYPE = security;
	loadData(network, dataset);

	if (network.compare("SecureML") == 0)
	{
		assert((dataset.compare("MNIST") == 0) && "SecureML only over MNIST");
		NUM_LAYERS = 6;
		WITH_NORMALIZATION = true;
		FCConfig* l0 = new FCConfig(784, MINI_BATCH_SIZE, 128); 
		ReLUConfig* l1 = new ReLUConfig(128, MINI_BATCH_SIZE);
		FCConfig* l2 = new FCConfig(128, MINI_BATCH_SIZE, 128); 
		ReLUConfig* l3 = new ReLUConfig(128, MINI_BATCH_SIZE);
		FCConfig* l4 = new FCConfig(128, MINI_BATCH_SIZE, 10); 
		ReLUConfig* l5 = new ReLUConfig(10, MINI_BATCH_SIZE);
		// BNConfig* l6 = new BNConfig(10, MINI_BATCH_SIZE);
		config->addLayer(l0);
		config->addLayer(l1);
		config->addLayer(l2);
		config->addLayer(l3);
		config->addLayer(l4);
		config->addLayer(l5);
		// config->addLayer(l6);
	}
	else if (network.compare("Sarda") == 0)
	{
		assert((dataset.compare("MNIST") == 0) && "Sarda only over MNIST");
		NUM_LAYERS = 5;
		WITH_NORMALIZATION = true;
		CNNConfig* l0 = new CNNConfig(28,28,1,5,2,2,0,MINI_BATCH_SIZE);
		ReLUConfig* l1 = new ReLUConfig(980, MINI_BATCH_SIZE);
		FCConfig* l2 = new FCConfig(980, MINI_BATCH_SIZE, 100);
		ReLUConfig* l3 = new ReLUConfig(100, MINI_BATCH_SIZE);
		FCConfig* l4 = new FCConfig(100, MINI_BATCH_SIZE, 10);
		config->addLayer(l0);
		config->addLayer(l1);
		config->addLayer(l2);
		config->addLayer(l3);
		config->addLayer(l4);
	}
	else if (network.compare("MiniONN") == 0)
	{
		assert((dataset.compare("MNIST") == 0) && "MiniONN only over MNIST");
		NUM_LAYERS = 10;
		WITH_NORMALIZATION = true;
		CNNConfig* l0 = new CNNConfig(28,28,1,16,5,1,0,MINI_BATCH_SIZE);
		MaxpoolConfig* l1 = new MaxpoolConfig(24,24,16,2,2,MINI_BATCH_SIZE);
		ReLUConfig* l2 = new ReLUConfig(12*12*16, MINI_BATCH_SIZE);
		CNNConfig* l3 = new CNNConfig(12,12,16,16,5,1,0,MINI_BATCH_SIZE);
		MaxpoolConfig* l4 = new MaxpoolConfig(8,8,16,2,2,MINI_BATCH_SIZE);
		ReLUConfig* l5 = new ReLUConfig(4*4*16, MINI_BATCH_SIZE);
		FCConfig* l6 = new FCConfig(4*4*16, MINI_BATCH_SIZE, 100);
		ReLUConfig* l7 = new ReLUConfig(100, MINI_BATCH_SIZE);
		FCConfig* l8 = new FCConfig(100, MINI_BATCH_SIZE, 10);
		ReLUConfig* l9 = new ReLUConfig(10, MINI_BATCH_SIZE);
		config->addLayer(l0);
		config->addLayer(l1);
		config->addLayer(l2);
		config->addLayer(l3);
		config->addLayer(l4);
		config->addLayer(l5);
		config->addLayer(l6);
		config->addLayer(l7);
		config->addLayer(l8);
		config->addLayer(l9);
	}
	else if (network.compare("LeNet") == 0)     //测试网络
	{
		assert((dataset.compare("MNIST") == 0) && "LeNet only over MNIST");
		NUM_LAYERS = 1;
		WITH_NORMALIZATION = true;
		// CNNConfig* l0 = new CNNConfig(512, 512, 3, 4, 3, 4, 1, MINI_BATCH_SIZE);  //膨胀卷积没有问题
		// // MaxpoolConfig* l1 = new MaxpoolConfig(56, 56, 64, 2, 2, MINI_BATCH_SIZE);
		// BNConfig* l2 = new BNConfig(128 * 128 * 4, MINI_BATCH_SIZE, 4, 128, 128);
		ReLUConfig* l3 = new ReLUConfig( 320 * 320, MINI_BATCH_SIZE);
		// UpsampleConfig* l4 = new UpsampleConfig(256, 256, 3, MINI_BATCH_SIZE, 2); //没问题
		// SigmoidConfig* l5 = new SigmoidConfig(512*512*4, MINI_BATCH_SIZE);

		// SigmoidConfig* Fd1 = new SigmoidConfig(32*32*1, MINI_BATCH_SIZE);
		// MaxpoolConfig* l0 = new MaxpoolConfig(4, 4, 3, 2, 2, MINI_BATCH_SIZE);
		// // UpsampleConfig* l0 = new UpsampleConfig(320, 320, 3, MINI_BATCH_SIZE, 2); //没问题
		// BNConfig* l1 = new BNConfig(320 * 320 * 64, MINI_BATCH_SIZE, 64, 320, 320);
		// ReLUConfig* Fd1 = new ReLUConfig(512 * 512, MINI_BATCH_SIZE);

		// CNNConfig* l3 = new CNNConfig(320, 320, 64, 32, 3, 1, 1, MINI_BATCH_SIZE); 
		// BNConfig* l4 = new BNConfig(320 * 320 * 32, MINI_BATCH_SIZE, 32, 320, 320);
		// ReLUConfig* l5 = new ReLUConfig(320 * 320 * 32, MINI_BATCH_SIZE);

		// MaxpoolConfig* l6 = new MaxpoolConfig(320, 320, 32, 2, 2, MINI_BATCH_SIZE);

		// CNNConfig* l7 = new CNNConfig(160, 160, 32, 32, 3, 1, 1, MINI_BATCH_SIZE);
		// BNConfig* l8 = new BNConfig(160 * 160 * 32, MINI_BATCH_SIZE, 32, 160, 160);
		// ReLUConfig* l9 = new ReLUConfig(160 * 160 * 32, MINI_BATCH_SIZE);
		
		// MaxpoolConfig* l10 = new MaxpoolConfig(160, 160, 32, 2, 2, MINI_BATCH_SIZE);
		// //conv3
		// CNNConfig* l11 = new CNNConfig(80, 80, 32, 32, 3, 1, 1, MINI_BATCH_SIZE);
		// BNConfig* l12 = new BNConfig(80 * 80 * 32, MINI_BATCH_SIZE, 32, 80 , 80 );
		// ReLUConfig* l13 = new ReLUConfig(80 * 80  * 32, MINI_BATCH_SIZE);

		// MaxpoolConfig* l14 = new MaxpoolConfig(80, 80, 32, 2, 2, MINI_BATCH_SIZE);

		// config->addLayer(l0);
		// // config->addLayer(l1);
		// config->addLayer(l2);
		config->addLayer(l3);
		// config->addLayer(l4);
		// config->addLayer(l5);
		// config->addLayer(l6);
		// config->addLayer(l7);
		// config->addLayer(l8);
		// config->addLayer(l9);
		// config->addLayer(l10);
		// config->addLayer(l11);
		// config->addLayer(l12);
		// config->addLayer(l13);
		// config->addLayer(l14);

	}
	// size_t B 	= conf.batchSize;
	// size_t iw 	= conf.imageWidth;
	// size_t ih 	= conf.imageHeight;
	// size_t f 	= conf.filterSize;
	// size_t Din 	= conf.inputFeatures;
	// size_t Dout = conf.filters;
	// size_t P 	= conf.padding;
	// size_t S 	= conf.stride;
	//测试网络
	else if (network.compare("U2netNet") == 0)
	{
		NUM_LAYERS = 307;  //421  307
    	WITH_NORMALIZATION = true;

    	CNNConfig* rsu7_l0 = new CNNConfig(320, 320, 3, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l1 = new BNConfig(320 * 320 * 64, MINI_BATCH_SIZE, 64, 320, 320);
		ReLUConfig* rsu7_l2 = new ReLUConfig(320 * 320 * 64, MINI_BATCH_SIZE);

		//conv1
		CNNConfig* rsu7_l3 = new CNNConfig(320, 320, 64, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l4 = new BNConfig(320 * 320 * 16, MINI_BATCH_SIZE, 16, 320, 320);
		ReLUConfig* rsu7_l5 = new ReLUConfig(320 * 320 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu7_l6 = new MaxpoolConfig(320, 320, 16, 2, 2, MINI_BATCH_SIZE);
		// conv2
		CNNConfig* rsu7_l7 = new CNNConfig(160, 160, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l8 = new BNConfig(160 * 160 * 16, MINI_BATCH_SIZE, 16, 160, 160);
		ReLUConfig* rsu7_l9 = new ReLUConfig(160 * 160 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu7_l10 = new MaxpoolConfig(160, 160, 16, 2, 2, MINI_BATCH_SIZE);
		// conv3
		CNNConfig* rsu7_l11 = new CNNConfig(80, 80, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l12 = new BNConfig(80 * 80 * 16, MINI_BATCH_SIZE, 16, 80 , 80 );
		ReLUConfig* rsu7_l13 = new ReLUConfig(80 * 80  * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu7_l14 = new MaxpoolConfig(80, 80, 16, 2, 2, MINI_BATCH_SIZE);
		// conv4
		CNNConfig* rsu7_l15 = new CNNConfig(40, 40, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l16 = new BNConfig(40 * 40 * 16, MINI_BATCH_SIZE, 16, 40 , 40 );
		ReLUConfig* rsu7_l17 = new ReLUConfig(40 * 40  * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu7_l18 = new MaxpoolConfig(40, 40, 16, 2, 2, MINI_BATCH_SIZE);
		// conv5
		CNNConfig* rsu7_l19 = new CNNConfig(20, 20, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l20 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20 , 20 );
		ReLUConfig* rsu7_l21 = new ReLUConfig(20 * 20  * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu7_l22 = new MaxpoolConfig(20, 20, 16, 2, 2, MINI_BATCH_SIZE);
		// conv6
		CNNConfig* rsu7_l23 = new CNNConfig(10, 10, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l24 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10 , 10 );
		ReLUConfig* rsu7_l25 = new ReLUConfig(10 * 10  * 16, MINI_BATCH_SIZE);

		// conv7
		CNNConfig* rsu7_l26 = new CNNConfig(10, 10, 16, 16, 3, 1, 2, MINI_BATCH_SIZE);
		BNConfig* rsu7_l27 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10 , 10 );
		ReLUConfig* rsu7_l28 = new ReLUConfig(10 * 10  * 16, MINI_BATCH_SIZE);

		// conv6d
		CNNConfig* rsu7_l29 = new CNNConfig(10, 10, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l30 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10 , 10 );
		ReLUConfig* rsu7_l31 = new ReLUConfig(10 * 10  * 16, MINI_BATCH_SIZE);

		UpsampleConfig* rsu7_l32 = new UpsampleConfig(10, 10, 16, MINI_BATCH_SIZE, 2);

		// conv5d
		CNNConfig* rsu7_l33 = new CNNConfig(20, 20, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l34 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20 , 20 );
		ReLUConfig* rsu7_l35 = new ReLUConfig(20 * 20  * 16, MINI_BATCH_SIZE);

		UpsampleConfig* rsu7_l36 = new UpsampleConfig(20, 20, 16, MINI_BATCH_SIZE, 2);

		// conv4d
		CNNConfig* rsu7_l37 = new CNNConfig(40, 40, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l38 = new BNConfig(40 * 40 * 16, MINI_BATCH_SIZE, 16, 40 , 40);
		ReLUConfig* rsu7_l39 = new ReLUConfig(40 * 40  * 16, MINI_BATCH_SIZE);

		UpsampleConfig* rsu7_l40 = new UpsampleConfig(40, 40, 16, MINI_BATCH_SIZE, 2);

		//conv3d
		CNNConfig* rsu7_l41 = new CNNConfig(80, 80, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l42 = new BNConfig(80 * 80 * 16, MINI_BATCH_SIZE, 16, 80 , 80);
		ReLUConfig* rsu7_l43 = new ReLUConfig(80 * 80  * 16, MINI_BATCH_SIZE);	

		UpsampleConfig* rsu7_l44 = new UpsampleConfig(80, 80, 16, MINI_BATCH_SIZE, 2);

		// conv2d
		CNNConfig* rsu7_l45 = new CNNConfig(160, 160, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l46 = new BNConfig(160 * 160 * 16, MINI_BATCH_SIZE, 16, 160 , 160);
		ReLUConfig* rsu7_l47 = new ReLUConfig(160 * 160  * 16, MINI_BATCH_SIZE);

		UpsampleConfig* rsu7_l48 = new UpsampleConfig(160, 160, 16, MINI_BATCH_SIZE, 2);

		// conv1d
		CNNConfig* rsu7_l49 = new CNNConfig(320, 320, 2 * 16, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l50 = new BNConfig(320 * 320 * 64, MINI_BATCH_SIZE, 64, 320 , 320);
		ReLUConfig* rsu7_l51 = new ReLUConfig(320 * 320  * 64, MINI_BATCH_SIZE);
		// RSU7 return rsu7_l51+rsu7_l2

		MaxpoolConfig* rsu7_l52 = new MaxpoolConfig(320, 320, 64, 2, 2, MINI_BATCH_SIZE);
		//stage1 end
		
		//convin
		CNNConfig* rsu6_l53 = new CNNConfig(160, 160, 64, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l54 = new BNConfig(160 * 160 * 64, MINI_BATCH_SIZE, 64, 160, 160);
		ReLUConfig* rsu6_l55 = new ReLUConfig(160 * 160 * 64, MINI_BATCH_SIZE);

		//conv1
		CNNConfig* rsu6_l56 = new CNNConfig(160, 160, 64, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l57 = new BNConfig(160 * 160 * 16, MINI_BATCH_SIZE, 16, 160, 160);
		ReLUConfig* rsu6_l58 = new ReLUConfig(160 * 160 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu6_l59 = new MaxpoolConfig(160, 160, 16, 2, 2, MINI_BATCH_SIZE);

		//conv2
		CNNConfig* rsu6_l60 = new CNNConfig(80, 80, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l61 = new BNConfig(80 * 80 * 16, MINI_BATCH_SIZE, 16, 80, 80);
		ReLUConfig* rsu6_l62 = new ReLUConfig(80 * 80 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu6_l63 = new MaxpoolConfig(80, 80, 16, 2, 2, MINI_BATCH_SIZE);

		//conv3
		CNNConfig* rsu6_l64 = new CNNConfig(40, 40, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l65 = new BNConfig(40 * 40 * 16, MINI_BATCH_SIZE, 16, 40, 40);
		ReLUConfig* rsu6_l66 = new ReLUConfig(40 * 40 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu6_l67 = new MaxpoolConfig(40, 40, 16, 2, 2, MINI_BATCH_SIZE);

		//conv4
		CNNConfig* rsu6_l68 = new CNNConfig(20, 20, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l69 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20, 20);
		ReLUConfig* rsu6_l70 = new ReLUConfig(20 * 20 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu6_l71 = new MaxpoolConfig(20, 20, 16, 2, 2, MINI_BATCH_SIZE);

		//conv5
		CNNConfig* rsu6_l72 = new CNNConfig(10, 10, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l73 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10, 10);
		ReLUConfig* rsu6_l74 = new ReLUConfig(10 * 10 * 16, MINI_BATCH_SIZE);

		//conv6
		CNNConfig* rsu6_l75 = new CNNConfig(10, 10, 16, 16, 3, 1, 2, MINI_BATCH_SIZE);
		BNConfig* rsu6_l76 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10, 10);
		ReLUConfig* rsu6_l77 = new ReLUConfig(10 * 10 * 16, MINI_BATCH_SIZE);

		//conv5d
		CNNConfig* rsu6_l78 = new CNNConfig(10, 10, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l79 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10 , 10 );
		ReLUConfig* rsu6_l80 = new ReLUConfig(10 * 10  * 16, MINI_BATCH_SIZE);

		UpsampleConfig* rsu6_l81 = new UpsampleConfig(10, 10, 16, MINI_BATCH_SIZE, 2);

		//conv4d
		CNNConfig* rsu6_l82 = new CNNConfig(20, 20, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l83 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20 , 20 );
		ReLUConfig* rsu6_l84 = new ReLUConfig(20 * 20  * 16, MINI_BATCH_SIZE);

		UpsampleConfig* rsu6_l85 = new UpsampleConfig(20, 20, 16, MINI_BATCH_SIZE, 2);

		//conv3d 
		CNNConfig* rsu6_l86 = new CNNConfig(40, 40, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l87 = new BNConfig(40 * 40 * 16, MINI_BATCH_SIZE, 16, 40 , 40 );
		ReLUConfig* rsu6_l88 = new ReLUConfig(40 * 40  * 16, MINI_BATCH_SIZE);

		UpsampleConfig* rsu6_l89 = new UpsampleConfig(40, 40, 16, MINI_BATCH_SIZE, 2);

		//conv2d
		CNNConfig* rsu6_l90 = new CNNConfig(80, 80, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l91 = new BNConfig(80 * 80 * 16, MINI_BATCH_SIZE, 16, 80 , 80 );
		ReLUConfig* rsu6_l92 = new ReLUConfig(80 * 80  * 16, MINI_BATCH_SIZE);

		UpsampleConfig* rsu6_l93 = new UpsampleConfig(80, 80, 16, MINI_BATCH_SIZE, 2);

		//conv1d
		CNNConfig* rsu6_l94 = new CNNConfig(160, 160, 2 * 16, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l95 = new BNConfig(160 * 160 * 64, MINI_BATCH_SIZE, 64, 160 , 160 );
		ReLUConfig* rsu6_l96 = new ReLUConfig(160 * 160  * 64, MINI_BATCH_SIZE);
		//return rsu6_l96+l55
		MaxpoolConfig* rsu6_l97 = new MaxpoolConfig(160, 160, 64, 2, 2, MINI_BATCH_SIZE);
		//stage2 end
		
		//convin
		CNNConfig* rsu5_l98 = new CNNConfig(80, 80, 64, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu5_l99 = new BNConfig(80 * 80 * 64, MINI_BATCH_SIZE, 64, 80, 80);
		ReLUConfig* rsu5_l100 = new ReLUConfig(80 * 80 * 64, MINI_BATCH_SIZE);

		//conv1
		CNNConfig* rsu5_l101 = new CNNConfig(80, 80, 64, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu5_l102 = new BNConfig(80 * 80 * 16, MINI_BATCH_SIZE, 16, 80, 80);
		ReLUConfig* rsu5_l103 = new ReLUConfig(80 * 80 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu5_l104 = new MaxpoolConfig(80, 80, 16, 2, 2, MINI_BATCH_SIZE);

		//conv2
		CNNConfig* rsu5_l105 = new CNNConfig(40, 40, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu5_l106 = new BNConfig(40 * 40 * 16, MINI_BATCH_SIZE, 16, 40, 40);
		ReLUConfig* rsu5_l107 = new ReLUConfig(40 * 40 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu5_l108 = new MaxpoolConfig(40, 40, 16, 2, 2, MINI_BATCH_SIZE);

		//conv3
		CNNConfig* rsu5_l109 = new CNNConfig(20, 20, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu5_l110 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20, 20);
		ReLUConfig* rsu5_l111 = new ReLUConfig(20 * 20 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu5_l112 = new MaxpoolConfig(20, 20, 16, 2, 2, MINI_BATCH_SIZE);

		//conv4
		CNNConfig* rsu5_l113 = new CNNConfig(10, 10, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu5_l114 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10, 10);
		ReLUConfig* rsu5_l115 = new ReLUConfig(10 * 10 * 16, MINI_BATCH_SIZE);

		//conv5
		CNNConfig* rsu5_l116 = new CNNConfig(10, 10, 16, 16, 3, 1, 2, MINI_BATCH_SIZE);
		BNConfig* rsu5_l117 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10, 10);
		ReLUConfig* rsu5_l118 = new ReLUConfig(10 * 10 * 16, MINI_BATCH_SIZE);

		//conv4d
		CNNConfig* rsu5_l119 = new CNNConfig(10, 10, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu5_l120 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10 , 10 );
		ReLUConfig* rsu5_l121 = new ReLUConfig(10 * 10  * 16, MINI_BATCH_SIZE);

		UpsampleConfig* rsu5_l122 = new UpsampleConfig(10, 10, 16, MINI_BATCH_SIZE, 2);

		//conv3d
		CNNConfig* rsu5_l123 = new CNNConfig(20, 20, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu5_l124 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20 , 20 );
		ReLUConfig* rsu5_l125 = new ReLUConfig(20 * 20  * 16, MINI_BATCH_SIZE);

		UpsampleConfig* rsu5_l126 = new UpsampleConfig(20, 20, 16, MINI_BATCH_SIZE, 2);

		//conv2d
		CNNConfig* rsu5_l127 = new CNNConfig(40, 40, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu5_l128 = new BNConfig(40 * 40 * 16, MINI_BATCH_SIZE, 16, 40 , 40 );
		ReLUConfig* rsu5_l129 = new ReLUConfig(40 * 40  * 16, MINI_BATCH_SIZE);

		UpsampleConfig* rsu5_l130 = new UpsampleConfig(40, 40, 16, MINI_BATCH_SIZE, 2);

		//conv1d
		CNNConfig* rsu5_l131 = new CNNConfig(80, 80, 2 * 16, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu5_l132 = new BNConfig(80 * 80 * 64, MINI_BATCH_SIZE, 64, 80 , 80 );
		ReLUConfig* rsu5_l133 = new ReLUConfig(80 * 80  * 64, MINI_BATCH_SIZE);
		//return rsu5_l133+l100

		MaxpoolConfig* rsu5_l134 = new MaxpoolConfig(80, 80, 64, 2, 2, MINI_BATCH_SIZE);
		//stage3 end
		

		// convin
		CNNConfig* rsu4_l135 = new CNNConfig(40, 40, 64, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4_l136 = new BNConfig(40 * 40 * 64, MINI_BATCH_SIZE, 64, 40, 40);
		ReLUConfig* rsu4_l137 = new ReLUConfig(40 * 40 * 64, MINI_BATCH_SIZE);

		// conv1
		CNNConfig* rsu4_l138 = new CNNConfig(40, 40, 64, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4_l139 = new BNConfig(40 * 40 * 16, MINI_BATCH_SIZE, 16, 40, 40);
		ReLUConfig* rsu4_l140 = new ReLUConfig(40 * 40 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu4_l141 = new MaxpoolConfig(40, 40, 16, 2, 2, MINI_BATCH_SIZE);

		// conv2
		CNNConfig* rsu4_l142 = new CNNConfig(20, 20, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4_l143 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20, 20);
		ReLUConfig* rsu4_l144 = new ReLUConfig(20 * 20 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu4_l145 = new MaxpoolConfig(20, 20, 16, 2, 2, MINI_BATCH_SIZE);

		// conv3
		CNNConfig* rsu4_l146 = new CNNConfig(10, 10, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4_l147 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10, 10);
		ReLUConfig* rsu4_l148 = new ReLUConfig(10 * 10 * 16, MINI_BATCH_SIZE);

		// conv4
		CNNConfig* rsu4_l149 = new CNNConfig(10, 10, 16, 16, 3, 1, 2, MINI_BATCH_SIZE);
		BNConfig* rsu4_l150 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10, 10);
		ReLUConfig* rsu4_l151 = new ReLUConfig(10 * 10 * 16, MINI_BATCH_SIZE);

		//conv3d
		CNNConfig* rsu4_l152 = new CNNConfig(10, 10, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4_l153 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10 , 10 );
		ReLUConfig* rsu4_l154 = new ReLUConfig(10 * 10  * 16, MINI_BATCH_SIZE);

		UpsampleConfig* rsu4_l155 = new UpsampleConfig(10, 10, 16, MINI_BATCH_SIZE, 2);

		//conv2d
		CNNConfig* rsu4_l156 = new CNNConfig(20, 20, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4_l157 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20 , 20 );
		ReLUConfig* rsu4_l158 = new ReLUConfig(20 * 20  * 16, MINI_BATCH_SIZE);

		UpsampleConfig* rsu4_l159 = new UpsampleConfig(20, 20, 16, MINI_BATCH_SIZE, 2);

		//conv1d
		CNNConfig* rsu4_l160 = new CNNConfig(40, 40, 2 * 16, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4_l161 = new BNConfig(40 * 40 * 64, MINI_BATCH_SIZE, 64, 40 , 40 );
		ReLUConfig* rsu4_l162 = new ReLUConfig(40 * 40  * 64, MINI_BATCH_SIZE);
		//return rsu4_l162+l137

		MaxpoolConfig* rsu4_l163 = new MaxpoolConfig(40, 40, 64, 2, 2, MINI_BATCH_SIZE);
		//stage4 end

	
		// convin
		CNNConfig* rsu4f_l164 = new CNNConfig(20, 20, 64, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l165 = new BNConfig(20 * 20 * 64, MINI_BATCH_SIZE, 64, 20, 20);
		ReLUConfig* rsu4f_l166 = new ReLUConfig(20 * 20 * 64, MINI_BATCH_SIZE);

		// conv1
		CNNConfig* rsu4f_l167 = new CNNConfig(20, 20, 64, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l168 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20, 20);
		ReLUConfig* rsu4f_l169 = new ReLUConfig(20 * 20 * 16, MINI_BATCH_SIZE);

		// conv2
		CNNConfig* rsu4f_l170 = new CNNConfig(20, 20, 16, 16, 3, 1, 2, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l171 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20, 20);
		ReLUConfig* rsu4f_l172 = new ReLUConfig(20 * 20 * 16, MINI_BATCH_SIZE);

		// conv3
		CNNConfig* rsu4f_l173 = new CNNConfig(20, 20, 16, 16, 3, 1, 4, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l174 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20, 20);
		ReLUConfig* rsu4f_l175 = new ReLUConfig(20 * 20 * 16, MINI_BATCH_SIZE);

		// conv4
		CNNConfig* rsu4f_l176 = new CNNConfig(20, 20, 16, 16, 3, 1, 8, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l177 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20, 20);
		ReLUConfig* rsu4f_l178 = new ReLUConfig(20 * 20 * 16, MINI_BATCH_SIZE);

		// conv3d
		CNNConfig* rsu4f_l179 = new CNNConfig(20, 20, 2 * 16, 16, 3, 1, 4, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l180 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20, 20);
		ReLUConfig* rsu4f_l181 = new ReLUConfig(20 * 20 * 16, MINI_BATCH_SIZE);

		// conv2d
		CNNConfig* rsu4f_l182 = new CNNConfig(20, 20, 2 * 16, 16, 3, 1, 2, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l183 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20, 20);
		ReLUConfig* rsu4f_l184 = new ReLUConfig(20 * 20 * 16, MINI_BATCH_SIZE);

		// conv1d
		CNNConfig* rsu4f_l185 = new CNNConfig(20, 20, 2 * 16, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l186 = new BNConfig(20 * 20 * 64, MINI_BATCH_SIZE, 64, 20, 20);
		ReLUConfig* rsu4f_l187 = new ReLUConfig(20 * 20 * 64, MINI_BATCH_SIZE);
		//return rsu4f_l187+l166

		MaxpoolConfig* rsu4f_l188 = new MaxpoolConfig(20, 20, 64, 2, 2, MINI_BATCH_SIZE);
		//stage5 end

		
		// convin
		CNNConfig* rsu4f_l189 = new CNNConfig(10, 10, 64, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l190 = new BNConfig(10 * 10 * 64, MINI_BATCH_SIZE, 64, 10, 10);
		ReLUConfig* rsu4f_l191 = new ReLUConfig(10 * 10 * 64, MINI_BATCH_SIZE);

		// conv1
		CNNConfig* rsu4f_l192 = new CNNConfig(10, 10, 64, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l193 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10, 10);
		ReLUConfig* rsu4f_l194 = new ReLUConfig(10 * 10 * 16, MINI_BATCH_SIZE);

		// conv2
		CNNConfig* rsu4f_l195 = new CNNConfig(10, 10, 16, 16, 3, 1, 2, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l196 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10, 10);
		ReLUConfig* rsu4f_l197 = new ReLUConfig(10 * 10 * 16, MINI_BATCH_SIZE);

		// conv3
		CNNConfig* rsu4f_l198 = new CNNConfig(10, 10, 16, 16, 3, 1, 4, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l199 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10, 10);
		ReLUConfig* rsu4f_l200 = new ReLUConfig(10 * 10 * 16, MINI_BATCH_SIZE);

		// conv4
		CNNConfig* rsu4f_l201 = new CNNConfig(10, 10, 16, 16, 3, 1, 8, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l202 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10, 10);
		ReLUConfig* rsu4f_l203 = new ReLUConfig(10 * 10 * 16, MINI_BATCH_SIZE);

		// conv3d
		CNNConfig* rsu4f_l204 = new CNNConfig(10, 10, 2 * 16, 16, 3, 1, 4, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l205 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10, 10);
		ReLUConfig* rsu4f_l206 = new ReLUConfig(10 * 10 * 16, MINI_BATCH_SIZE);

		// conv2d
		CNNConfig* rsu4f_l207 = new CNNConfig(10, 10, 2 * 16, 16, 3, 1, 2, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l208 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10, 10);
		ReLUConfig* rsu4f_l209 = new ReLUConfig(10 * 10 * 16, MINI_BATCH_SIZE);

		// conv1d
		CNNConfig* rsu4f_l210 = new CNNConfig(10, 10, 2 * 16, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l211 = new BNConfig(10 * 10 * 64, MINI_BATCH_SIZE, 64, 10, 10);
		ReLUConfig* rsu4f_l212 = new ReLUConfig(10 * 10 * 64, MINI_BATCH_SIZE);
		//return rsu4f_l212+l191

		UpsampleConfig* rsu4f_l213 = new UpsampleConfig(10, 10, 64, MINI_BATCH_SIZE, 2);
		//stage6 end


		// convin
		CNNConfig* rsu4f_l214 = new CNNConfig(20, 20, 128, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l215 = new BNConfig(20 * 20 * 64, MINI_BATCH_SIZE, 64, 20, 20);
		ReLUConfig* rsu4f_l216 = new ReLUConfig(20 * 20 * 64, MINI_BATCH_SIZE);

		// conv1
		CNNConfig* rsu4f_l217 = new CNNConfig(20, 20, 64, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l218 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20, 20);
		ReLUConfig* rsu4f_l219 = new ReLUConfig(20 * 20 * 16, MINI_BATCH_SIZE);

		// conv2
		CNNConfig* rsu4f_l220 = new CNNConfig(20, 20, 16, 16, 3, 1, 2, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l221 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20, 20);
		ReLUConfig* rsu4f_l222 = new ReLUConfig(20 * 20 * 16, MINI_BATCH_SIZE);

		// conv3
		CNNConfig* rsu4f_l223 = new CNNConfig(20, 20, 16, 16, 3, 1, 4, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l224 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20, 20);
		ReLUConfig* rsu4f_l225 = new ReLUConfig(20 * 20 * 16, MINI_BATCH_SIZE);

		// conv4
		CNNConfig* rsu4f_l226 = new CNNConfig(20, 20, 16, 16, 3, 1, 8, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l227 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20, 20);
		ReLUConfig* rsu4f_l228 = new ReLUConfig(20 * 20 * 16, MINI_BATCH_SIZE);

		// conv3d
		CNNConfig* rsu4f_l229 = new CNNConfig(20, 20, 2 * 16, 16, 3, 1, 4, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l230 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20, 20);
		ReLUConfig* rsu4f_l231 = new ReLUConfig(20 * 20 * 16, MINI_BATCH_SIZE);

		// conv2d
		CNNConfig* rsu4f_l232 = new CNNConfig(20, 20, 2 * 16, 16, 3, 1, 2, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l233 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20, 20);
		ReLUConfig* rsu4f_l234 = new ReLUConfig(20 * 20 * 16, MINI_BATCH_SIZE);

		// conv1d
		CNNConfig* rsu4f_l235 = new CNNConfig(20, 20, 2 * 16, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4f_l236 = new BNConfig(20 * 20 * 64, MINI_BATCH_SIZE, 64, 20, 20);
		ReLUConfig* rsu4f_l237 = new ReLUConfig(20 * 20 * 64, MINI_BATCH_SIZE);
		//return l237+216

		UpsampleConfig* rsu4f_l238 = new UpsampleConfig(20, 20, 64, MINI_BATCH_SIZE, 2);
		//stage5d end

		
		// convin
		CNNConfig* rsu4_l239 = new CNNConfig(40, 40, 128, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4_l240 = new BNConfig(40 * 40 * 64, MINI_BATCH_SIZE, 64, 40, 40);
		ReLUConfig* rsu4_l241 = new ReLUConfig(40 * 40 * 64, MINI_BATCH_SIZE);

		// conv1
		CNNConfig* rsu4_l242 = new CNNConfig(40, 40, 64, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4_l243 = new BNConfig(40 * 40 * 16, MINI_BATCH_SIZE, 16, 40, 40);
		ReLUConfig* rsu4_l244 = new ReLUConfig(40 * 40 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu4_l245 = new MaxpoolConfig(40, 40, 16, 2, 2, MINI_BATCH_SIZE);

		// conv2
		CNNConfig* rsu4_l246 = new CNNConfig(20, 20, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4_l247 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20, 20);
		ReLUConfig* rsu4_l248 = new ReLUConfig(20 * 20 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu4_l249 = new MaxpoolConfig(20, 20, 16, 2, 2, MINI_BATCH_SIZE);

		// conv3
		CNNConfig* rsu4_l250 = new CNNConfig(10, 10, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4_l251 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10, 10);
		ReLUConfig* rsu4_l252 = new ReLUConfig(10 * 10 * 16, MINI_BATCH_SIZE);

		// conv4
		CNNConfig* rsu4_l253 = new CNNConfig(10, 10, 16, 16, 3, 1, 2, MINI_BATCH_SIZE);
		BNConfig* rsu4_l254 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10, 10);
		ReLUConfig* rsu4_l255 = new ReLUConfig(10 * 10 * 16, MINI_BATCH_SIZE);

		// conv3d 255+252
		CNNConfig* rsu4_l256 = new CNNConfig(10, 10, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4_l257 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10, 10);
		ReLUConfig* rsu4_l258 = new ReLUConfig(10 * 10 * 16, MINI_BATCH_SIZE);

		UpsampleConfig* rsu4_l259 = new UpsampleConfig(10, 10, 16, MINI_BATCH_SIZE, 2);

		// conv2d 259+248
		CNNConfig* rsu4_l260 = new CNNConfig(20, 20, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4_l261 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20, 20);
		ReLUConfig* rsu4_l262 = new ReLUConfig(20 * 20 * 16, MINI_BATCH_SIZE);

		UpsampleConfig* rsu4_l263 = new UpsampleConfig(20, 20, 16, MINI_BATCH_SIZE, 2);

		// conv1d 263+244
		CNNConfig* rsu4_l264 = new CNNConfig(40, 40, 2 * 16, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu4_l265 = new BNConfig(40 * 40 * 64, MINI_BATCH_SIZE, 64, 40, 40);
		ReLUConfig* rsu4_l266 = new ReLUConfig(40 * 40 * 64, MINI_BATCH_SIZE);
		//return l266+241

		UpsampleConfig* rsu4_l267 = new UpsampleConfig(40, 40, 64, MINI_BATCH_SIZE, 2);
		// stage4d end

		
		//convin
		CNNConfig* rsu5_l268 = new CNNConfig(80, 80, 128, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu5_l269 = new BNConfig(80 * 80 * 64, MINI_BATCH_SIZE, 64, 80, 80);
		ReLUConfig* rsu5_l270 = new ReLUConfig(80 * 80 * 64, MINI_BATCH_SIZE);

		//conv1
		CNNConfig* rsu5_l271 = new CNNConfig(80, 80, 64, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu5_l272 = new BNConfig(80 * 80 * 16, MINI_BATCH_SIZE, 16, 80, 80);
		ReLUConfig* rsu5_l273 = new ReLUConfig(80 * 80 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu5_l274 = new MaxpoolConfig(80, 80, 16, 2, 2, MINI_BATCH_SIZE);

		//conv2
		CNNConfig* rsu5_l275 = new CNNConfig(40, 40, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu5_l276 = new BNConfig(40 * 40 * 16, MINI_BATCH_SIZE, 16, 40, 40);
		ReLUConfig* rsu5_l277 = new ReLUConfig(40 * 40 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu5_l278 = new MaxpoolConfig(40, 40, 16, 2, 2, MINI_BATCH_SIZE);

		//conv3
		CNNConfig* rsu5_l279 = new CNNConfig(20, 20, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu5_l280 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20, 20);
		ReLUConfig* rsu5_l281 = new ReLUConfig(20 * 20 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu5_l282 = new MaxpoolConfig(20, 20, 16, 2, 2, MINI_BATCH_SIZE);

		//conv4
		CNNConfig* rsu5_l283 = new CNNConfig(10, 10, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu5_l284 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10, 10);
		ReLUConfig* rsu5_l285 = new ReLUConfig(10 * 10 * 16, MINI_BATCH_SIZE);

		//conv5
		CNNConfig* rsu5_l286 = new CNNConfig(10, 10, 16, 16, 3, 1, 2, MINI_BATCH_SIZE);  //膨胀卷积
		BNConfig* rsu5_l287 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10, 10);
		ReLUConfig* rsu5_l288 = new ReLUConfig(10 * 10 * 16, MINI_BATCH_SIZE);

		//conv4d   288+285
		CNNConfig* rsu5_l289 = new CNNConfig(10, 10, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu5_l290 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10 , 10 );
		ReLUConfig* rsu5_l291 = new ReLUConfig(10 * 10  * 16, MINI_BATCH_SIZE);	

		UpsampleConfig* rsu5_l292 = new UpsampleConfig(10, 10, 16, MINI_BATCH_SIZE, 2);

		//conv3d   292+281
		CNNConfig* rsu5_l293 = new CNNConfig(20, 20, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu5_l294 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20 , 20 );
		ReLUConfig* rsu5_l295 = new ReLUConfig(20 * 20  * 16, MINI_BATCH_SIZE);	

		UpsampleConfig* rsu5_l296 = new UpsampleConfig(20, 20, 16, MINI_BATCH_SIZE, 2);

		//conv2d   296+277
		CNNConfig* rsu5_l297 = new CNNConfig(40, 40, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu5_l298 = new BNConfig(40 * 40 * 16, MINI_BATCH_SIZE, 16, 40 , 40 );
		ReLUConfig* rsu5_l299 = new ReLUConfig(40 * 40  * 16, MINI_BATCH_SIZE);	

		UpsampleConfig* rsu5_l300 = new UpsampleConfig(40, 40, 16, MINI_BATCH_SIZE, 2);

		//conv1d   300+273
		CNNConfig* rsu5_l301 = new CNNConfig(80, 80, 2 * 16, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu5_l302 = new BNConfig(80 * 80 * 64, MINI_BATCH_SIZE, 64, 80 , 80 );
		ReLUConfig* rsu5_l303 = new ReLUConfig(80 * 80  * 64, MINI_BATCH_SIZE);	
		//return rsu5_l303+l270

		UpsampleConfig* rsu5_l304 = new UpsampleConfig(80, 80, 64, MINI_BATCH_SIZE, 2);
		// stage3d end

		CNNConfig* side3_l305 = new CNNConfig(80, 80, 64, 1, 3, 1, 1, MINI_BATCH_SIZE);
		UpsampleConfig* side3_l306 = new UpsampleConfig(80, 80, 1, MINI_BATCH_SIZE, 4); 
		
		//convin
		CNNConfig* rsu6_l305 = new CNNConfig(160, 160, 128, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l306 = new BNConfig(160 * 160 * 64, MINI_BATCH_SIZE, 64, 160, 160);
		ReLUConfig* rsu6_l307 = new ReLUConfig(160 * 160 * 64, MINI_BATCH_SIZE);

		//conv1
		CNNConfig* rsu6_l308 = new CNNConfig(160, 160, 64, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l309 = new BNConfig(160 * 160 * 16, MINI_BATCH_SIZE, 16, 160, 160);
		ReLUConfig* rsu6_l310 = new ReLUConfig(160 * 160 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu6_l311 = new MaxpoolConfig(160, 160, 16, 2, 2, MINI_BATCH_SIZE);

		//conv2
		CNNConfig* rsu6_l312 = new CNNConfig(80, 80, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l313 = new BNConfig(80 * 80 * 16, MINI_BATCH_SIZE, 16, 80, 80);
		ReLUConfig* rsu6_l314 = new ReLUConfig(80 * 80 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu6_l315 = new MaxpoolConfig(80, 80, 16, 2, 2, MINI_BATCH_SIZE);

		//conv3
		CNNConfig* rsu6_l316 = new CNNConfig(40, 40, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l317 = new BNConfig(40 * 40 * 16, MINI_BATCH_SIZE, 16, 40, 40);
		ReLUConfig* rsu6_l318 = new ReLUConfig(40 * 40 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu6_l319 = new MaxpoolConfig(40, 40, 16, 2, 2, MINI_BATCH_SIZE);

		//conv4
		CNNConfig* rsu6_l320 = new CNNConfig(20, 20, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l321 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20, 20);
		ReLUConfig* rsu6_l322 = new ReLUConfig(20 * 20 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu6_l323 = new MaxpoolConfig(20, 20, 16, 2, 2, MINI_BATCH_SIZE);

		//conv5
		CNNConfig* rsu6_l324 = new CNNConfig(10, 10, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l325 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10, 10);
		ReLUConfig* rsu6_l326 = new ReLUConfig(10 * 10 * 16, MINI_BATCH_SIZE);

		//conv6
		CNNConfig* rsu6_l327 = new CNNConfig(10, 10, 16, 16, 3, 1, 2, MINI_BATCH_SIZE);  //膨胀卷积
		BNConfig* rsu6_l328 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10, 10);
		ReLUConfig* rsu6_l329 = new ReLUConfig(10 * 10 * 16, MINI_BATCH_SIZE);

		//conv5d   329+326
		CNNConfig* rsu6_l330 = new CNNConfig(10, 10, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l331 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10 , 10 );
		ReLUConfig* rsu6_l332 = new ReLUConfig(10 * 10  * 16, MINI_BATCH_SIZE);	

		UpsampleConfig* rsu6_l333 = new UpsampleConfig(10, 10, 16, MINI_BATCH_SIZE, 2);

		//conv4d   333+322
		CNNConfig* rsu6_l334 = new CNNConfig(20, 20, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l335 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20 , 20 );
		ReLUConfig* rsu6_l336 = new ReLUConfig(20 * 20  * 16, MINI_BATCH_SIZE);	

		UpsampleConfig* rsu6_l337 = new UpsampleConfig(20, 20, 16, MINI_BATCH_SIZE, 2);

		//conv3d   337+318
		CNNConfig* rsu6_l338 = new CNNConfig(40, 40, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l339 = new BNConfig(40 * 40 * 16, MINI_BATCH_SIZE, 16, 40 , 40 );
		ReLUConfig* rsu6_l340 = new ReLUConfig(40 * 40  * 16, MINI_BATCH_SIZE);	

		UpsampleConfig* rsu6_l341 = new UpsampleConfig(40, 40, 16, MINI_BATCH_SIZE, 2);

		//conv2d   341+314
		CNNConfig* rsu6_l342 = new CNNConfig(80, 80, 2 * 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l343 = new BNConfig(80 * 80 * 16, MINI_BATCH_SIZE, 16, 80 , 80 );
		ReLUConfig* rsu6_l344 = new ReLUConfig(80 * 80  * 16, MINI_BATCH_SIZE);	

		UpsampleConfig* rsu6_l345 = new UpsampleConfig(80, 80, 16, MINI_BATCH_SIZE, 2);

		//conv1d   
		CNNConfig* rsu6_l346 = new CNNConfig(160, 160, 2 * 16, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu6_l347 = new BNConfig(160 * 160 * 64, MINI_BATCH_SIZE, 64, 160 , 160 );
		ReLUConfig* rsu6_l348 = new ReLUConfig(160 * 160  * 64, MINI_BATCH_SIZE);

		UpsampleConfig* rsu6_l349 = new UpsampleConfig(160, 160, 64, MINI_BATCH_SIZE, 2);
		
		//stage2d end
		/*
		CNNConfig* side2_l349 = new CNNConfig(160, 160, 64, 1, 3, 1, 1, MINI_BATCH_SIZE);
		UpsampleConfig* side2_l350 = new UpsampleConfig(160, 160, 1, MINI_BATCH_SIZE, 2);   //return d2=404

		SigmoidConfig* Fd1 = new SigmoidConfig(320*320*1, MINI_BATCH_SIZE);
		*/
		//convin
		CNNConfig* rsu7_l350 = new CNNConfig(320, 320, 128, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l351 = new BNConfig(320 * 320 * 64, MINI_BATCH_SIZE, 64, 320, 320);
		ReLUConfig* rsu7_l352 = new ReLUConfig(320 * 320 * 64, MINI_BATCH_SIZE);

		//conv1
		CNNConfig* rsu7_l353 = new CNNConfig(320, 320, 64, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l354 = new BNConfig(320 * 320 * 16, MINI_BATCH_SIZE, 16, 320, 320);
		ReLUConfig* rsu7_l355 = new ReLUConfig(320 * 320 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu7_l356 = new MaxpoolConfig(320, 320, 16, 2, 2, MINI_BATCH_SIZE);
		//conv2
		CNNConfig* rsu7_l357 = new CNNConfig(160, 160, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l358 = new BNConfig(160 * 160 * 16, MINI_BATCH_SIZE, 16, 160, 160);
		ReLUConfig* rsu7_l359 = new ReLUConfig(160 * 160 * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu7_l360 = new MaxpoolConfig(160, 160, 16, 2, 2, MINI_BATCH_SIZE);
		//conv3
		CNNConfig* rsu7_l361 = new CNNConfig(80, 80, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l362 = new BNConfig(80 * 80 * 16, MINI_BATCH_SIZE, 16, 80 , 80 );
		ReLUConfig* rsu7_l363 = new ReLUConfig(80 * 80  * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu7_l364 = new MaxpoolConfig(80, 80, 16, 2, 2, MINI_BATCH_SIZE);
		//conv4
		CNNConfig* rsu7_l365 = new CNNConfig(40, 40, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l366 = new BNConfig(40 * 40 * 16, MINI_BATCH_SIZE, 16, 40 , 40 );
		ReLUConfig* rsu7_l367 = new ReLUConfig(40 * 40  * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu7_l368 = new MaxpoolConfig(40, 40, 16, 2, 2, MINI_BATCH_SIZE);
		//conv5
		CNNConfig* rsu7_l369 = new CNNConfig(20, 20, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l370 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20 , 20 );
		ReLUConfig* rsu7_l371 = new ReLUConfig(20 * 20  * 16, MINI_BATCH_SIZE);

		MaxpoolConfig* rsu7_l372 = new MaxpoolConfig(20, 20, 16, 2, 2, MINI_BATCH_SIZE);
		//conv6
		CNNConfig* rsu7_l373 = new CNNConfig(10, 10, 16, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l374 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10 , 10 );
		ReLUConfig* rsu7_l375 = new ReLUConfig(10 * 10  * 16, MINI_BATCH_SIZE);

		//conv7
		CNNConfig* rsu7_l376 = new CNNConfig(10, 10, 16, 16, 3, 1, 2, MINI_BATCH_SIZE);  //膨胀卷积
		BNConfig* rsu7_l377 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10 , 10 );
		ReLUConfig* rsu7_l378 = new ReLUConfig(10 * 10  * 16, MINI_BATCH_SIZE);

		//conv6d 378+375
		CNNConfig* rsu7_l379 = new CNNConfig(10, 10, 32, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l380 = new BNConfig(10 * 10 * 16, MINI_BATCH_SIZE, 16, 10 , 10 );
		ReLUConfig* rsu7_l381 = new ReLUConfig(10 * 10  * 16, MINI_BATCH_SIZE);

		UpsampleConfig* rsu7_l382 = new UpsampleConfig(10, 10, 16, MINI_BATCH_SIZE, 2);

		//conv5d  382+371
		CNNConfig* rsu7_l383 = new CNNConfig(20, 20, 32, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l384 = new BNConfig(20 * 20 * 16, MINI_BATCH_SIZE, 16, 20 , 20 );
		ReLUConfig* rsu7_l385 = new ReLUConfig(20 * 20  * 16, MINI_BATCH_SIZE);

		UpsampleConfig* rsu7_l386 = new UpsampleConfig(20, 20, 16, MINI_BATCH_SIZE, 2);

		//conv4d  386+367
		CNNConfig* rsu7_l387 = new CNNConfig(40, 40, 32, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l388 = new BNConfig(40 * 40 * 16, MINI_BATCH_SIZE, 16, 40 , 40);
		ReLUConfig* rsu7_l389 = new ReLUConfig(40 * 40  * 16, MINI_BATCH_SIZE);

		UpsampleConfig* rsu7_l390 = new UpsampleConfig(40, 40, 16, MINI_BATCH_SIZE, 2);

		//conv3d  390+363
		CNNConfig* rsu7_l391 = new CNNConfig(80, 80, 32, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l392 = new BNConfig(80 * 80 * 16, MINI_BATCH_SIZE, 16, 80 , 80);
		ReLUConfig* rsu7_l393 = new ReLUConfig(80 * 80  * 16, MINI_BATCH_SIZE);

		UpsampleConfig* rsu7_l394 = new UpsampleConfig(80, 80, 32, MINI_BATCH_SIZE, 2);

        //conv2d  394+359
		CNNConfig* rsu7_l395 = new CNNConfig(160, 160, 32, 16, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l396 = new BNConfig(160 * 160 * 16, MINI_BATCH_SIZE, 16, 160 , 160);
		ReLUConfig* rsu7_l397 = new ReLUConfig(160 * 160  * 16, MINI_BATCH_SIZE);	

		UpsampleConfig* rsu7_l398 = new UpsampleConfig(160, 160, 32, MINI_BATCH_SIZE, 2);

		//conv1d   398+355
		CNNConfig* rsu7_l399 = new CNNConfig(320, 320, 32, 64, 3, 1, 1, MINI_BATCH_SIZE);
		BNConfig* rsu7_l400 = new BNConfig(320 * 320 * 64, MINI_BATCH_SIZE, 64, 320 , 320);
		ReLUConfig* rsu7_l401 = new ReLUConfig(320 * 320  * 64, MINI_BATCH_SIZE);
		// RSU7 return rsu7_401+352

		//stage1d end

		// CNNConfig* side1_l402 = new CNNConfig(320, 320, 64, 1, 3, 1, 1, MINI_BATCH_SIZE);   //return d1=042

		// CNNConfig* side2_l403 = new CNNConfig(160, 160, 64, 1, 3, 1, 1, MINI_BATCH_SIZE);
		// UpsampleConfig* side2_l404 = new UpsampleConfig(160, 160, 1, MINI_BATCH_SIZE, 2);   //return d2=404

		// SigmoidConfig* Fd1 = new SigmoidConfig(320*320*1, MINI_BATCH_SIZE);
		// CNNConfig* side3_l405 = new CNNConfig(80, 80, 128, 1, 3, 1, 1, MINI_BATCH_SIZE);
		// UpsampleConfig* side3_l406 = new UpsampleConfig(80, 80, 1, MINI_BATCH_SIZE, 4);   //return d3=406

		// CNNConfig* side4_l407 = new CNNConfig(40, 40, 256, 1, 3, 1, 1, MINI_BATCH_SIZE);
		// UpsampleConfig* side4_l408 = new UpsampleConfig(40, 40, 1, MINI_BATCH_SIZE, 8);   //return d4=408

		// CNNConfig* side5_l409 = new CNNConfig(20, 20, 512, 1, 3, 1, 1, MINI_BATCH_SIZE);
		// UpsampleConfig* side5_l410 = new UpsampleConfig(20, 20, 1, MINI_BATCH_SIZE, 16);   //return d5=410

		// CNNConfig* side6_l411 = new CNNConfig(10, 10, 512, 1, 3, 1, 1, MINI_BATCH_SIZE);
		// UpsampleConfig* side6_l412 = new UpsampleConfig(10, 10, 1, MINI_BATCH_SIZE, 32);   //return d6=412

		// CNNConfig* d0_l413 = new CNNConfig(320, 320, 6, 1, 3, 1, 1, MINI_BATCH_SIZE);

		// SigmoidConfig* Fd0 = new SigmoidConfig(320*320*1, MINI_BATCH_SIZE);
		// SigmoidConfig* Fd1 = new SigmoidConfig(320*320*1, MINI_BATCH_SIZE);
		// SigmoidConfig* Fd2 = new SigmoidConfig(320*320*1, MINI_BATCH_SIZE);
		// SigmoidConfig* Fd3 = new SigmoidConfig(320*320*1, MINI_BATCH_SIZE);
		// SigmoidConfig* Fd4 = new SigmoidConfig(320*320*1, MINI_BATCH_SIZE);
		// SigmoidConfig* Fd5 = new SigmoidConfig(320*320*1, MINI_BATCH_SIZE);
		// SigmoidConfig* Fd6 = new SigmoidConfig(320*320*1, MINI_BATCH_SIZE);

    	config->addLayer(rsu7_l0);
		config->addLayer(rsu7_l1);
		config->addLayer(rsu7_l2);
    	config->addLayer(rsu7_l3);
		config->addLayer(rsu7_l4);
		config->addLayer(rsu7_l5);
		config->addLayer(rsu7_l6);
		config->addLayer(rsu7_l7);
		config->addLayer(rsu7_l8);
		config->addLayer(rsu7_l9);
		config->addLayer(rsu7_l10);
		config->addLayer(rsu7_l11);
		config->addLayer(rsu7_l12);
		config->addLayer(rsu7_l13);
		config->addLayer(rsu7_l14);
		config->addLayer(rsu7_l15);
		config->addLayer(rsu7_l16);
		config->addLayer(rsu7_l17);
		config->addLayer(rsu7_l18);
		config->addLayer(rsu7_l19);
		config->addLayer(rsu7_l20);
		config->addLayer(rsu7_l21);
		config->addLayer(rsu7_l22);
		config->addLayer(rsu7_l23);
		config->addLayer(rsu7_l24);
		config->addLayer(rsu7_l25);
		config->addLayer(rsu7_l26);
		config->addLayer(rsu7_l27);
		config->addLayer(rsu7_l28);
		config->addLayer(rsu7_l29);
		config->addLayer(rsu7_l30);
		config->addLayer(rsu7_l31);
		config->addLayer(rsu7_l32);
		config->addLayer(rsu7_l33);
		config->addLayer(rsu7_l34);
		config->addLayer(rsu7_l35);
		config->addLayer(rsu7_l36);
		config->addLayer(rsu7_l37);
		config->addLayer(rsu7_l38);
		config->addLayer(rsu7_l39);
		config->addLayer(rsu7_l40);
		config->addLayer(rsu7_l41);
		config->addLayer(rsu7_l42);
		config->addLayer(rsu7_l43);
		config->addLayer(rsu7_l44);
		config->addLayer(rsu7_l45);
		config->addLayer(rsu7_l46);
		config->addLayer(rsu7_l47);
		config->addLayer(rsu7_l48);
		config->addLayer(rsu7_l49);
		config->addLayer(rsu7_l50);
		config->addLayer(rsu7_l51);
		config->addLayer(rsu7_l52);

		config->addLayer(rsu6_l53);
		config->addLayer(rsu6_l54);
		config->addLayer(rsu6_l55);
		config->addLayer(rsu6_l56);
		config->addLayer(rsu6_l57);
		config->addLayer(rsu6_l58);
		config->addLayer(rsu6_l59);
		config->addLayer(rsu6_l60);
		config->addLayer(rsu6_l61);
		config->addLayer(rsu6_l62);
		config->addLayer(rsu6_l63);
		config->addLayer(rsu6_l64);
		config->addLayer(rsu6_l65);
		config->addLayer(rsu6_l66);
		config->addLayer(rsu6_l67);
		config->addLayer(rsu6_l68);
		config->addLayer(rsu6_l69);
		config->addLayer(rsu6_l70);
		config->addLayer(rsu6_l71);
		config->addLayer(rsu6_l72);
		config->addLayer(rsu6_l73);
		config->addLayer(rsu6_l74);
		config->addLayer(rsu6_l75);
		config->addLayer(rsu6_l76);
		config->addLayer(rsu6_l77);
		config->addLayer(rsu6_l78);
		config->addLayer(rsu6_l79);
		config->addLayer(rsu6_l80);
		config->addLayer(rsu6_l81);
		config->addLayer(rsu6_l82);
		config->addLayer(rsu6_l83);
		config->addLayer(rsu6_l84);
		config->addLayer(rsu6_l85);
		config->addLayer(rsu6_l86);
		config->addLayer(rsu6_l87);
		config->addLayer(rsu6_l88);
		config->addLayer(rsu6_l89);
		config->addLayer(rsu6_l90);
		config->addLayer(rsu6_l91);
		config->addLayer(rsu6_l92);
		config->addLayer(rsu6_l93);
		config->addLayer(rsu6_l94);
		config->addLayer(rsu6_l95);
		config->addLayer(rsu6_l96);
		config->addLayer(rsu6_l97);

		config->addLayer(rsu5_l98);
		config->addLayer(rsu5_l99);
		config->addLayer(rsu5_l100);
		config->addLayer(rsu5_l101);
		config->addLayer(rsu5_l102);
		config->addLayer(rsu5_l103);
		config->addLayer(rsu5_l104);
		config->addLayer(rsu5_l105);
		config->addLayer(rsu5_l106);
		config->addLayer(rsu5_l107);
		config->addLayer(rsu5_l108);
		config->addLayer(rsu5_l109);
		config->addLayer(rsu5_l110);
		config->addLayer(rsu5_l111);
		config->addLayer(rsu5_l112);
		config->addLayer(rsu5_l113);
		config->addLayer(rsu5_l114);
		config->addLayer(rsu5_l115);
		config->addLayer(rsu5_l116);
		config->addLayer(rsu5_l117);
		config->addLayer(rsu5_l118);
		config->addLayer(rsu5_l119);
		config->addLayer(rsu5_l120);
		config->addLayer(rsu5_l121);
		config->addLayer(rsu5_l122);
		config->addLayer(rsu5_l123);
		config->addLayer(rsu5_l124);
		config->addLayer(rsu5_l125);
		config->addLayer(rsu5_l126);
		config->addLayer(rsu5_l127);
		config->addLayer(rsu5_l128);
		config->addLayer(rsu5_l129);
		config->addLayer(rsu5_l130);
		config->addLayer(rsu5_l131);
		config->addLayer(rsu5_l132);
		config->addLayer(rsu5_l133);
		config->addLayer(rsu5_l134);

		config->addLayer(rsu4_l135);
		config->addLayer(rsu4_l136);
		config->addLayer(rsu4_l137);
		config->addLayer(rsu4_l138);
		config->addLayer(rsu4_l139);
		config->addLayer(rsu4_l140);
		config->addLayer(rsu4_l141);
		config->addLayer(rsu4_l142);
		config->addLayer(rsu4_l143);
		config->addLayer(rsu4_l144);
		config->addLayer(rsu4_l145);
		config->addLayer(rsu4_l146);
		config->addLayer(rsu4_l147);
		config->addLayer(rsu4_l148);
		config->addLayer(rsu4_l149);
		config->addLayer(rsu4_l150);
		config->addLayer(rsu4_l151);
		config->addLayer(rsu4_l152);
		config->addLayer(rsu4_l153);
		config->addLayer(rsu4_l154);
		config->addLayer(rsu4_l155);
		config->addLayer(rsu4_l156);
		config->addLayer(rsu4_l157);
		config->addLayer(rsu4_l158);
		config->addLayer(rsu4_l159);
		config->addLayer(rsu4_l160);
		config->addLayer(rsu4_l161);
		config->addLayer(rsu4_l162);
		config->addLayer(rsu4_l163);

		config->addLayer(rsu4f_l164);
		config->addLayer(rsu4f_l165);
		config->addLayer(rsu4f_l166);
		config->addLayer(rsu4f_l167);
		config->addLayer(rsu4f_l168);
		config->addLayer(rsu4f_l169);
		config->addLayer(rsu4f_l170);
		config->addLayer(rsu4f_l171);
		config->addLayer(rsu4f_l172);
		config->addLayer(rsu4f_l173);
		config->addLayer(rsu4f_l174);
		config->addLayer(rsu4f_l175);
		config->addLayer(rsu4f_l176);
		config->addLayer(rsu4f_l177);
		config->addLayer(rsu4f_l178);
		config->addLayer(rsu4f_l179);
		config->addLayer(rsu4f_l180);
		config->addLayer(rsu4f_l181);
		config->addLayer(rsu4f_l182);
		config->addLayer(rsu4f_l183);
		config->addLayer(rsu4f_l184);
		config->addLayer(rsu4f_l185);
		config->addLayer(rsu4f_l186);
		config->addLayer(rsu4f_l187);
		config->addLayer(rsu4f_l188);
		config->addLayer(rsu4f_l189);
		config->addLayer(rsu4f_l190);
		config->addLayer(rsu4f_l191);
		config->addLayer(rsu4f_l192);
		config->addLayer(rsu4f_l193);
		config->addLayer(rsu4f_l194);
		config->addLayer(rsu4f_l195);
		config->addLayer(rsu4f_l196);
		config->addLayer(rsu4f_l197);
		config->addLayer(rsu4f_l198);
		config->addLayer(rsu4f_l199);
		config->addLayer(rsu4f_l200);
		config->addLayer(rsu4f_l201);
		config->addLayer(rsu4f_l202);
		config->addLayer(rsu4f_l203);
		config->addLayer(rsu4f_l204);
		config->addLayer(rsu4f_l205);
		config->addLayer(rsu4f_l206);
		config->addLayer(rsu4f_l207);
		config->addLayer(rsu4f_l208);
		config->addLayer(rsu4f_l209);
		config->addLayer(rsu4f_l210);
		config->addLayer(rsu4f_l211);
		config->addLayer(rsu4f_l212);
		config->addLayer(rsu4f_l213);
		config->addLayer(rsu4f_l214);
		config->addLayer(rsu4f_l215);
		config->addLayer(rsu4f_l216);
		config->addLayer(rsu4f_l217);
		config->addLayer(rsu4f_l218);
		config->addLayer(rsu4f_l219);
		config->addLayer(rsu4f_l220);
		config->addLayer(rsu4f_l221);
		config->addLayer(rsu4f_l222);
		config->addLayer(rsu4f_l223);
		config->addLayer(rsu4f_l224);
		config->addLayer(rsu4f_l225);
		config->addLayer(rsu4f_l226);
		config->addLayer(rsu4f_l227);
		config->addLayer(rsu4f_l228);
		config->addLayer(rsu4f_l229);
		config->addLayer(rsu4f_l230);
		config->addLayer(rsu4f_l231);
		config->addLayer(rsu4f_l232);
		config->addLayer(rsu4f_l233);
		config->addLayer(rsu4f_l234);
		config->addLayer(rsu4f_l235);
		config->addLayer(rsu4f_l236);
		config->addLayer(rsu4f_l237);
		config->addLayer(rsu4f_l238);

		config->addLayer(rsu4_l239);
		config->addLayer(rsu4_l240);
		config->addLayer(rsu4_l241);
		config->addLayer(rsu4_l242);
		config->addLayer(rsu4_l243);
		config->addLayer(rsu4_l244);
		config->addLayer(rsu4_l245);
		config->addLayer(rsu4_l246);
		config->addLayer(rsu4_l247);
		config->addLayer(rsu4_l248);
		config->addLayer(rsu4_l249);
		config->addLayer(rsu4_l250);
		config->addLayer(rsu4_l251);
		config->addLayer(rsu4_l252);
		config->addLayer(rsu4_l253);
		config->addLayer(rsu4_l254);
		config->addLayer(rsu4_l255);
		config->addLayer(rsu4_l256);
		config->addLayer(rsu4_l257);
		config->addLayer(rsu4_l258);
		config->addLayer(rsu4_l259);
		config->addLayer(rsu4_l260);
		config->addLayer(rsu4_l261);
		config->addLayer(rsu4_l262);
		config->addLayer(rsu4_l263);
		config->addLayer(rsu4_l264);
		config->addLayer(rsu4_l265);
		config->addLayer(rsu4_l266);
		config->addLayer(rsu4_l267);

		config->addLayer(rsu5_l268);
		config->addLayer(rsu5_l269);
		config->addLayer(rsu5_l270);
		config->addLayer(rsu5_l271);
		config->addLayer(rsu5_l272);
		config->addLayer(rsu5_l273);
		config->addLayer(rsu5_l274);
		config->addLayer(rsu5_l275);
		config->addLayer(rsu5_l276);
		config->addLayer(rsu5_l277);
		config->addLayer(rsu5_l278);
		config->addLayer(rsu5_l279);
		config->addLayer(rsu5_l280);
		config->addLayer(rsu5_l281);
		config->addLayer(rsu5_l282);
		config->addLayer(rsu5_l283);
		config->addLayer(rsu5_l284);
		config->addLayer(rsu5_l285);
		config->addLayer(rsu5_l286);
		config->addLayer(rsu5_l287);
		config->addLayer(rsu5_l288);
		config->addLayer(rsu5_l289);
		config->addLayer(rsu5_l290);
		config->addLayer(rsu5_l291);
		config->addLayer(rsu5_l292);
		config->addLayer(rsu5_l293);
		config->addLayer(rsu5_l294);
		config->addLayer(rsu5_l295);
		config->addLayer(rsu5_l296);
		config->addLayer(rsu5_l297);
		config->addLayer(rsu5_l298);
		config->addLayer(rsu5_l299);
		config->addLayer(rsu5_l300);
		config->addLayer(rsu5_l301);
		config->addLayer(rsu5_l302);
		config->addLayer(rsu5_l303);
		config->addLayer(rsu5_l304);

		config->addLayer(rsu6_l305);
		config->addLayer(rsu6_l306);
		config->addLayer(rsu6_l307);
		config->addLayer(rsu6_l308);
		config->addLayer(rsu6_l309);
		config->addLayer(rsu6_l310);
		config->addLayer(rsu6_l311);
		config->addLayer(rsu6_l312);
		config->addLayer(rsu6_l313);
		config->addLayer(rsu6_l314);
		config->addLayer(rsu6_l315);
		config->addLayer(rsu6_l316);
		config->addLayer(rsu6_l317);
		config->addLayer(rsu6_l318);
		config->addLayer(rsu6_l319);
		config->addLayer(rsu6_l320);
		config->addLayer(rsu6_l321);
		config->addLayer(rsu6_l322);
		config->addLayer(rsu6_l323);
		config->addLayer(rsu6_l324);
		config->addLayer(rsu6_l325);
		config->addLayer(rsu6_l326);
		config->addLayer(rsu6_l327);
		config->addLayer(rsu6_l328);
		config->addLayer(rsu6_l329);
		config->addLayer(rsu6_l330);
		config->addLayer(rsu6_l331);
		config->addLayer(rsu6_l332);
		config->addLayer(rsu6_l333);
		config->addLayer(rsu6_l334);
		config->addLayer(rsu6_l335);
		config->addLayer(rsu6_l336);
		config->addLayer(rsu6_l337);
		config->addLayer(rsu6_l338);
		config->addLayer(rsu6_l339);
		config->addLayer(rsu6_l340);
		config->addLayer(rsu6_l341);
		config->addLayer(rsu6_l342);
		config->addLayer(rsu6_l343);
		config->addLayer(rsu6_l344);
		config->addLayer(rsu6_l345);
		config->addLayer(rsu6_l346);
		config->addLayer(rsu6_l347);
		config->addLayer(rsu6_l348);
		config->addLayer(rsu6_l349);

		config->addLayer(rsu7_l350);
		config->addLayer(rsu7_l351);
		config->addLayer(rsu7_l352);
		config->addLayer(rsu7_l353);
		config->addLayer(rsu7_l354);
		config->addLayer(rsu7_l355);
		config->addLayer(rsu7_l356);
		config->addLayer(rsu7_l357);
		config->addLayer(rsu7_l358);
		config->addLayer(rsu7_l359);
		config->addLayer(rsu7_l360);
		config->addLayer(rsu7_l361);
		config->addLayer(rsu7_l362);
		config->addLayer(rsu7_l363);
		config->addLayer(rsu7_l364);
		config->addLayer(rsu7_l365);
		config->addLayer(rsu7_l366);
		config->addLayer(rsu7_l367);
		config->addLayer(rsu7_l368);
		config->addLayer(rsu7_l369);
		config->addLayer(rsu7_l370);
		config->addLayer(rsu7_l371);
		config->addLayer(rsu7_l372);
		config->addLayer(rsu7_l373);
		config->addLayer(rsu7_l374);
		config->addLayer(rsu7_l375);
		config->addLayer(rsu7_l376);
		config->addLayer(rsu7_l377);
		config->addLayer(rsu7_l378);
		config->addLayer(rsu7_l379);
		config->addLayer(rsu7_l380);
		config->addLayer(rsu7_l381);
		config->addLayer(rsu7_l382);
		config->addLayer(rsu7_l383);
		config->addLayer(rsu7_l384);
		config->addLayer(rsu7_l385);
		config->addLayer(rsu7_l386);
		config->addLayer(rsu7_l387);
		config->addLayer(rsu7_l388);
		config->addLayer(rsu7_l389);
		config->addLayer(rsu7_l390);
		config->addLayer(rsu7_l391);
		config->addLayer(rsu7_l392);
		config->addLayer(rsu7_l393);
		config->addLayer(rsu7_l394);
		config->addLayer(rsu7_l395);
		config->addLayer(rsu7_l396);
		config->addLayer(rsu7_l397);
		config->addLayer(rsu7_l398);
		config->addLayer(rsu7_l399);
		config->addLayer(rsu7_l400);
		config->addLayer(rsu7_l401);

		config->addLayer(side1_l402);
		config->addLayer(side3_l305);
		config->addLayer(side3_l306);
		config->addLayer(side3_l405);
		config->addLayer(side3_l406);
		config->addLayer(side4_l407);
		config->addLayer(side4_l408);
		config->addLayer(side5_l409);
		config->addLayer(side5_l410);
		config->addLayer(side6_l411);
		config->addLayer(side6_l412);
		config->addLayer(d0_l413);
		// config->addLayer(Fd0);
		// config->addLayer(Fd1);
		// config->addLayer(Fd2);
		// config->addLayer(Fd3);
		// config->addLayer(Fd4);
		// config->addLayer(Fd5);
		// config->addLayer(Fd6);

	}
	else if (network.compare("AlexNet") == 0)
	{
		if(dataset.compare("MNIST") == 0)
			assert(false && "No AlexNet on MNIST");
		else if (dataset.compare("CIFAR10") == 0)
		{
			NUM_LAYERS = 20;
			// NUM_LAYERS = 18;		//Without BN
			WITH_NORMALIZATION = false;
			CNNConfig* l0 = new CNNConfig(33,33,3,96,11,4,9,MINI_BATCH_SIZE);
			MaxpoolConfig* l1 = new MaxpoolConfig(11,11,96,3,2,MINI_BATCH_SIZE);
			ReLUConfig* l2 = new ReLUConfig(5*5*96,MINI_BATCH_SIZE);		
			//BNConfig * l3 = new BNConfig(5*5*96,MINI_BATCH_SIZE);

			CNNConfig* l4 = new CNNConfig(5,5,96,256,5,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l5 = new MaxpoolConfig(3,3,256,3,2,MINI_BATCH_SIZE);
			ReLUConfig* l6 = new ReLUConfig(1*1*256,MINI_BATCH_SIZE);		
			//BNConfig * l7 = new BNConfig(1*1*256,MINI_BATCH_SIZE);

			CNNConfig* l8 = new CNNConfig(1,1,256,384,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l9 = new ReLUConfig(1*1*384,MINI_BATCH_SIZE);
			CNNConfig* l10 = new CNNConfig(1,1,384,384,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l11 = new ReLUConfig(1*1*384,MINI_BATCH_SIZE);
			CNNConfig* l12 = new CNNConfig(1,1,384,256,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l13 = new ReLUConfig(1*1*256,MINI_BATCH_SIZE);

			FCConfig* l14 = new FCConfig(1*1*256,MINI_BATCH_SIZE,256);
			ReLUConfig* l15 = new ReLUConfig(256,MINI_BATCH_SIZE);
			FCConfig* l16 = new FCConfig(256,MINI_BATCH_SIZE,256);
			ReLUConfig* l17 = new ReLUConfig(256,MINI_BATCH_SIZE);
			FCConfig* l18 = new FCConfig(256,MINI_BATCH_SIZE,10);
			ReLUConfig* l19 = new ReLUConfig(10,MINI_BATCH_SIZE);
			config->addLayer(l0);
			config->addLayer(l1);
			config->addLayer(l2);
			//config->addLayer(l3);
			config->addLayer(l4);
			config->addLayer(l5);
			config->addLayer(l6);
			//config->addLayer(l7);
			config->addLayer(l8);
			config->addLayer(l9);
			config->addLayer(l10);
			config->addLayer(l11);
			config->addLayer(l12);
			config->addLayer(l13);
			config->addLayer(l14);
			config->addLayer(l15);
			config->addLayer(l16);
			config->addLayer(l17);
			config->addLayer(l18);
			config->addLayer(l19);
		}
		else if (dataset.compare("ImageNet") == 0)
		{
			NUM_LAYERS = 19;
			// NUM_LAYERS = 17;		//Without BN
			WITH_NORMALIZATION = false;
			CNNConfig* l0 = new CNNConfig(56,56,3,64,7,1,3,MINI_BATCH_SIZE);
			CNNConfig* l1 = new CNNConfig(56,56,64,64,5,1,2,MINI_BATCH_SIZE);
			MaxpoolConfig* l2 = new MaxpoolConfig(56,56,64,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l3 = new ReLUConfig(28*28*64,MINI_BATCH_SIZE);		
			//BNConfig * l4 = new BNConfig(28*28*64,MINI_BATCH_SIZE);

			CNNConfig* l5 = new CNNConfig(28,28,64,128,5,1,2,MINI_BATCH_SIZE);
			MaxpoolConfig* l6 = new MaxpoolConfig(28,28,128,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l7 = new ReLUConfig(14*14*128,MINI_BATCH_SIZE);		
			//BNConfig * l8 = new BNConfig(14*14*128,MINI_BATCH_SIZE);

			CNNConfig* l9 = new CNNConfig(14,14,128,256,3,1,1,MINI_BATCH_SIZE);
			CNNConfig* l10 = new CNNConfig(14,14,256,256,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l11 = new MaxpoolConfig(14,14,256,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l12 = new ReLUConfig(7*7*256,MINI_BATCH_SIZE);

			FCConfig* l13 = new FCConfig(7*7*256,MINI_BATCH_SIZE,1024);
			ReLUConfig* l14 = new ReLUConfig(1024,MINI_BATCH_SIZE);
			FCConfig* l15 = new FCConfig(1024,MINI_BATCH_SIZE,1024);
			ReLUConfig* l16 = new ReLUConfig(1024,MINI_BATCH_SIZE);
			FCConfig* l17 = new FCConfig(1024,MINI_BATCH_SIZE,200);
			ReLUConfig* l18 = new ReLUConfig(200,MINI_BATCH_SIZE);
			config->addLayer(l0);
			config->addLayer(l1);
			config->addLayer(l2);
			config->addLayer(l3);
			//config->addLayer(l4);
			config->addLayer(l5);
			config->addLayer(l6);
			config->addLayer(l7);
			//config->addLayer(l8);
			config->addLayer(l9);
			config->addLayer(l10);
			config->addLayer(l11);
			config->addLayer(l12);
			config->addLayer(l13);
			config->addLayer(l14);
			config->addLayer(l15);
			config->addLayer(l16);
			config->addLayer(l17);
			config->addLayer(l18);
		}
	}
	else if (network.compare("VGG16") == 0)
	{
		if(dataset.compare("MNIST") == 0)
			assert(false && "No VGG16 on MNIST");
		else if (dataset.compare("CIFAR10") == 0)
		{
			NUM_LAYERS = 37;
			WITH_NORMALIZATION = false;
			CNNConfig* l0 = new CNNConfig(32,32,3,64,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l1 = new ReLUConfig(32*32*64,MINI_BATCH_SIZE);		
			CNNConfig* l2 = new CNNConfig(32,32,64,64,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l3 = new MaxpoolConfig(32,32,64,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l4 = new ReLUConfig(16*16*64,MINI_BATCH_SIZE);

			CNNConfig* l5 = new CNNConfig(16,16,64,128,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l6 = new ReLUConfig(16*16*128,MINI_BATCH_SIZE);
			CNNConfig* l7 = new CNNConfig(16,16,128,128,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l8 = new MaxpoolConfig(16,16,128,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l9 = new ReLUConfig(8*8*128,MINI_BATCH_SIZE);

			CNNConfig* l10 = new CNNConfig(8,8,128,256,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l11 = new ReLUConfig(8*8*256,MINI_BATCH_SIZE);
			CNNConfig* l12 = new CNNConfig(8,8,256,256,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l13 = new ReLUConfig(8*8*256,MINI_BATCH_SIZE);
			CNNConfig* l14 = new CNNConfig(8,8,256,256,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l15 = new MaxpoolConfig(8,8,256,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l16 = new ReLUConfig(4*4*256,MINI_BATCH_SIZE);

			CNNConfig* l17 = new CNNConfig(4,4,256,512,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l18 = new ReLUConfig(4*4*512,MINI_BATCH_SIZE);
			CNNConfig* l19 = new CNNConfig(4,4,512,512,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l20 = new ReLUConfig(4*4*512,MINI_BATCH_SIZE);
			CNNConfig* l21 = new CNNConfig(4,4,512,512,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l22 = new MaxpoolConfig(4,4,512,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l23 = new ReLUConfig(2*2*512,MINI_BATCH_SIZE);

			CNNConfig* l24 = new CNNConfig(2,2,512,512,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l25 = new ReLUConfig(2*2*512,MINI_BATCH_SIZE);
			CNNConfig* l26 = new CNNConfig(2,2,512,512,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l27 = new ReLUConfig(2*2*512,MINI_BATCH_SIZE);
			CNNConfig* l28 = new CNNConfig(2,2,512,512,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l29 = new MaxpoolConfig(2,2,512,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l30 = new ReLUConfig(1*1*512,MINI_BATCH_SIZE);

			FCConfig* l31 = new FCConfig(1*1*512,MINI_BATCH_SIZE,4096);
			ReLUConfig* l32 = new ReLUConfig(4096,MINI_BATCH_SIZE);
			FCConfig* l33 = new FCConfig(4096, MINI_BATCH_SIZE, 4096);
			ReLUConfig* l34 = new ReLUConfig(4096, MINI_BATCH_SIZE);
			FCConfig* l35 = new FCConfig(4096, MINI_BATCH_SIZE, 1000);
			ReLUConfig* l36 = new ReLUConfig(1000, MINI_BATCH_SIZE);
			config->addLayer(l0);
			config->addLayer(l1);
			config->addLayer(l2);
			config->addLayer(l3);
			config->addLayer(l4);
			config->addLayer(l5);
			config->addLayer(l6);
			config->addLayer(l7);
			config->addLayer(l8);
			config->addLayer(l9);
			config->addLayer(l10);
			config->addLayer(l11);
			config->addLayer(l12);
			config->addLayer(l13);
			config->addLayer(l14);
			config->addLayer(l15);
			config->addLayer(l16);
			config->addLayer(l17);
			config->addLayer(l18);
			config->addLayer(l19);
			config->addLayer(l20);
			config->addLayer(l21);
			config->addLayer(l22);
			config->addLayer(l23);
			config->addLayer(l24);
			config->addLayer(l25);
			config->addLayer(l26);
			config->addLayer(l27);
			config->addLayer(l28);
			config->addLayer(l29);
			config->addLayer(l30);
			config->addLayer(l31);
			config->addLayer(l32);
			config->addLayer(l33);
			config->addLayer(l34);
			config->addLayer(l35);
			config->addLayer(l36);
		}
		else if (dataset.compare("ImageNet") == 0)
		{
			NUM_LAYERS = 37;
			WITH_NORMALIZATION = false;
			CNNConfig* l0 = new CNNConfig(64,64,3,64,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l1 = new ReLUConfig(64*64*64,MINI_BATCH_SIZE);		
			CNNConfig* l2 = new CNNConfig(64,64,64,64,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l3 = new MaxpoolConfig(64,64,64,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l4 = new ReLUConfig(32*32*64,MINI_BATCH_SIZE);

			CNNConfig* l5 = new CNNConfig(32,32,64,128,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l6 = new ReLUConfig(32*32*128,MINI_BATCH_SIZE);
			CNNConfig* l7 = new CNNConfig(32,32,128,128,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l8 = new MaxpoolConfig(32,32,128,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l9 = new ReLUConfig(16*16*128,MINI_BATCH_SIZE);

			CNNConfig* l10 = new CNNConfig(16,16,128,256,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l11 = new ReLUConfig(16*16*256,MINI_BATCH_SIZE);
			CNNConfig* l12 = new CNNConfig(16,16,256,256,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l13 = new ReLUConfig(16*16*256,MINI_BATCH_SIZE);
			CNNConfig* l14 = new CNNConfig(16,16,256,256,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l15 = new MaxpoolConfig(16,16,256,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l16 = new ReLUConfig(8*8*256,MINI_BATCH_SIZE);

			CNNConfig* l17 = new CNNConfig(8,8,256,512,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l18 = new ReLUConfig(8*8*512,MINI_BATCH_SIZE);
			CNNConfig* l19 = new CNNConfig(8,8,512,512,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l20 = new ReLUConfig(8*8*512,MINI_BATCH_SIZE);
			CNNConfig* l21 = new CNNConfig(8,8,512,512,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l22 = new MaxpoolConfig(8,8,512,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l23 = new ReLUConfig(4*4*512,MINI_BATCH_SIZE);

			CNNConfig* l24 = new CNNConfig(4,4,512,512,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l25 = new ReLUConfig(4*4*512,MINI_BATCH_SIZE);
			CNNConfig* l26 = new CNNConfig(4,4,512,512,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l27 = new ReLUConfig(4*4*512,MINI_BATCH_SIZE);
			CNNConfig* l28 = new CNNConfig(4,4,512,512,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l29 = new MaxpoolConfig(4,4,512,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l30 = new ReLUConfig(2*2*512,MINI_BATCH_SIZE);

			FCConfig* l31 = new FCConfig(2*2*512,MINI_BATCH_SIZE,2048);
			ReLUConfig* l32 = new ReLUConfig(2048,MINI_BATCH_SIZE);
			FCConfig* l33 = new FCConfig(2048, MINI_BATCH_SIZE, 2048);
			ReLUConfig* l34 = new ReLUConfig(2048, MINI_BATCH_SIZE);
			FCConfig* l35 = new FCConfig(2048, MINI_BATCH_SIZE, 200);
			ReLUConfig* l36 = new ReLUConfig(200, MINI_BATCH_SIZE);
			config->addLayer(l0);
			config->addLayer(l1);
			config->addLayer(l2);
			config->addLayer(l3);
			config->addLayer(l4);
			config->addLayer(l5);
			config->addLayer(l6);
			config->addLayer(l7);
			config->addLayer(l8);
			config->addLayer(l9);
			config->addLayer(l10);
			config->addLayer(l11);
			config->addLayer(l12);
			config->addLayer(l13);
			config->addLayer(l14);
			config->addLayer(l15);
			config->addLayer(l16);
			config->addLayer(l17);
			config->addLayer(l18);
			config->addLayer(l19);
			config->addLayer(l20);
			config->addLayer(l21);
			config->addLayer(l22);
			config->addLayer(l23);
			config->addLayer(l24);
			config->addLayer(l25);
			config->addLayer(l26);
			config->addLayer(l27);
			config->addLayer(l28);
			config->addLayer(l29);
			config->addLayer(l30);
			config->addLayer(l31);
			config->addLayer(l32);
			config->addLayer(l33);
			config->addLayer(l34);
			config->addLayer(l35);
			config->addLayer(l36);
		}
	}
	else
		assert(false && "Only SecureML, Sarda, Gazelle, LeNet, AlexNet, and VGG16 Networks supported");
}

void runOnly(NeuralNetwork* net, size_t l, string what, string& network)
{
	size_t total_layers = net->layers.size();
	assert((l >= 0 and l < total_layers) && "Incorrect layer number for runOnly"); 
	network = network + " L" + std::to_string(l) + " " + what;

	if (what.compare("F") == 0)
	{
		if (l == 0)
			net->layers[0]->forward(net->inputData);
		else
			net->layers[l]->forward(*(net->layers[l-1]->getActivation()));
	}
	else if (what.compare("D") == 0)
	{
		if (l != 0)
			net->layers[l]->computeDelta(*(net->layers[l-1]->getDelta()));	
	}
	else if (what.compare("U") == 0)
	{
		if (l == 0)
			net->layers[0]->updateEquations(net->inputData);
		else
			net->layers[l]->updateEquations(*(net->layers[l-1]->getActivation()));
	}
	else
		assert(false && "Only F,D or U allowed in runOnly");
}






/********************* COMMUNICATION AND HELPERS *********************/

void start_m()
{
	// cout << endl;
	start_time();
	start_communication();
}

void end_m(string str)
{
	end_time(str);
	pause_communication();
	aggregateCommunication();
	end_communication(str);
}

void start_time()
{
	if (alreadyMeasuringTime)
	{
		cout << "Nested timing measurements" << endl;
		exit(-1);
	}

	tStart = clock();
	clock_gettime(CLOCK_REALTIME, &requestStart);
	alreadyMeasuringTime = true;
}

void end_time(string str)
{
	if (!alreadyMeasuringTime)
	{
		cout << "start_time() never called" << endl;
		exit(-1);
	}

	clock_gettime(CLOCK_REALTIME, &requestEnd);
	cout << "----------------------------------------------" << endl;
	cout << "Wall Clock time for " << str << ": " << diff(requestStart, requestEnd) << " sec\n";
	cout << "CPU time for " << str << ": " << (double)(clock() - tStart)/CLOCKS_PER_SEC << " sec\n";
	cout << "----------------------------------------------" << endl;	
	alreadyMeasuringTime = false;
}


void start_rounds()
{
	if (alreadyMeasuringRounds)
	{
		cout << "Nested round measurements" << endl;
		exit(-1);
	}

	roundComplexitySend = 0;
	roundComplexityRecv = 0;
	alreadyMeasuringRounds = true;
}

void end_rounds(string str)
{
	if (!alreadyMeasuringTime)
	{
		cout << "start_rounds() never called" << endl;
		exit(-1);
	}

	cout << "----------------------------------------------" << endl;
	cout << "Send Round Complexity of " << str << ": " << roundComplexitySend << endl;
	cout << "Recv Round Complexity of " << str << ": " << roundComplexityRecv << endl;
	cout << "----------------------------------------------" << endl;	
	alreadyMeasuringRounds = false;
}

void aggregateCommunication()
{
	vector<myType> vec(4, 0), temp(4, 0);
	vec[0] = commObject.getSent();
	vec[1] = commObject.getRecv();
	vec[2] = commObject.getRoundsSent();
	vec[3] = commObject.getRoundsRecv();

	if (partyNum == PARTY_B or partyNum == PARTY_C)
		sendVector<myType>(vec, PARTY_A, 4);

	if (partyNum == PARTY_A)
	{
		receiveVector<myType>(temp, PARTY_B, 4);
		for (size_t i = 0; i < 4; ++i)
			vec[i] = temp[i] + vec[i];
		receiveVector<myType>(temp, PARTY_C, 4);
		for (size_t i = 0; i < 4; ++i)
			vec[i] = temp[i] + vec[i];
	}

	if (partyNum == PARTY_A)
	{
		cout << "----------------------------------------------" << endl;
		cout << "Total communication: " << (float)vec[0]/1000000 << "MB (sent) and " << (float)vec[1]/1000000 << "MB (recv)\n";
		cout << "Total calls: " << vec[2] << " (sends) and " << vec[3] << " (recvs)" << endl;
		cout << "----------------------------------------------" << endl;
	}
}


void print_usage (const char * bin) 
{
    cout << "Usage: ./" << bin << " PARTY_NUM IP_ADDR_FILE AES_SEED_INDEP AES_SEED_NEXT AES_SEED_PREV" << endl;
    cout << endl;
    cout << "Required Arguments:\n";
    cout << "PARTY_NUM			Party Identifier (0,1, or 2)\n";
    cout << "IP_ADDR_FILE		\tIP Address file (use makefile for automation)\n";
    cout << "AES_SEED_INDEP		\tAES seed file independent\n";
    cout << "AES_SEED_NEXT		\t \tAES seed file next\n";
    cout << "AES_SEED_PREV		\t \tAES seed file previous\n";
    cout << endl;
    cout << "Report bugs to swagh@princeton.edu" << endl;
    exit(-1);
}

double diff(timespec start, timespec end)
{
    timespec temp;

    if ((end.tv_nsec-start.tv_nsec)<0)
    {
            temp.tv_sec = end.tv_sec-start.tv_sec-1;
            temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    }
    else 
    {
            temp.tv_sec = end.tv_sec-start.tv_sec;
            temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp.tv_sec + (double)temp.tv_nsec/NANOSECONDS_PER_SEC;
}


void deleteObjects()
{
	//close connection
	for (int i = 0; i < NUM_OF_PARTIES; i++)
	{
		if (i != partyNum)
		{
			delete communicationReceivers[i];
			delete communicationSenders[i];
		}
	}
	delete[] communicationReceivers;
	delete[] communicationSenders;
	delete[] addrs;
}


/************************ AlexNet on ImageNet ************************/
// NUM_LAYERS = 21;
// WITH_NORMALIZATION = false;
// CNNConfig* l0 = new CNNConfig(227,227,3,96,11,4,0,MINI_BATCH_SIZE);
// MaxpoolConfig* l1 = new MaxpoolConfig(55,55,96,3,2,MINI_BATCH_SIZE);
// ReLUConfig* l2 = new ReLUConfig(27*27*96,MINI_BATCH_SIZE);		
// BNConfig * l3 = new BNConfig(27*27*96,MINI_BATCH_SIZE);

// CNNConfig* l4 = new CNNConfig(27,27,96,256,5,1,2,MINI_BATCH_SIZE);
// MaxpoolConfig* l5 = new MaxpoolConfig(27,27,256,3,2,MINI_BATCH_SIZE);
// ReLUConfig* l6 = new ReLUConfig(13*13*256,MINI_BATCH_SIZE);		
// BNConfig * l7 = new BNConfig(13*13*256,MINI_BATCH_SIZE);

// CNNConfig* l8 = new CNNConfig(13,13,256,384,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l9 = new ReLUConfig(13*13*384,MINI_BATCH_SIZE);
// CNNConfig* l10 = new CNNConfig(13,13,384,384,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l11 = new ReLUConfig(13*13*384,MINI_BATCH_SIZE);
// CNNConfig* l12 = new CNNConfig(13,13,384,256,3,1,1,MINI_BATCH_SIZE);
// MaxpoolConfig* l13 = new MaxpoolConfig(13,13,256,3,2,MINI_BATCH_SIZE);
// ReLUConfig* l14 = new ReLUConfig(6*6*256,MINI_BATCH_SIZE);

// FCConfig* l15 = new FCConfig(6*6*256,MINI_BATCH_SIZE,4096);
// ReLUConfig* l16 = new ReLUConfig(4096,MINI_BATCH_SIZE);
// FCConfig* l17 = new FCConfig(4096,MINI_BATCH_SIZE,4096);
// ReLUConfig* l18 = new ReLUConfig(4096,MINI_BATCH_SIZE);
// FCConfig* l19 = new FCConfig(4096,MINI_BATCH_SIZE,1000);
// ReLUConfig* l20 = new ReLUConfig(1000,MINI_BATCH_SIZE);
// config->addLayer(l0);
// config->addLayer(l1);
// config->addLayer(l2);
// config->addLayer(l3);
// config->addLayer(l4);
// config->addLayer(l5);
// config->addLayer(l6);
// config->addLayer(l7);
// config->addLayer(l8);
// config->addLayer(l9);
// config->addLayer(l10);
// config->addLayer(l11);
// config->addLayer(l12);
// config->addLayer(l13);
// config->addLayer(l14);
// config->addLayer(l15);
// config->addLayer(l16);
// config->addLayer(l17);
// config->addLayer(l18);
// config->addLayer(l19);
// config->addLayer(l20);


/************************ VGG16 on ImageNet ************************/
// NUM_LAYERS = 37;
// WITH_NORMALIZATION = false;
// CNNConfig* l0 = new CNNConfig(224,224,3,64,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l1 = new ReLUConfig(224*224*64,MINI_BATCH_SIZE);		
// CNNConfig* l2 = new CNNConfig(224,224,64,64,3,1,1,MINI_BATCH_SIZE);
// MaxpoolConfig* l3 = new MaxpoolConfig(224,224,64,2,2,MINI_BATCH_SIZE);
// ReLUConfig* l4 = new ReLUConfig(112*112*64,MINI_BATCH_SIZE);

// CNNConfig* l5 = new CNNConfig(112,112,64,128,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l6 = new ReLUConfig(112*112*128,MINI_BATCH_SIZE);
// CNNConfig* l7 = new CNNConfig(112,112,128,128,3,1,1,MINI_BATCH_SIZE);
// MaxpoolConfig* l8 = new MaxpoolConfig(112,112,128,2,2,MINI_BATCH_SIZE);
// ReLUConfig* l9 = new ReLUConfig(56*56*128,MINI_BATCH_SIZE);

// CNNConfig* l10 = new CNNConfig(56,56,128,256,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l11 = new ReLUConfig(56*56*256,MINI_BATCH_SIZE);
// CNNConfig* l12 = new CNNConfig(56,56,256,256,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l13 = new ReLUConfig(56*56*256,MINI_BATCH_SIZE);
// CNNConfig* l14 = new CNNConfig(56,56,256,256,3,1,1,MINI_BATCH_SIZE);
// MaxpoolConfig* l15 = new MaxpoolConfig(56,56,256,2,2,MINI_BATCH_SIZE);
// ReLUConfig* l16 = new ReLUConfig(28*28*256,MINI_BATCH_SIZE);

// CNNConfig* l17 = new CNNConfig(28,28,256,512,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l18 = new ReLUConfig(28*28*512,MINI_BATCH_SIZE);
// CNNConfig* l19 = new CNNConfig(28,28,512,512,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l20 = new ReLUConfig(28*28*512,MINI_BATCH_SIZE);
// CNNConfig* l21 = new CNNConfig(28,28,512,512,3,1,1,MINI_BATCH_SIZE);
// MaxpoolConfig* l22 = new MaxpoolConfig(28,28,512,2,2,MINI_BATCH_SIZE);
// ReLUConfig* l23 = new ReLUConfig(14*14*512,MINI_BATCH_SIZE);

// CNNConfig* l24 = new CNNConfig(14,14,512,512,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l25 = new ReLUConfig(14*14*512,MINI_BATCH_SIZE);
// CNNConfig* l26 = new CNNConfig(14,14,512,512,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l27 = new ReLUConfig(14*14*512,MINI_BATCH_SIZE);
// CNNConfig* l28 = new CNNConfig(14,14,512,512,3,1,1,MINI_BATCH_SIZE);
// MaxpoolConfig* l29 = new MaxpoolConfig(14,14,512,2,2,MINI_BATCH_SIZE);
// ReLUConfig* l30 = new ReLUConfig(7*7*512,MINI_BATCH_SIZE);

// FCConfig* l31 = new FCConfig(7*7*512,MINI_BATCH_SIZE,4096);
// ReLUConfig* l32 = new ReLUConfig(4096,MINI_BATCH_SIZE);
// FCConfig* l33 = new FCConfig(4096, MINI_BATCH_SIZE, 4096);
// ReLUConfig* l34 = new ReLUConfig(4096, MINI_BATCH_SIZE);
// FCConfig* l35 = new FCConfig(4096, MINI_BATCH_SIZE, 1000);
// ReLUConfig* l36 = new ReLUConfig(1000, MINI_BATCH_SIZE);
// config->addLayer(l0);
// config->addLayer(l1);
// config->addLayer(l2);
// config->addLayer(l3);
// config->addLayer(l4);
// config->addLayer(l5);
// config->addLayer(l6);
// config->addLayer(l7);
// config->addLayer(l8);
// config->addLayer(l9);
// config->addLayer(l10);
// config->addLayer(l11);
// config->addLayer(l12);
// config->addLayer(l13);
// config->addLayer(l14);
// config->addLayer(l15);
// config->addLayer(l16);
// config->addLayer(l17);
// config->addLayer(l18);
// config->addLayer(l19);
// config->addLayer(l20);
// config->addLayer(l21);
// config->addLayer(l22);
// config->addLayer(l23);
// config->addLayer(l24);
// config->addLayer(l25);
// config->addLayer(l26);
// config->addLayer(l27);
// config->addLayer(l28);
// config->addLayer(l29);
// config->addLayer(l30);
// config->addLayer(l31);
// config->addLayer(l32);
// config->addLayer(l33);
// config->addLayer(l34);
// config->addLayer(l35);
// config->addLayer(l36);
