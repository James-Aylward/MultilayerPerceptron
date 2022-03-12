#pragma once

#include "Perceptron.h"
#include <vector>
#include <string>

// Activation Functions
#define HEAVISIDE 1
#define SIGMOID 2
#define RELU 3
#define LEAKY_RELU 4

// Post Processing Functions
#define RAW 1
#define ARGMAX 2
#define MAX_INDEX 3
#define SOFTMAX 4

class MultilayerPerceptron
{
    public:
    
                                                MultilayerPerceptron(std::vector<int> configuration, int activationFunction);
        std::vector<float>                      predict(std::vector<float> inputs, int postFunction, bool storeData = false);
        void                                    backpropogate(std::vector<float> trainingInputs, std::vector<float> label);
        void                                    step(float learningRate);
        void                                    saveNetwork(std::string filename);
        void                                    loadNetwork(std::string filename);

    private:
        std::vector<std::vector<Perceptron>>    _network;
        std::vector<std::vector<float>>         _lastOutputs;
        std::vector<std::vector<float>>         _lastSumBiases;
        int                                     _activationFunction;
        float                                   activationHandler(float sum);
        float                                   activationDerivativeHandler(float sum);
        float                                   mserror(std::vector<float> predictedOutputs, std::vector<float> label);
};