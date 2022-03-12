#pragma once

#include <vector>

class Perceptron
{
    public: 

                                        Perceptron(std::vector<float> givenWeights);
        float                           lastDelta;
        std::vector<float>              weights;
        std::vector<std::vector<float>> desiredChanges;

};