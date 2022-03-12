#include "../include/Perceptron.h"
#include <stdexcept>
#include <math.h>

using namespace std;

// Constructor
Perceptron::Perceptron(vector<float> givenWeights)
{
    weights = givenWeights;
    weights.insert(weights.begin(), 0);
    desiredChanges.assign(givenWeights.size() + 1, {}); // +1 for bias
}