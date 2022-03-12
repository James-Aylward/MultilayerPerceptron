#include "../include/MultilayerPerceptron.h"
#include <vector>
#include <math.h>
#include <random>
#include <chrono>
#include <sstream>
#include <numeric>
#include <fstream>
#include <algorithm>

using namespace std;

// Constructor
MultilayerPerceptron::MultilayerPerceptron(vector<int> configuration, int activationFunction)
{
    _activationFunction = activationFunction;

    mt19937 gen(chrono::high_resolution_clock::now().time_since_epoch().count());

    // Loop through each layer
    for (int layer = 1; layer < configuration.size(); layer++)
    {
        // Create a layer vector with an amount of perceptrons equal to the configuration.
        // Each perceptron has inputs equal to the amount of perceptrons in the previous layer

        vector<Perceptron> currentLayer;
        for (int i = 0; i < configuration[layer]; i++)                               // Loop through perceptrons
        {                                                                            // todo removed sqrt 2/N
            normal_distribution<float> dis(0, sqrt(2.0 / configuration[layer - 1])); // Create a new normal distribution with mean 0, variance is He initalization
            vector<float> weights;                                                   // Create a new vector to store the weights for the perceptron we are about to generate

            // For each input, create a weight and add it to the weight array
            for (int j = 0; j < configuration[layer - 1]; j++)
                weights.push_back(dis(gen)); // Add random weight (He initialization)

            currentLayer.push_back(Perceptron(weights));
        }

        _network.push_back(currentLayer);
    }
}

// Predict an output
vector<float> MultilayerPerceptron::predict(vector<float> inputs, int postFunction, bool storeData)
{

    if (storeData)
    {
        _lastSumBiases.clear();
        _lastOutputs.clear();

        _lastOutputs.push_back(inputs);
    }

    vector<float> previousLayerOutputs = inputs, currentLayerOutputs;

    // Loop through each layer in the network
    for (int layer = 0; layer < _network.size(); layer++)
    {

        vector<float> layerSumBiases;

        // Loop through each perceptron in the layer
        for (int perceptron = 0; perceptron < _network[layer].size(); perceptron++)
        {

            // For each perceptron, predict based on the previous layers outputs and store in a vector
            float prediction, sum = 1 * _network[layer][perceptron].weights[0]; // Sum is initially bias // TODO change bias

            for (int i = 0; i != previousLayerOutputs.size(); i++)
                sum += previousLayerOutputs[i] * _network[layer][perceptron].weights[i + 1]; // Summation of inputs * weights

            layerSumBiases.push_back(sum);
            currentLayerOutputs.push_back(activationHandler(sum));
        }

        if (storeData)
        {
            _lastOutputs.push_back(currentLayerOutputs);
            _lastSumBiases.push_back(layerSumBiases);
        }

        // Use the newly calculated outputs as inputs for the next layer
        previousLayerOutputs = currentLayerOutputs;
        currentLayerOutputs.clear();
    }

    switch (postFunction)
    {
    case RAW:
        return previousLayerOutputs;
    case ARGMAX:
    {
        vector<float> out(previousLayerOutputs.size(), 0);
        out[distance(previousLayerOutputs.begin(), max_element(previousLayerOutputs.begin(), previousLayerOutputs.end()))] = 1;
        return out;
    }
    case MAX_INDEX:
        return {(float)distance(previousLayerOutputs.begin(), max_element(previousLayerOutputs.begin(), previousLayerOutputs.end()))};
    case SOFTMAX:
    {
        vector<float> out;
        float g = 0;

        for (int i = 0; i < previousLayerOutputs.size(); i++)
            g += exp(previousLayerOutputs[i]);

        for (int i = 0; i < previousLayerOutputs.size(); i++)
            out.push_back(exp(previousLayerOutputs[i]) / g);

        return out;
    }
    default:
        return previousLayerOutputs;
    }
}

// The scary function (calculates some desired changes for network weights)
void MultilayerPerceptron::backpropogate(vector<float> trainingInputs, vector<float> label)
{
    // Get the MSE and store data from a prediction
    float cost = mserror(predict(trainingInputs, RAW, true), label);

    // Loop through layer backwards
    for (int l = _network.size() - 1; l >= 0; l--)
    {
        // Loop through perceptrons in layer
        for (int p = 0; p < _network[l].size(); p++)
        {
            float delta = 0;

            // We are on the last layer
            if (l == _network.size() - 1)
                delta = (_lastOutputs[l + 1][p] - label[p]) * activationDerivativeHandler(_lastSumBiases[l][p]);

            // Not the last layer
            else
            {
                for (int i = 0; i < _network[l + 1].size(); i++)
                    delta += _network[l + 1][i].lastDelta * _network[l + 1][i].weights[p + 1];
                delta = delta * activationDerivativeHandler(_lastSumBiases[l][p]);
            }

            _network[l][p].lastDelta = delta;

            for (int w = 0; w < _network[l][p].weights.size(); w++)
            {
                float associatedOutput = (w == 0) ? 1 : _lastOutputs[l][w - 1]; // TODO change bias
                _network[l][p].desiredChanges[w].push_back(-delta * associatedOutput);
            }
        }
    }
}

// Computes a step
void MultilayerPerceptron::step(float learningRate)
{
    // Loop through every weight in the network
    for (int l = 0; l < _network.size(); l++)
    {
        for (int p = 0; p < _network[l].size(); p++)
        {
            for (int w = 0; w < _network[l][p].weights.size(); w++)
            {
                // Sum desired changes for weight
                float changes = accumulate(_network[l][p].desiredChanges[w].begin(), _network[l][p].desiredChanges[w].end(), 0.0f);
                _network[l][p].weights[w] += learningRate * changes / _network[l][p].desiredChanges[w].size();
                _network[l][p].desiredChanges[w].clear();
            }
        }
    }
}

// Handles all the activation functions
float MultilayerPerceptron::activationHandler(float sum)
{
    switch (_activationFunction)
    {
    case HEAVISIDE:
        return sum >= 0; // Return 1 if zero or greater and 0 is less than zero
    case SIGMOID:
        return 1 / (1 + exp(-sum)); // S curve
    case RELU:
        return (sum <= 0 ? 0 : sum);
    case LEAKY_RELU:
        return (sum <= 0 ? 0.01 * sum : sum); // Alpha is 0.01
    default:
        return false;
    }
}

// Handles all the derivatives of the activation functions
float MultilayerPerceptron::activationDerivativeHandler(float sum)
{
    switch (_activationFunction)
    {
    case HEAVISIDE:
        return 0;
    case SIGMOID:
        return activationHandler(sum) * (1 - activationHandler(sum));
    case RELU:
        return sum > 0;
    case LEAKY_RELU:
        return (sum > 0 ? 1 : 0.01);
    default:
        return false;
    }
}

// Returns the MSE of a forward pass
float MultilayerPerceptron::mserror(vector<float> predictedOutputs, vector<float> label)
{
    float cost = 0;

    for (int i = 0; i < predictedOutputs.size(); i++)
        cost += (label[i] - predictedOutputs[i]) * (label[i] - predictedOutputs[i]);

    return cost / 2; // TODO must be average later on because of reasons
}

// Save the network parameters to a CSV file
void MultilayerPerceptron::saveNetwork(string filename)
{
    ofstream file(filename);

    // Write activation
    file << _activationFunction << "\n";

    // Loop through layers
    for (int l = 0; l < _network.size(); l++)
    {
        // Loop through perceptrons
        for (int p = 0; p < _network[l].size(); p++)
        {
            // Loop through weights
            for (int w = 0; w < _network[l][p].weights.size(); w++)
            {
                file << _network[l][p].weights[w] << (w == _network[l][p].weights.size() - 1 ? "\n" : ",");
            }
        }
        file << "\n";
    }
    file.close();
}

// Load the network parameters from a CSV file
void MultilayerPerceptron::loadNetwork(string filename)
{
    ifstream file(filename);

    string line;
    float value;

    int l, p;
    l = p = 0;

    // Extract activation function
    getline(file, line);
    stringstream ss(line);
    ss >> _activationFunction;

    // Loop through the rows
    while (getline(file, line))
    {
        ss = stringstream(line);
        vector<float> weights;

        // Loop through the cell values
        while (ss >> value)
        {
            if (ss.peek() == ',')
                ss.ignore();

            weights.push_back(value);
        }

        // Empty line, meaning next layer
        if (weights.size() == 0)
        {
            l++;
            p = 0;
        }
        else
        {
            _network[l][p].weights = weights;
            p++;
        }
    }

    file.close();
}
