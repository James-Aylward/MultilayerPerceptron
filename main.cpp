#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <string>
#include <numeric>
#include "include/MultilayerPerceptron.h"
#include "include/mnist_loader.h"

using namespace std;

void printImage(vector<float> image)
{
    cout << "Image: " << endl;
    for (int y = 0; y < 28; ++y)
    {
        for (int x = 0; x < 28; ++x)
        {
            cout << ((image[y * 28 + x] == 0.0) ? ' ' : '*');
            // cout << image[y * 28 + x] << endl;
        }
        cout << endl;
    }
}

int main()
{
    MultilayerPerceptron brain({784, 32, 16, 10}, LEAKY_RELU);

    // Load training data set - 59999 max
    int size;
    cout << "Please enter number of training images to load (max 60000): ";
    cin >> size;
    cout << "\nLoading MNIST data set...";
    mnist_loader data("data/train-images.idx3-ubyte",
                      "data/train-labels.idx1-ubyte", size);
    cout << " done\n";

    // Load test data set - 10000 digits
    cout << "Loading MNIST test data set...";
    mnist_loader test("data/t10k-images.idx3-ubyte",
                      "data/t10k-labels.idx1-ubyte");
    cout << " done\n\n";

    cout << "Train or load...? : ";
    char loadtrain;
    cin >> loadtrain;
    if (loadtrain == 't')
    {
        // Generate our random numbers
        vector<int> batchNumbers(size / 10);
        iota(batchNumbers.begin(), batchNumbers.end(), 0);

        // Training
        int epoch;
        cout << "Please enter epoch: ";
        cin >> epoch;

        std::random_device random_dev;
        std::mt19937 generator(random_dev());

        for (int e = 0; e < epoch; e++)
        {

            shuffle(batchNumbers.begin(), batchNumbers.end(), generator);

            for (int b = 0; b < batchNumbers.size(); b++)
            {
                for (int i = 0; i < 10; i++)
                {
                    vector<float> label(10, 0);
                    label[data.labels(batchNumbers[b] * 10 + i)] = 1;
                    brain.backpropogate(data.images(batchNumbers[b] * 10 + i), label);
                }
                brain.step(0.2);
            }
            cout << ".";
            brain.saveNetwork("mnist_network.csv");
        }
    }
    else
    {
        string file;
        cin >> file;
        brain.loadNetwork(file);
    }

    int correct = 0;
    vector<int> wrongDigitIndex;
    for (int i = 0; i < 10000; i++)
    {
        vector<float> prediction = brain.predict(test.images(i), RAW);
        int answer = distance(prediction.begin(), max_element(prediction.begin(), prediction.end()));

        if (answer == test.labels(i))
            correct++;
        else
            wrongDigitIndex.push_back(i);
    }

    cout << "||||||||||||| TEST IS DONE |||||||||||||\nPercentage correct: " << (double)(correct / 10000.0f) * 100 << "%\n\n";

    // for (int i = 0; i < 10; i++) // wrongDigitIndex.size()
    // {
    //     cout << "---------------------------------\n";
    //     printImage(test.images(wrongDigitIndex[i]));
    //     cout << "Label: " << test.labels(wrongDigitIndex[i]) << "\nPrediction: ";

    //     cout << brain.predict(test.images(wrongDigitIndex[i]), MAX_INDEX)[0] << endl;

    //     vector<float> prediction = brain.predict(test.images(wrongDigitIndex[i]), SOFTMAX);
    //     for (int j = 0; j < prediction.size(); j++)
    //         cout << j << ": " << prediction[j] * 100 << "%" << endl;

    //     cout << endl
    //          << endl;
    // }

    // Prediction
    while (true)
    {
        int image;
        cout << "\n\nImage Index: ";
        cin >> image;

        printImage(data.images(image));
        cout << "Label: " << data.labels(image) << "\nPrediction: ";

        vector<float> prediction = brain.predict(data.images(image), RAW);
        cout << distance(prediction.begin(), max_element(prediction.begin(), prediction.end()));
    }

    return 0;
}