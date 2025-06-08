#pragma once
#include <iostream>
#include <cmath>
#include "Eigen/Dense"

using namespace Eigen;

class Perceptron {
private:
    MatrixXd weights_; 
    int input_size_;   
    int num_samples_;   

    static double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

public:

    Perceptron(int input_size, int num_samples)
        : input_size_(input_size),
        num_samples_(num_samples),
        weights_(input_size + 1, 1) {
        weights_.setRandom();  
    }

    void train(const MatrixXd& train_features, const MatrixXd& train_labels,
        const MatrixXd& test_features, const MatrixXd& test_labels,
        int batch_size, double learning_rate, int num_epochs) {

        MatrixXd train_features_with_bias(num_samples_, input_size_ + 1);
        train_features_with_bias << train_features, VectorXd::Ones(num_samples_);
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            for (int i = 0; i < num_samples_ / batch_size; ++i) {
                int start_row = i * batch_size;
                MatrixXd batch_features = train_features_with_bias.block(
                    start_row, 0, batch_size, input_size_ + 1);
                MatrixXd batch_labels = train_labels.block(
                    start_row, 0, batch_size, 1);
                MatrixXd z = batch_features * weights_;
                MatrixXd predictions = z.unaryExpr(&Perceptron::sigmoid);
                MatrixXd error = predictions - batch_labels;
                MatrixXd gradient = batch_features.transpose() * error;
                weights_ -= (learning_rate / batch_size) * gradient;
                if (i % 10 == 0) {
                    evaluate(test_features, test_labels);
                }
            }
            std::cout << "Epoch " << epoch + 1 << " completed" << std::endl;
        }
    }

    void evaluate(const MatrixXd& features, const MatrixXd& labels) const {
        MatrixXd features_with_bias(features.rows(), input_size_ + 1);
        features_with_bias << features, VectorXd::Ones(features.rows());
        MatrixXd predictions = features_with_bias * weights_;
        MatrixXd binary_predictions = predictions.unaryExpr(
            [](double x) { return x >= 0 ? 1.0 : 0.0; });
        double accuracy = (binary_predictions.array() == labels.array()).cast<double>().mean();
        std::cout << "Accuracy: " << accuracy * 100 << "%, "
            << "Error: " << (1.0 - accuracy) * 100 << "%" << std::endl;
    }

};
