#include <iostream>
#include <ctime>
#include "Eigen/Dense"
#include "Perceptronchik.h"

using namespace Eigen;

int main() {
    const int num_samples = 1000000;
    const int num_features = 30;
    const int batch_size = 100000;
    const double learning_rate = 0.2;
    const int num_epochs = 10;

    std::srand(static_cast<unsigned>(std::time(0)));

    MatrixXd train_features(num_samples, num_features);
    MatrixXd test_features(num_samples, num_features);
    train_features.setRandom();
    test_features.setRandom();

    MatrixXd train_features_with_bias(num_samples, num_features + 1);
    train_features_with_bias << train_features, VectorXd::Ones(num_samples);

    MatrixXd test_features_with_bias(num_samples, num_features + 1);
    test_features_with_bias << test_features, VectorXd::Ones(num_samples);


    MatrixXd true_weights(num_features + 1, 1);
    true_weights.setRandom();
    true_weights *= 10;


    MatrixXd train_labels = (train_features_with_bias * true_weights).unaryExpr(
        [](double x) { return x < 0 ? 0.0 : 1.0; });

    MatrixXd test_labels = (test_features_with_bias * true_weights).unaryExpr(
        [](double x) { return x < 0 ? 0.0 : 1.0; });

    train_features *= 5;
    test_features *= 5;

    Perceptron perceptron(num_features, num_samples);
    perceptron.train(train_features, train_labels, test_features, test_labels,
        batch_size, learning_rate, num_epochs);

    perceptron.evaluate(test_features, test_labels);

    return 0;
}
