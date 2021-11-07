#include "Matrix/Matrix.h"
#include "ML/SVM/SupportVectorMachine.h"
#include <iostream>
#include <memory>
#include <random>

namespace fc = frozenca;

int main() {
    std::unique_ptr<fc::SVMModel> sm = std::make_unique<fc::SVMCSVC>(std::make_unique<fc::KernelRBF>());

    std::mt19937 gen(std::random_device{}());

    constexpr std::size_t num_samples = 200;

    std::normal_distribution<float> male_height (177.0f, 8.0f);
    std::normal_distribution<float> male_weight (70.0f, 6.0f);

    std::normal_distribution<float> female_height (162.0f, 7.0f);
    std::normal_distribution<float> female_weight (53.0f, 5.5f);

    std::vector<std::vector<float>> X;
    std::vector<float> y;

    for (std::size_t i = 0; i < num_samples; ++i) {
        auto height = male_height(gen);
        auto weight = male_weight(gen);
        X.push_back(std::vector<float>{height, weight});
        y.push_back(+1);
    }
    for (std::size_t i = 0; i < num_samples; ++i) {
        auto height = female_height(gen);
        auto weight = female_weight(gen);
        X.push_back(std::vector<float>{height, weight});
        y.push_back(-1);
    }
    fc::SVMTrainData data(X, y);
    fc::SVMParams params;
    params.shrinking = false;
    sm->fit(data, params);

    std::vector<std::vector<float>> X1;

    for (std::size_t i = 0; i < 10; ++i) {
        auto height = male_height(gen);
        auto weight = male_weight(gen);
        X1.push_back({height, weight});
    }

    auto y1 = sm->predict(X1);

    for (auto y_pred : y1) {
        std::cout << y_pred << ' ';
    }
    std::cout << '\n';

    std::vector<std::vector<float>> X2;

    for (std::size_t i = 0; i < 10; ++i) {
        auto height = female_height(gen);
        auto weight = female_weight(gen);
        X2.push_back({height, weight});
    }

    auto y2 = sm->predict(X2);

    for (auto y_pred : y2) {
        std::cout << y_pred << ' ';
    }
    std::cout << '\n';

}
