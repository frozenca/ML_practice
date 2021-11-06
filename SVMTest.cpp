#include "Matrix/Matrix.h"
#include "ML/SVM/SupportVectorMachine.h"
#include <iostream>
#include <memory>
#include <random>

namespace fc = frozenca;

int main() {
    std::shared_ptr<fc::Kernel> rbf = std::make_shared<fc::KernelRBF>();
    std::unique_ptr<fc::SVMModel> sm = std::make_unique<fc::SVMCSVC>(rbf);

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
    sm->fit(data, params);

    std::vector<std::vector<float>> X_;

    for (std::size_t i = 0; i < 10; ++i) {
        auto height = male_height(gen);
        auto weight = male_weight(gen);
        X_.push_back({height, weight});
    }

    auto y_ = sm->predict(X_);


}
