#ifndef FROZENCA_SVMMODEL_H
#define FROZENCA_SVMMODEL_H

#include "SVMData.h"
#include "SVMKernel.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace frozenca {

struct SVMParams {
    std::shared_ptr<Kernel> kernel_;
    float C = 1.0f;
    float nu = 0.5f;
    float p = 0.1f;
    std::vector<float> weight_scale;
    bool shrinking = true; // use the shrinking heuristics
    float eps = 1e-3;
    bool compute_probability = false;
};

class SVMModel {
protected:
    // actual model parameters
    // support vectors for each class (size : k)
    std::unordered_map<std::size_t, std::vector<std::vector<float>>> SV_;
    // coefficients for SVs in decision functions. classifier between class (i, j) => SV_coeff_[i * k + j]
    std::unordered_map<std::size_t, std::vector<float>> SV_coeff_;
    // constants in decision functions (size: k * (k - 1) / 2)
    std::vector<float> rho_;

    std::vector<float> prob_A_;
    std::vector<float> prob_B_;

public:
    std::shared_ptr<Kernel> kernel_ = nullptr;
    friend class SVMTwoClass;

    SVMModel(std::shared_ptr<Kernel> kernel) : kernel_(std::move(kernel)) {}
    virtual ~SVMModel() = default;

    virtual void fit(const SVMTrainData& svm_data, const SVMParams& params) = 0;
    virtual std::vector<float> crossValidate(const SVMTrainData& svm_data, const SVMParams& params, std::size_t count_fold) final;

    [[nodiscard]] virtual float predict(const std::vector<float>& sample) const = 0;
    [[nodiscard]] virtual std::vector<float> predict(const std::vector<std::vector<float>>& samples) const final {
        std::vector<float> results;
        for (const auto& sample: samples) {
            results.push_back(predict(sample));
        }
        return results;
    }

};

std::vector<float> SVMModel::crossValidate(const SVMTrainData& svm_data, const SVMParams& params, std::size_t count_fold) {
    const std::size_t l = svm_data.n_;
    if (count_fold > l) {
        count_fold = l;
        std::cerr << "WARNING: #folds > #data. Will use #folds = #data (LOOCV)\n";
    }
    std::vector<std::size_t> fold_indices(l);
    for (std::size_t f = 0; f < count_fold; ++f) {
        for (std::size_t i = f * l / count_fold; i < (f + 1) * l / count_fold; ++i) {
            fold_indices[i] = f;
        }
    }
    std::mt19937 gen(std::random_device{}());
    std::ranges::shuffle(fold_indices, gen);
    std::unordered_map<std::size_t, std::unordered_set<std::size_t>> fold_sets;
    for (std::size_t i = 0; i < l; ++i) {
        fold_sets[fold_indices[i]].insert(i);
    }

    std::vector<float> target(l);

    for (std::size_t f = 0; f < count_fold; ++f) {
        // leave f-th fold out
        SubSamples X_without_f;
        std::vector<float> y_without_f;
        for (std::size_t fd = 0; fd < count_fold; ++fd) {
            if (fd == f) {
                continue;
            }
            auto& fold_fd = fold_sets[fd];
            for (auto fd_sample : fold_fd) {
                X_without_f.push_back(svm_data.getXRef(fd_sample));
                y_without_f.push_back(svm_data.y_[fd_sample]);
            }
        }

        SVMTrainData CV_data_ffold (X_without_f, y_without_f);
        fit(CV_data_ffold, params);
        auto& fold_f = fold_sets[f];
        for (auto f_sample : fold_f) {
            target[f_sample] = predict(svm_data.getX(f_sample));
        }
    }
    return target;
}

} // namespace frozenca

#endif //FROZENCA_SVMMODEL_H
