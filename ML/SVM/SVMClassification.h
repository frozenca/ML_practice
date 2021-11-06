#ifndef FROZENCA_SVMCLASSIFICATION_H
#define FROZENCA_SVMCLASSIFICATION_H

#include "SVMModel.h"
#include "SVMSolver.h"
#include <algorithm>
#include <functional>
#include <ranges>
#include <unordered_set>
#include <unordered_map>

namespace frozenca {

class SVMClassification : public SVMModel {
private:
    std::unordered_map<int, std::unordered_set<std::size_t>> groups_;
    std::vector<char> nonzero_;
    std::vector<int> data_label_;
    std::vector<int> labels_;
public:
    SVMClassification(const std::shared_ptr<Kernel>& kernel) : SVMModel(kernel) {}
    virtual ~SVMClassification() = default;
    virtual void fit(const SVMTrainData& svm_data, const SVMParams& params) final;
    virtual void constructGroups(const SVMTrainData& svm_data) final;
    virtual void constructModel(const SVMTrainData& svm_data,
                        const std::vector<std::pair<std::vector<float>, float>>& decision_functions) final;
    virtual std::pair<Mat<float>, std::vector<float>> computeQs(std::size_t l, const SubSamples& X, const std::vector<char>& y) const final;
    [[nodiscard]] virtual float predict(const std::vector<float>& sample) const final;
protected:
    virtual std::pair<std::vector<float>, float> fitOne(const SVMTrainData& svm_sub_data, const SVMParams& params,
                                                        float Cp, float Cn) = 0;
};

void SVMClassification::fit(const SVMTrainData& svm_data, const SVMParams& params) {
    constructGroups(svm_data);
    const std::size_t l = svm_data.n_;
    const std::size_t k = groups_.size();

    // train k * (k - 1) / 2 binary classifiers
    nonzero_.clear();
    nonzero_.resize(l);

    std::vector<SubSamples> class_X (k);

    for (std::size_t i = 0; i < k; ++i) {
        auto label_i = labels_[i];
        auto& group_i = groups_[label_i];
        for (auto index_i : group_i) {
            class_X[i].push_back(svm_data.getXRef(index_i));
        }
    }

    std::vector<std::pair<std::vector<float>, float>> decision_functions;
    for (std::size_t i = 0; i < k; ++i) {
        auto& group_i = groups_[labels_[i]];
        for (std::size_t j = i + 1; j < k; ++j) {
            auto& group_j = groups_[labels_[j]];
            SubSamples sample_ij;
            std::ranges::copy(class_X[i], std::back_inserter(sample_ij));
            std::ranges::copy(class_X[j], std::back_inserter(sample_ij));
            std::vector<float> label_ij;
            for (std::size_t ci = 0; ci < class_X[i].size(); ++ci) {
                label_ij.push_back(+1);
            }
            for (std::size_t cj = 0; cj < class_X[j].size(); ++cj) {
                label_ij.push_back(-1);
            }
            SVMTrainData sub_data(sample_ij, label_ij);
            auto wi = params.weight_scale.empty() ? 1.0f : params.weight_scale[i];
            auto wj = params.weight_scale.empty() ? 1.0f : params.weight_scale[j];
            auto f = fitOne(sub_data, params, params.C * wi, params.C * wj);
            std::size_t alpha_index = 0;
            for (auto index_i : group_i) {
                if (!nonzero_[index_i] && std::fabs(f.first[alpha_index]) > 0) {
                    nonzero_[index_i] = true;
                }
                alpha_index++;
            }
            for (auto index_j : group_j) {
                if (!nonzero_[index_j] && std::fabs(f.first[alpha_index]) > 0) {
                    nonzero_[index_j] = true;
                }
                alpha_index++;
            }
            decision_functions.push_back(std::move(f));
        }
    }
    constructModel(svm_data, decision_functions);
}

void SVMClassification::constructGroups(const SVMTrainData& svm_data) {
    if (!groups_.empty()) {
        return;
    }
    const std::size_t l = svm_data.n_;
    data_label_.resize(l);
    for (std::size_t i = 0; i < l; ++i) {
        int this_label = static_cast<int>(svm_data.y_[i]);
        groups_[this_label].insert(i);
        data_label_[i] = this_label;
    }
    for (const auto& [label, _] : groups_) {
        labels_.push_back(label);
    }
    if (groups_.size() == 1) {
        std::cerr << "WARNING: training data in only one class\n";
    }
}

void SVMClassification::constructModel(const SVMTrainData& svm_data,
                    const std::vector<std::pair<std::vector<float>, float>>& decision_functions) {
    const std::size_t k = groups_.size();

    SV_.clear();
    SV_coeff_.clear();
    rho_.clear();

    for (const auto& [alpha, rho] : decision_functions) {
        rho_.push_back(rho);
    }
    for (std::size_t i = 0; i < k; ++i) {
        for (auto idx : groups_[labels_[i]]) {
            if (nonzero_[idx]) {
                SV_[i].push_back(svm_data.getX(i));
            }
        }
    }
    std::size_t decision_function_index = 0;
    for (std::size_t i = 0; i < k; ++i) {
        auto& group_i = groups_[labels_[i]];
        for (std::size_t j = i + 1; j < k; ++j) {
            auto& group_j = groups_[labels_[j]];
            auto& alpha = decision_functions[decision_function_index].first;
            std::size_t alpha_index = 0;
            for (auto index_i : group_i) {
                if (nonzero_[index_i]) {
                    SV_coeff_[i * k + j].push_back(alpha[alpha_index]);
                }
                ++alpha_index;
            }
            for (auto index_j : group_j) {
                if (nonzero_[index_j]) {
                    SV_coeff_[i * k + j].push_back(alpha[alpha_index]);
                }
                ++alpha_index;
            }
            ++decision_function_index;
        }
    }
}

std::pair<Mat<float>, std::vector<float>> SVMClassification::computeQs(std::size_t l,
                                                                       const SubSamples& X,
                                                                       const std::vector<char>& y) const {
    Mat<float> Q (l, l);
    std::vector<float> QD (l);
    for (std::size_t i = 0; i < l; ++i) {
        for (std::size_t j = 0; j < l; ++j) {
            Q[{i, j}] = y[i] * y[j] * (*kernel_)(X[i], X[j]);
        }
        QD[i] = Q[{i, i}];
    }
    return {Q, QD};
}

float SVMClassification::predict(const std::vector<float>& sample) const {
    const std::size_t k = groups_.size();
    std::vector<std::size_t> vote(k);

    std::unordered_map<std::size_t, std::vector<float>> k_values;
    for (std::size_t i = 0; i < k; ++i) {
        std::size_t SV_index = 0;
        for (auto idx : groups_.at(labels_[i])) {
            if (nonzero_[idx]) {
                k_values[i].push_back((*kernel_)(sample, SV_.at(i)[SV_index++]));
            }
        }
    }
    std::size_t rho_index = 0;
    for (std::size_t i = 0; i < k; ++i) {
        auto& group_i = groups_.at(labels_[i]);
        for (std::size_t j = i + 1; j < k; ++j) {
            float sum = 0.0f;
            auto& group_j = groups_.at(labels_[j]);
            auto& curr_coeffs = SV_coeff_.at(i * k + j);
            for (std::size_t idx_i = 0; idx_i < group_i.size(); ++idx_i) {
                sum += curr_coeffs[idx_i] * k_values[i][idx_i];
            }
            for (std::size_t idx_j = 0; idx_j < group_j.size(); ++idx_j) {
                sum += curr_coeffs[group_i.size() + idx_j] * k_values[j][idx_j];
            }
            sum -= rho_[rho_index++];
            if (sum > 0.0f) {
                ++vote[i];
            } else {
                ++vote[j];
            }
        }
    }
    auto vote_max_index = std::distance(vote.begin(), std::ranges::max_element(vote));
    return static_cast<float>(vote_max_index);
}

class SVMCSVC final : public SVMClassification {
public:
    SVMCSVC(const std::shared_ptr<Kernel>& kernel) : SVMClassification(kernel) {}

    std::pair<std::vector<float>, float> fitOne(const SVMTrainData& svm_data, const SVMParams& params,
                                                float Cp, float Cn);
};

std::pair<std::vector<float>, float> SVMCSVC::fitOne(const SVMTrainData& svm_data,
                                                     const SVMParams& params,
                                                     float Cp, float Cn) {
    const std::size_t l = svm_data.n_;
    std::vector<float> minus_ones (l, -1.0f);
    std::vector<char> y (l);
    std::vector<float> alpha (l);
    for (std::size_t i = 0; i < l; ++i) {
        if (svm_data.y_[i] > 0) {
            y[i] = +1;
        } else {
            y[i] = -1;
        }
    }

    auto [Q, QD] = computeQs(l, svm_data.getXRef(), y);
    SolverOrdinary s(l, Q, QD, minus_ones, y, alpha, {Cp, Cn}, params.eps, params.shrinking);
    auto solution = s.Solve();

    if (Cp == Cn) {
        float sum_alpha = std::accumulate(solution.alpha.begin(), solution.alpha.end(), 0.0f);
        std::cout << "nu = " << sum_alpha / (Cp * l) << '\n';
    }
    std::ranges::transform(solution.alpha, y, solution.alpha.begin(), std::multiplies{});
    return {solution.alpha, solution.rho};
}

class SVMNuSVC final : public SVMClassification {
public:
    SVMNuSVC(const std::shared_ptr<Kernel>& kernel, const SVMParams& params) : SVMClassification(kernel) {}
    std::pair<std::vector<float>, float> fitOne(const SVMTrainData& svm_data, const SVMParams& params,
                                                float Cp, float Cn);
};


std::pair<std::vector<float>, float> SVMNuSVC::fitOne(const SVMTrainData& svm_data, const SVMParams& params,
                                                     float Cp, float Cn) {
    const std::size_t l = svm_data.n_;

    std::vector<float> alpha (l);
    std::vector<char> y (l);
    auto sum_pos = params.nu * l / 2.0f;
    auto sum_neg = params.nu * l / 2.0f;
    for (std::size_t i = 0; i < l; ++i) {
        if (svm_data.y_[i] > 0) {
            y[i] = +1;
            alpha[i] = std::min(1.0f, sum_pos);
            sum_pos -= alpha[i];
        } else {
            y[i] = -1;
            alpha[i] = std::min(1.0f, sum_neg);
            sum_neg -= alpha[i];
        }
    }

    std::vector<float> zeros(l);

    auto [Q, QD] = computeQs(l, svm_data.getXRef(), y);
    SolverOrdinary s(l, Q, QD, zeros, y, alpha, {1.0f, 1.0f}, params.eps, params.shrinking);
    auto solution = s.Solve();

    auto r = solution.r;
    std::cout << "C = " << 1.0f / r << '\n';

    std::ranges::transform(solution.alpha, y, solution.alpha.begin(), [&r](auto a, auto y) {
        return a * y / r;
    });
    solution.rho /= r;
    solution.obj /= std::pow(r, 2.0f);
    solution.upper_bound = {1.0f / r, 1.0f / r};
    return {solution.alpha, solution.rho};
}

} // namespace frozenca

#endif //FROZENCA_SVMCLASSIFICATION_H
