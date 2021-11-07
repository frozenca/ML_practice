#ifndef FROZENCA_SVMREGRESSION_H
#define FROZENCA_SVMREGRESSION_H

#include "SVMModel.h"
#include "SVMSolver.h"

namespace frozenca {

class SVMTwoClass : public SVMModel {
public:
    SVMTwoClass(std::unique_ptr<Kernel> kernel) : SVMModel(std::move(kernel)) {}
    virtual ~SVMTwoClass() = default;
    virtual void constructModel(const SVMTrainData& svm_data, const Solution& solution) final;
    [[nodiscard]] virtual float predict(const std::vector<float>& sample) const;
protected:
    virtual std::pair<Mat<float>, std::vector<float>> computeQs(std::size_t l, const std::vector<std::vector<float>>& X) const = 0;
};

void SVMTwoClass::constructModel(const SVMTrainData& svm_data, const Solution& solution) {
    SV_.clear();
    SV_coeff_.clear();
    rho_.clear();

    std::size_t count_SV = 0;
    std::size_t count_BSV = 0;

    // construct support vectors

    const std::size_t l = solution.alpha.size();
    auto& alpha = solution.alpha;
    for (std::size_t i = 0; i < l; ++i) {
        if (std::fabs(alpha[i]) > 0) {
            ++count_SV;
            if (svm_data.y_[i] > 0) {
                if (std::fabs(alpha[i]) >= solution.upper_bound.first) {
                    ++count_BSV;
                }
            } else {
                if (std::fabs(alpha[i]) >= solution.upper_bound.second) {
                    ++count_BSV;
                }
            }

            SV_[0].push_back(svm_data.getX(i));
            SV_coeff_[0].push_back(alpha[i]);
        }
    }

    std::cout << "nSV = " << count_SV << ", nBSV = " << count_BSV << '\n';

    rho_.push_back(solution.rho);
}

[[nodiscard]] float SVMTwoClass::predict(const std::vector<float>& sample) const {
    auto& coeff = SV_coeff_.at(0);
    float sum = 0.0f;
    std::size_t svs = coeff.size();
    for (std::size_t i = 0; i < svs; ++i) {
        sum += coeff[i] * (*kernel_)(sample, SV_.at(0)[i]);
    }
    sum -= rho_[0];
    return sum;
}

class SVMOneClass final : public SVMTwoClass {
public:
    SVMOneClass(std::unique_ptr<Kernel> kernel) : SVMTwoClass(std::move(kernel)) {}
private:
    std::pair<Mat<float>, std::vector<float>> computeQs(std::size_t l, const std::vector<std::vector<float>>& X) const;
public:
    void fit(const SVMTrainData& svm_data, const SVMParams& params);
    [[nodiscard]] float predict(const std::vector<float>& sample) const;
};


std::pair<Mat<float>, std::vector<float>> SVMOneClass::computeQs(std::size_t l,
                                                                   const std::vector<std::vector<float>>& X) const {
    Mat<float> Q (l, l);
    std::vector<float> QD (l);
    for (std::size_t i = 0; i < l; ++i) {
        for (std::size_t j = 0; j < l; ++j) {
            Q[{i, j}] = (*kernel_)(X[i], X[j]);
        }
        QD[i] = Q[{i, i}];
    }
    return {Q, QD};
}

void SVMOneClass::fit(const SVMTrainData& svm_data, const SVMParams& params) {
    SVMModel::fit(svm_data, params);
    const std::size_t l = svm_data.n_;
    std::vector<float> alpha (l);
    int n = static_cast<int>(params.nu * l); // # of alpha's at upper bound
    for (std::size_t i = 0; i < std::min(static_cast<std::size_t>(n), l); ++i) {
        alpha[i] = 1.0f;
    }
    if (n < l) {
        alpha[n] = static_cast<float>(params.nu * l - n);
    }
    for (std::size_t i = n + 1; i < l; ++i) {
        alpha[i] = 0.0f;
    }

    std::vector<float> zeros(l);
    std::vector<char> ones(l, 1);
    auto [Q, QD] = computeQs(l, svm_data.getX());

    SolverOrdinary s(l, Q, QD, zeros, ones, alpha, {1.0, 1.0}, params.eps, params.shrinking);
    auto solution = s.Solve();

    constructModel(svm_data, solution);
}

[[nodiscard]] float SVMOneClass::predict(const std::vector<float>& sample) const {
    auto res = SVMTwoClass::predict(sample);
    return (res > 0) ? +1 : -1;
}

class SVMRegression : public SVMTwoClass {
public:
    virtual ~SVMRegression() = default;
protected:
    SVMRegression(std::unique_ptr<Kernel> kernel) : SVMTwoClass(std::move(kernel)) {}
    std::pair<Mat<float>, std::vector<float>> computeQs(std::size_t l, const std::vector<std::vector<float>>& X) const final;
};

std::pair<Mat<float>, std::vector<float>> SVMRegression::computeQs(std::size_t l,
                                                                   const std::vector<std::vector<float>>& X) const {
    Mat<float> Q (l, l);
    std::vector<float> QD (2 * l);
    for (std::size_t i = 0; i < l; ++i) {
        for (std::size_t j = 0; j < l; ++j) {
            Q[{i, j}] = (*kernel_)(X[i], X[j]);
        }
        QD[i] = Q[{i, i}];
        QD[i + l] = Q[{i, i}];
    }
    return {Q, QD};
}

class SVMEpsSVR final : public SVMRegression {
public:
    SVMEpsSVR(std::unique_ptr<Kernel> kernel) : SVMRegression(std::move(kernel)) {}
    void fit(const SVMTrainData& svm_data, const SVMParams& params);
};

void SVMEpsSVR::fit(const SVMTrainData& svm_data, const SVMParams& params) {
    SVMModel::fit(svm_data, params);
    const std::size_t l = svm_data.n_;
    const std::size_t two_l = 2 * l;
    std::vector<float> alpha2 (two_l);
    std::vector<float> linear_term (two_l);
    std::vector<char> y (two_l);
    for (std::size_t i = 0; i < l; ++i) {
        linear_term[i] = params.p - svm_data.y_[i];
        y[i] = +1;
        linear_term[i + l] = params.p + svm_data.y_[i];
        y[i + l] = -1;
    }

    auto [Q, QD] = computeQs(l, svm_data.getX());
    SolverOrdinary s(two_l, Q, QD, linear_term, y, alpha2, {params.C, params.C}, params.eps, params.shrinking);
    auto solution = s.Solve();
    std::vector<float> alpha (l);
    float sum_alpha = 0.0f;
    for (std::size_t i = 0; i < l; ++i) {
        alpha[i] = solution.alpha[i] - solution.alpha[i + 1];
        sum_alpha += std::fabs(alpha[i]);
    }
    solution.alpha = alpha;
    std::cout << "nu = " << sum_alpha / (params.C * l) << '\n';
    constructModel(svm_data, solution);
}

class SVMNuSVR final : public SVMRegression {
public:
    SVMNuSVR(std::unique_ptr<Kernel> kernel) : SVMRegression(std::move(kernel)) {}
    void fit(const SVMTrainData& svm_data, const SVMParams& params);
};

void SVMNuSVR::fit(const SVMTrainData& svm_data, const SVMParams& params) {
    SVMModel::fit(svm_data, params);
    const std::size_t l = svm_data.n_;
    const std::size_t two_l = 2 * l;
    std::vector<float> alpha2 (two_l);
    std::vector<float> linear_term (two_l);
    std::vector<char> y (two_l);
    float sum = params.C * params.nu * l / 2.0f;
    for (std::size_t i = 0; i < l; ++i) {
        alpha2[i] = alpha2[i + l] = std::min(sum, params.C);
        sum -= alpha2[i];
        linear_term[i] = -svm_data.y_[i];
        y[i] = +1;
        linear_term[i + l] = svm_data.y_[i];
        y[i + l] = -1;
    }

    auto [Q, QD] = computeQs(l, svm_data.getX());

    SolverNU s(two_l, Q, QD, linear_term, y, alpha2, {params.C, params.C}, params.eps, params.shrinking);
    auto solution = s.Solve();
    std::vector<float> alpha (l);
    for (std::size_t i = 0; i < l; ++i) {
        alpha[i] = solution.alpha[i] - solution.alpha[i + 1];
    }
    solution.alpha = alpha;
    std::cout << "epsilon = " << -solution.r << '\n';
    constructModel(svm_data, solution);
}


} // namespace frozenca

#endif //FROZENCA_SVMREGRESSION_H
