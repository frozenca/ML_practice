#ifndef FROZENCA_LOGISTICREGRESSION_H
#define FROZENCA_LOGISTICREGRESSION_H

#include "../Matrix/Matrix.h"
#include <algorithm>
#include <concepts>
#include <cctype>
#include <functional>
#include <stdexcept>
#include <numeric>
#include <random>
#include <ranges>
#include <string>
#include <vector>

namespace frozenca {

class LogisticRegression {
private:
    Mat<float> theta_;
    std::size_t num_classes_ = 0;

    enum class Penalty {
        None,
        L1,
        L2
    };

public:
    LogisticRegression() : theta_(1, 1) {}

    void fit(const std::vector<std::vector<float>>& X, const std::vector<std::size_t>& y,
             std::size_t num_classes = 2, const std::pair<std::string, float>& penalty = {},
             std::size_t batch_size = 1) {
        if (X.empty() || y.empty()) {
            throw std::invalid_argument("Data set is empty");
        }
        if (X.size() != y.size()) {
            throw std::invalid_argument("Data set sizes do not match");
        }
        const std::size_t n = X[0].size();
        if (std::ranges::any_of(X, [&n](const auto& row) { return row.size() != n;})) {
            throw std::invalid_argument("Feature set sizes are not equal");
        }
        if (num_classes < 2) {
            throw std::invalid_argument("Number of classes should be at least 2");
        }
        num_classes_ = num_classes;
        auto [p_type, alpha] = penalty;
        Penalty penalty_type = Penalty::None;
        std::ranges::transform(p_type, p_type.begin(), [](auto c){ return std::tolower(c);});
        if (p_type == "") {
            penalty_type = Penalty::None;
            alpha = 0.0f;
        } else if (p_type == "l1" || p_type == "lasso") {
            penalty_type = Penalty::L1;
        } else if (p_type == "l2" || p_type == "ridge") {
            penalty_type = Penalty::L2;
        } else {
            throw std::invalid_argument("Unknown penalty type");
        }
        if (alpha < 0.0f) {
            throw std::invalid_argument("Alpha value must be non-negative");
        }
        const std::size_t num_samples = X.size();
        if (!batch_size || num_samples % batch_size) {
            throw std::invalid_argument("Batch size must divide number of samples");
        }

        std::size_t num_features = n + 1;

        Mat<float> M(num_samples, num_features);
        for (std::size_t i = 0; i < num_samples; ++i) {
            auto Mi = M.row(i);
            Mi[0] = 1.0f;
            for (std::size_t j = 0; j < n; ++j) {
                Mi[j + 1] = X[i][j];
            }
        }

        fit_theta(M, y, penalty_type, alpha, batch_size);
    }

    [[nodiscard]] std::size_t predict(const std::vector<float>& x) const {
        if (num_classes_ == 0) {
            throw std::runtime_error("The model has not been fit yet!");
        }
        Vec<float> X (x.size() + 1);
        X[0] = 1.0f;
        for (std::size_t i = 0; i < x.size(); ++i) {
            X[i + 1] = x[i];
        }
        std::vector<float> logits = compute_softmax(theta_, X);
        return static_cast<std::size_t>(std::distance(logits.begin(), std::ranges::max_element(logits)));
    }

private:

    static std::vector<float> compute_softmax(const Mat<float>& theta,
                                             const VecView<float>& xi) {
        const std::size_t K = theta.dims(0);
        std::vector<float> logits (K);
        for (std::size_t k = 0; k < K; ++k) {
            auto theta_k = theta.row(k);
            logits[k] = std::inner_product(theta_k.begin(), theta_k.end(), xi.begin(), 0.0f);
        }
        auto max_logit = *std::ranges::max_element(logits);
        std::ranges::transform(logits, logits.begin(), [&max_logit](auto lg) {
            return std::exp(lg - max_logit);
        });
        auto score_sum = std::accumulate(logits.begin(), logits.end(), 0.0f);
        std::ranges::transform(logits, logits.begin(), [&score_sum](auto lg) {
            return lg / score_sum;
        });
        return logits;
    }

    void fit_theta(const Mat<float>& X, const std::vector<std::size_t>& Y,
                   const Penalty& penalty_type, float alpha, std::size_t batch_size) {
        assert(num_classes_ >= 2);
        const std::size_t num_samples = X.dims(0);
        const std::size_t num_features = X.dims(1);

        theta_ = Mat<float>(num_classes_, num_features);

        // gradient descent

        // random initialization
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (std::size_t k = 0; k < num_classes_; ++k) {
            for (std::size_t i = 0; i < num_features; ++i) {
                theta_[{k, i}] = 0.0f;
            }
        }

        constexpr std::size_t n_epochs = 100;
        constexpr float tolerance = 1e-7;
        std::size_t num_batches = num_samples / batch_size;
        assert(!(num_samples % batch_size));
        std::vector<std::size_t> batch_indices(num_batches);
        std::iota(batch_indices.begin(), batch_indices.end(), 0lu);
        std::ranges::shuffle(batch_indices, gen);

        bool converged = false;
        for (std::size_t i = 0; i < n_epochs; ++i) {
            if (converged) {
                break;
            }
            for (std::size_t b = 0; b < num_batches; ++b) {
                float lr = 0.0001f * n_epochs * num_batches / (n_epochs + i) / (num_batches + b);

                Mat<float> gradient = zeros_like(theta_);
                for (std::size_t r = batch_indices[b] * batch_size; r < (batch_indices[b] + 1) * batch_size; ++r) {
                    auto xi = X.row(r);
                    auto yi = Y[r];
                    auto softmax_score = compute_softmax(theta_, xi);
                    for (std::size_t k = 0; k < num_classes_; ++k) {
                        auto grad_k = gradient.row(k);
                        Vec<float> xik = xi;
                        xik *= (softmax_score[k] - ((yi == k) ? 1.0f : 0.0f));
                        grad_k += xik;
                    }
                }
                gradient *= lr / batch_size;
                if (penalty_type == Penalty::None) {
                    theta_ = theta_ - gradient;
                } else if (penalty_type == Penalty::L2) {
                    auto pen = theta_;
                    pen *= lr * 2.0f * alpha;
                    theta_ = theta_ - gradient - pen;
                } else if (penalty_type == Penalty::L1) {
                    Mat<float> pen = zeros_like(theta_);
                    for (std::size_t k = 0; k < num_classes_; ++k) {
                        for (std::size_t f = 0; f < num_features; ++f) {
                            if (theta_[{k, f}] > 0.0f) {
                                pen[{k, f}] = 1.0f;
                            } else if (theta_[{k, f}] < 0.0f) {
                                pen[{k, f}] = -1.0f;
                            }
                        }
                    }
                    pen *= lr * alpha;
                    theta_ = theta_ - gradient - pen;
                }
                if (norm(gradient) < tolerance) {
                    converged = true;
                    break;
                }
            }
        }
    }

};

} // namespace frozenca


#endif //FROZENCA_LOGISTICREGRESSION_H
