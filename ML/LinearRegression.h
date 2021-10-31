#ifndef FROZENCA_LINEARREGRESSION_H
#define FROZENCA_LINEARREGRESSION_H

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

class LinearRegression {
private:
    std::vector<float> theta_;
    std::vector<std::vector<std::size_t>> monomials_;

    enum class Penalty {
        None,
        L1,
        L2
    };

public:
    void fit(const std::vector<std::vector<float>>& X, const std::vector<float>& y,
             const std::pair<std::string, float>& penalty = {}, std::size_t degree = 1,
             std::size_t batch_size = 0) {
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
        auto [p_type, alpha] = penalty;
        Penalty penalty_type = Penalty::None;
        std::ranges::transform(p_type, p_type.begin(), [](auto c){ return std::tolower(c);});
        if (p_type == "") {
            penalty_type = Penalty::None;
            alpha = 0.0f;
        } else if (p_type == "l1" || p_type == "lasso") {
            penalty_type = Penalty::L1;
            if (!batch_size) {
                batch_size = 1; // L1 regression only can be fit via gradient descent
            }
        } else if (p_type == "l2" || p_type == "ridge") {
            penalty_type = Penalty::L2;
        } else {
            throw std::invalid_argument("Unknown penalty type");
        }
        if (alpha < 0.0f) {
            throw std::invalid_argument("Alpha value must be non-negative");
        }
        if (degree == 0) {
            throw std::invalid_argument("Degree must be nonzero");
        }
        const std::size_t num_samples = X.size();
        if (batch_size && num_samples % batch_size) {
            throw std::invalid_argument("Batch size must divide number of samples");
        }

        std::size_t num_features = 1;
        for (std::size_t k = 1; k <= degree; ++k) {
            num_features *= (n + k);
            num_features /= k;
        }

        monomials_ = get_monomials(n, degree);

        Mat<float> M(num_samples, num_features);
        for (std::size_t i = 0; i < num_samples; ++i) {
            auto m_i = M.row(i);
            construct_features(m_i, X[i], monomials_);
        }

        Vec<float> Y(num_samples);
        for (std::size_t i = 0; i < num_samples; ++i) {
            Y[i] = y[i];
        }

        fit_theta(M, Y, penalty_type, alpha, batch_size);
    }

    [[nodiscard]] std::vector<float> getTheta() const {
        return theta_;
    }

    [[nodiscard]] float predict(const std::vector<float>& x) const {
        if (theta_.empty()) {
            throw std::runtime_error("The model has not been fit yet!");
        }
        auto res = 0.0f;
        for (std::size_t i = 0; i < theta_.size(); ++i) {
            res += theta_[i] * eval_monomial(x, monomials_[i]);
        }
        return res;
    }

private:

    void fit_theta(const Mat<float>& X, const Vec<float>& Y,
                   const Penalty& penalty_type, float alpha, std::size_t batch_size) {
        const std::size_t num_samples = X.dims(0);
        const std::size_t num_features = X.dims(1);
        Vec<float> Theta (num_features);
        if (!batch_size) {
            Mat<float> l2_pen = identity<float>(num_features);
            l2_pen *= alpha;
            Theta = dot(dot(inv(dot(transpose(X), X) + l2_pen), transpose(X)), Y);
        } else {
            // gradient descent

            // random initialization
            std::mt19937 gen(std::random_device{}());
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (std::size_t i = 0; i < num_features; ++i) {
                Theta[i] = dist(gen);
            }

            constexpr std::size_t n_epochs = 100;
            constexpr float tolerance = 1e-7;
            std::size_t num_batches = num_samples / batch_size;
            assert(!(num_samples % batch_size));
            std::vector<std::size_t> batch_indices (num_batches);
            std::iota(batch_indices.begin(), batch_indices.end(), 0lu);
            std::ranges::shuffle(batch_indices, gen);

            bool converged = false;
            for (std::size_t i = 0; i < n_epochs; ++i) {
                if (converged) {
                    break;
                }
                for (std::size_t b = 0; b < num_batches; ++b) {
                    auto xi = X.submatrix({batch_indices[b] * batch_size, 0}, {(batch_indices[b] + 1) * batch_size, num_features});
                    auto yi = Y.submatrix(batch_indices[b] * batch_size, (batch_indices[b] + 1) * batch_size);
                    auto gradient = dot(transpose(xi), dot(xi, Theta) - yi);
                    float lr = 0.1f * n_epochs * num_batches / (n_epochs + i) / (num_batches + b);
                    gradient *= 2.0f * lr / batch_size;
                    if (penalty_type == Penalty::None) {
                        Theta = Theta - gradient;
                    } else if (penalty_type == Penalty::L2) {
                        auto pen = Theta;
                        pen *= lr * 2.0f * alpha;
                        Theta = Theta - gradient - pen;
                    } else if (penalty_type == Penalty::L1) {
                        Vec<float> pen (num_features);
                        for (std::size_t f = 0; f < num_features; ++f) {
                            if (Theta[f] > 0.0f) {
                                pen[f] = 1.0f;
                            } else if (Theta[f] < 0.0f) {
                                pen[f] = -1.0f;
                            }
                        }
                        pen *= lr * alpha;
                        Theta = Theta - gradient - pen;
                    }
                    if (norm(gradient) < tolerance) {
                        converged = true;
                        break;
                    }
                }
            }
        }
        theta_.clear();
        for (std::size_t i = 0; i < num_features; ++i) {
            theta_.push_back(Theta[i]);
        }
    }

    // for n = 2 and degree 2,
    // 1, x, y, x^2, xy, y^2
    // {(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)}
    static std::vector<std::vector<std::size_t>> get_monomials(std::size_t n, std::size_t deg) {
        std::vector<std::vector<std::size_t>> d_monomials;
        d_monomials.push_back({0});
        construct_monomials(d_monomials, n, deg);
        std::ranges::sort(d_monomials);
        return d_monomials;
    }

    // recursive construct.
    // 0 -> 0
    //   -> 1
    //   -> 2
    // 1 -> 0
    //   -> 1
    // 2 -> 0
    static void construct_monomials(std::vector<std::vector<std::size_t>>& d_monomials, std::size_t n,
                               std::size_t deg) {
        if (!n) {
            return;
        }
        std::vector<std::vector<std::size_t>> new_monomials;
        while (!d_monomials.empty()) {
            auto back = d_monomials.back();
            std::size_t s = back[0];
            d_monomials.pop_back();
            for (std::size_t d = 0; d <= deg - s && d <= deg; ++d) {
                new_monomials.push_back(back);
                new_monomials.back().push_back(d);
                new_monomials.back()[0] += d;
            }
        }
        d_monomials = new_monomials;
        construct_monomials(d_monomials, n - 1, deg);
    }

    static float eval_monomial(const std::vector<float>& sample, const std::vector<std::size_t>& monomial) {
        return std::inner_product(sample.begin(), sample.end(), monomial.begin() + 1, 0.0f,
                                  std::plus<>(), [](auto s, auto exp) {
                    return std::pow(s, 1.0f * exp);
        });
    }

    static void construct_features(VecView<float>& row, const std::vector<float>& sample,
                            const std::vector<std::vector<std::size_t>>& monomials) {
        const std::size_t num_features = row.dims(0);
        const std::size_t n = sample.size();
        for (std::size_t i = 0; i < num_features; ++i) {
            row[i] = eval_monomial(sample, monomials[i]);
        }
    }

};

} // namespace frozenca

#endif //FROZENCA_LINEARREGRESSION_H
