#ifndef FROZENCA_SVMSOLVER_H
#define FROZENCA_SVMSOLVER_H

#include "../../Matrix/Matrix.h"
#include <cstddef>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace frozenca {

static constexpr float TAU = 1e-12;

struct Solution {
    std::vector<float> alpha;
    float obj = 0.0f;
    float rho = 0.0f;
    float r = 0.0f; // for Solver_NU
    std::pair<float, float> upper_bound = {0.0f, 0.0f};
};

class Solver {
public:
    enum class Alpha {
        Lower,
        Upper,
        Free
    };

    Solver(std::size_t l, Mat<float>& Q, std::vector<float> QD, std::vector<float> p, std::vector<char> y,
           std::vector<float> alpha, std::pair<float, float> C, float eps, bool shrinking);

    virtual ~Solver() = default;

protected:
    const std::size_t l_;
    Mat<float>& Q_;
    std::vector<float> QD_;
    std::vector<float> p_;
    std::vector<char> y_;
    std::vector<float> alpha_;
    std::pair<float, float> C_;
    float eps_;
    std::vector<Alpha> alpha_status_;
    std::vector<float> G_; // gradient of objective function
    std::vector<float> G_bar_; // gradient, if we treat free variable as 0
    bool shrinking_;
    bool unshrink_ = false;
    std::unordered_set<std::size_t> active_set_;

public:
    [[nodiscard]] float getC(std::size_t i) const {
        return (y_[i] > 0) ? C_.first : C_.second;
    }

    [[nodiscard]] std::vector<float> getQ(std::size_t i) const {
        auto Q_i = Q_.col(i);
        std::vector<float> Qi;
        for (auto q : Q_i) {
            Qi.push_back(q);
        }
        return Qi;
    }

    void updateAlphaStatus(std::size_t i) {
        if (alpha_[i] >= getC(i)) {
            alpha_status_[i] = Alpha::Upper;
        } else if (alpha_[i] <= 0) {
            alpha_status_[i] = Alpha::Lower;
        } else {
            alpha_status_[i] = Alpha::Free;
        }
    }

public:
    virtual Solution Solve() final;
    virtual void updateAlphaGradientValues(std::size_t i, std::size_t j) final;
    virtual void reconstructGradient() final;

protected:
    virtual std::tuple<bool, std::size_t, std::size_t> selectWorkingSet() const = 0;
    virtual void calculateRho(Solution& sol) const = 0;
    virtual void doShrinking() = 0;

};

Solver::Solver(std::size_t l, Mat<float>& Q, std::vector<float> QD,
               std::vector<float> p, std::vector<char> y,
               std::vector<float> alpha, std::pair<float, float> C, float eps, bool shrinking) :
        l_{l}, Q_(Q), QD_(std::move(QD)), p_(std::move(p)), y_(std::move(y)),
        alpha_(std::move(alpha)), C_(std::move(C)), eps_{eps}, shrinking_{shrinking} {
    alpha_status_.resize(l_);
    for (std::size_t i = 0; i < l_; ++i) {
        active_set_.insert(i);
        updateAlphaStatus(i);
    }
    G_ = p_;
    G_bar_.resize(l_);
    for (std::size_t i = 0; i < l_; ++i) {
        if (alpha_status_[i] != Alpha::Lower) {
            auto Q_i = getQ(i);
            for (std::size_t j = 0; j < l_; ++j) {
                G_[j] += alpha_[i] * Q_i[j];
            }
            if (alpha_status_[i] == Alpha::Upper) {
                for (std::size_t j = 0; j < l_; ++j) {
                    G_bar_[j] += getC(i) * Q_i[j];
                }
            }
        }
    }
}

// Solver ordinary (non NU)

class SolverOrdinary final : public Solver {
public:
    SolverOrdinary(std::size_t l, Mat<float>& Q, std::vector<float> QD, std::vector<float> p, std::vector<char> y,
                   std::vector<float> alpha, std::pair<float, float> C, float eps, bool shrinking)
            : Solver(l, Q, QD, p, y, alpha, C, eps, shrinking) {}

    std::tuple<bool, std::size_t, std::size_t> selectWorkingSet() const;
    void calculateRho(Solution& sol) const;
    void doShrinking() final;
    bool beShrunk(std::size_t i, float g_max1, float g_max2) const;
};

Solution Solver::Solve() {
    std::size_t iter = 0;
    std::size_t max_iter = std::max(1'000'000lu, l_ > std::numeric_limits<std::size_t>::max() / 100 ?
                                                 std::numeric_limits<std::size_t>::max() : 100 * l_);
    std::size_t counter = std::min(l_, 1'000lu) + 1;

    while (iter < max_iter) {
        if (!--counter) {
            counter = std::min(l_, 1'000lu);
            if (shrinking_) {
                doShrinking();
            }
        }
        bool already_opt = false;
        std::size_t i = 0, j = 0;
        std::tie(already_opt, i, j) = selectWorkingSet();
        if (already_opt) {
            // reconstruct the whole gradient
            reconstructGradient();
            std::tie(already_opt, i, j) = selectWorkingSet();
            if (already_opt) {
                break;
            } else {
                counter = 1; // do shrinking in next iteration
            }
        }
        ++iter;

        updateAlphaGradientValues(i, j);
    }

    if (iter >= max_iter) {
        if (active_set_.size() < l_) {
            reconstructGradient();
        }
        std::cerr << "WARNING: reaching max number of iterations\n";
    }

    Solution s;
    calculateRho(s);
    for (std::size_t i = 0; i < l_; ++i) {
        s.obj += alpha_[i] * (G_[i] + p_[i]);
    }
    s.obj /= 2.0f;
    s.alpha = alpha_;
    s.upper_bound = C_;

    std::cout << "Optimization finished, #iter = " << iter << '\n';
    return s;
}

void Solver::updateAlphaGradientValues(std::size_t i, std::size_t j) {
    assert(active_set_.contains(i) && active_set_.contains(j));
    // update alpha
    auto Q_i = getQ(i);
    auto Q_j = getQ(j);
    auto C_i = getC(i);
    auto C_j = getC(j);
    auto old_alpha_i = alpha_[i];
    auto old_alpha_j = alpha_[j];

    if (y_[i] != y_[j]) {
        float quad_coeff = QD_[i] + QD_[j] + 2.0f * Q_i[j];
        if (quad_coeff <= 0) {
            quad_coeff = TAU;
        }
        float delta = (-G_[i] - G_[j]) / quad_coeff;
        float diff = alpha_[i] - alpha_[j];
        alpha_[i] += delta;
        alpha_[j] += delta;
        if (diff > 0) {
            if (alpha_[j] > 0) {
                alpha_[j] = 0;
                alpha_[i] = diff;
            }
        } else {
            if (alpha_[i] < 0) {
                alpha_[i] = 0;
                alpha_[j] = -diff;
            }
        }
        if (diff > C_i - C_j)  {
            if (alpha_[i] > C_i) {
                alpha_[i] = C_i;
                alpha_[j] = C_i - diff;
            }
        } else {
            if (alpha_[j] > C_j) {
                alpha_[j] = C_j;
                alpha_[i] = C_j + diff;
            }
        }
    } else {
        float quad_coeff = QD_[i] + QD_[j] - 2.0f * Q_i[j];
        if (quad_coeff <= 0) {
            quad_coeff = TAU;
        }
        float delta = (G_[i] - G_[j]) / quad_coeff;
        float sum = alpha_[i] + alpha_[j];
        alpha_[i] -= delta;
        alpha_[j] += delta;
        if (sum > C_i) {
            if (alpha_[i] > C_i) {
                alpha_[i] = C_i;
                alpha_[j] = sum - C_i;
            }
        } else {
            if (alpha_[j] < 0) {
                alpha_[j] = 0;
                alpha_[i] = sum;
            }
        }
        if (sum > C_j)  {
            if (alpha_[i] > C_j) {
                alpha_[i] = C_j;
                alpha_[j] = sum - C_j;
            }
        } else {
            if (alpha_[i] < 0) {
                alpha_[i] = 0;
                alpha_[j] = sum;
            }
        }
    }

    // update gradient
    float delta_alpha_i = alpha_[i] - old_alpha_i;
    float delta_alpha_j = alpha_[j] - old_alpha_j;

    for (auto k : active_set_) {
        G_[k] += Q_i[k] * delta_alpha_i + Q_j[k] * delta_alpha_j;
    }

    // update alpha status and G_bar
    bool ui = alpha_status_[i] == Alpha::Upper;
    bool uj = alpha_status_[j] == Alpha::Upper;
    updateAlphaStatus(i);
    updateAlphaStatus(j);
    if (ui != (alpha_status_[i] == Alpha::Upper)) {
        Q_i = getQ(i);
        if (ui) {
            for (std::size_t k = 0; k < l_; ++k) {
                G_bar_[k] -= C_i * Q_i[k];
            }
        } else {
            for (std::size_t k = 0; k < l_; ++k) {
                G_bar_[k] += C_i * Q_i[k];
            }
        }
    }
    if (uj != (alpha_status_[j] == Alpha::Upper)) {
        Q_j = getQ(j);
        if (uj) {
            for (std::size_t k = 0; k < l_; ++k) {
                G_bar_[k] -= C_j * Q_j[k];
            }
        } else {
            for (std::size_t k = 0; k < l_; ++k) {
                G_bar_[k] += C_j * Q_j[k];
            }
        }
    }
}

void Solver::reconstructGradient() {
    if (active_set_.size() == l_) {
        return;
    }
    std::unordered_set<std::size_t> inactive_set;
    for (std::size_t j = 0; j < l_; ++j) {
        if (!active_set_.contains(j)) {
            inactive_set.insert(j);
        }
    }

    std::size_t free_count = 0;

    for (std::size_t j = 0; j < l_; ++j) {
        if (active_set_.contains(j)) {
            if (alpha_status_[j] == Alpha::Free) {
                ++free_count;
            }
        } else {
            G_[j] = G_bar_[j] + p_[j];
        }
    }

    if (2 * free_count < active_set_.size()) {
        std::cerr << "WARNING: deactivating shrinking may be faster\n";
    }

    if (free_count * l_ > 2 * active_set_.size() * (l_ - active_set_.size())) {
        for (auto i : inactive_set) {
            auto Q_i = getQ(i);
            for (auto j : active_set_) {
                if (alpha_status_[j] == Alpha::Free) {
                    G_[i] += alpha_[j] * Q_i[j];
                }
            }
        }
    } else {
        for (auto i : active_set_) {
            if (alpha_status_[i] == Alpha::Free) {
                auto Q_i = getQ(i);
                for (auto j : inactive_set) {
                    G_[j] += alpha_[i] * Q_i[j];
                }
            }
        }
    }
    for (std::size_t i = 0; i < l_; ++i) {
        active_set_.insert(i);
    }
}

// [0] : return whether if already optimal
// [1], [2] : return i, j such that
// i : maximizes -y_i * grad(f)_i in I_high(alpha)
// j : minimizes the decreases of obj value
// (if quadratic coefficient <= 0, replace it with tau)
// -y_j * grad(f)_j < -y_i * grad(f)_i, j in I_low(alpha)
std::tuple<bool, std::size_t, std::size_t> SolverOrdinary::selectWorkingSet() const {
    float g_max1 = std::numeric_limits<float>::lowest();
    std::size_t g_max_idx = -1;
    for (auto i : active_set_) {
        if (y_[i] == +1) {
            if (alpha_status_[i] != Alpha::Upper) {
                if (g_max1 <= -G_[i]) {
                    g_max1 = -G_[i];
                    g_max_idx = i;
                }
            }
        } else {
            if (alpha_status_[i] != Alpha::Lower) {
                if (g_max1 <= G_[i]) {
                    g_max1 = G_[i];
                    g_max_idx = i;
                }
            }
        }
    }

    std::size_t i = g_max_idx;
    std::vector<float> Qi;
    if (i != -1) {
        Qi = getQ(i);
    }
    std::size_t g_min_idx = -1;
    float g_max2 = std::numeric_limits<float>::lowest();
    float obj_diff_min = std::numeric_limits<float>::max();
    for (auto j : active_set_) {
        if (y_[j] == +1) {
            if (alpha_status_[j] != Alpha::Lower) {
                float grad_diff = g_max1 + G_[j];
                if (G_[j] >= g_max2) {
                    g_max2 = G_[j];
                }
                if (grad_diff > 0) {
                    float obj_diff = 0.0f;
                    float quad_coeff = QD_[i] + QD_[j] - 2.0f * y_[i] * Qi[j];
                    if (quad_coeff > 0) {
                        obj_diff = -std::pow(grad_diff, 2.0f) / quad_coeff;
                    } else {
                        obj_diff = -std::pow(grad_diff, 2.0f) / TAU;
                    }
                    if (obj_diff <= obj_diff_min) {
                        g_min_idx = j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        } else {
            if (alpha_status_[j] != Alpha::Upper) {
                float grad_diff = g_max1 - G_[j];
                if (-G_[j] >= g_max2) {
                    g_max2 = -G_[j];
                }
                if (grad_diff > 0) {
                    float obj_diff = 0.0f;
                    float quad_coeff = QD_[i] + QD_[j] + 2.0f * y_[i] * Qi[j];
                    if (quad_coeff > 0) {
                        obj_diff = -std::pow(grad_diff, 2.0f) / quad_coeff;
                    } else {
                        obj_diff = -std::pow(grad_diff, 2.0f) / TAU;
                    }
                    if (obj_diff <= obj_diff_min) {
                        g_min_idx = j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
    }
    if (g_max1 + g_max2 < eps_ || g_min_idx == -1) {
        return {true, -1, -1};
    }
    return {false, g_max_idx, g_min_idx};
}

void SolverOrdinary::calculateRho(Solution& sol) const {
    float res = 0.0f;
    std::size_t count_free = 0;
    float ub = std::numeric_limits<float>::max();
    float lb = std::numeric_limits<float>::lowest();
    float sum_free = 0.0f;
    for (auto i : active_set_) {
        auto yG = y_[i] * G_[i];
        if (alpha_status_[i] == Alpha::Upper) {
            if (y_[i] == -1) {
                ub = std::min(ub, yG);
            } else {
                lb = std::max(lb, yG);
            }
        } else if (alpha_status_[i] == Alpha::Lower) {
            if (y_[i] == +1) {
                ub = std::min(ub, yG);
            } else {
                lb = std::max(lb, yG);
            }
        } else {
            ++count_free;
            sum_free += yG;
        }
    }
    if (count_free) {
        res = sum_free / count_free;
    } else {
        res = (ub + lb) / 2.0f;
    }
    sol.rho = res;
}

void SolverOrdinary::doShrinking() {
    float g_max1 = std::numeric_limits<float>::lowest();
    float g_max2 = std::numeric_limits<float>::lowest();

    for (auto i : active_set_) {
        if (y_[i] == +1) {
            if (alpha_status_[i] != Alpha::Upper) {
                g_max1 = std::max(g_max1, -G_[i]);
            }
            if (alpha_status_[i] != Alpha::Lower) {
                g_max2 = std::max(g_max2, G_[i]);
            }
        } else {
            if (alpha_status_[i] != Alpha::Upper) {
                g_max2 = std::max(g_max2, -G_[i]);
            }
            if (alpha_status_[i] != Alpha::Lower) {
                g_max1 = std::max(g_max1, G_[i]);
            }
        }
    }
    if (!unshrink_ && g_max1 + g_max2 <= eps_ * 10) {
        unshrink_ = true;
        reconstructGradient();
    }

    std::unordered_set<std::size_t> to_shrunk;
    for (auto i : active_set_) {
        if (beShrunk(i, g_max1, g_max2)) {
            to_shrunk.insert(i);
        }
    }
    for (auto i : to_shrunk) {
        active_set_.erase(i);
    }
}

bool SolverOrdinary::beShrunk(std::size_t i, float g_max1, float g_max2) const {
    assert(i < l_);
    if (alpha_status_[i] == Alpha::Upper) {
        if (y_[i] == +1) {
            return (-G_[i] > g_max1);
        } else {
            return (-G_[i] > g_max2);
        }
    } else if (alpha_status_[i] == Alpha::Lower) {
        if (y_[i] == +1) {
            return (G_[i] > g_max2);
        } else {
            return (G_[i] > g_max1);
        }
    }
    return false;
}


// Solver NU

class SolverNU final : public Solver {
public:
    SolverNU(std::size_t l, Mat<float>& Q, std::vector<float> QD, std::vector<float> p, std::vector<char> y,
             std::vector<float> alpha, std::pair<float, float> C, float eps, bool shrinking)
            : Solver(l, Q, std::move(QD), std::move(p), std::move(y), std::move(alpha), C, eps, shrinking) {}

    std::tuple<bool, std::size_t, std::size_t> selectWorkingSet() const;
    void calculateRho(Solution& sol) const;
    void doShrinking();
    bool beShrunk(std::size_t i, float g_max1, float g_max2, float g_max3, float g_max4) const;
};

// [0] : return whether if already optimal
// [1], [2] : return i, j such that
// i : maximizes -y_i * grad(f)_i in I_high(alpha)
// j : minimizes the decreases of obj value
// (if quadratic coefficient <= 0, replace it with tau)
// -y_j * grad(f)_j < -y_i * grad(f)_i, j in I_low(alpha)
std::tuple<bool, std::size_t, std::size_t> SolverNU::selectWorkingSet() const {
    float g_maxp1 = std::numeric_limits<float>::lowest();
    float g_maxn1 = std::numeric_limits<float>::lowest();
    std::size_t g_maxp_idx = -1;
    std::size_t g_maxn_idx = -1;
    for (auto i : active_set_) {
        if (y_[i] == +1) {
            if (alpha_status_[i] != Alpha::Upper) {
                if (g_maxp1 <= -G_[i]) {
                    g_maxp1 = -G_[i];
                    g_maxp_idx = i;
                }
            }
        } else {
            if (alpha_status_[i] != Alpha::Lower) {
                if (g_maxn1 <= G_[i]) {
                    g_maxn1 = G_[i];
                    g_maxn_idx = i;
                }
            }
        }
    }

    std::size_t ip = g_maxp_idx;
    std::vector<float> Qip;
    if (ip != -1) {
        Qip = getQ(ip);
    }
    std::size_t in = g_maxn_idx;
    std::vector<float> Qin;
    if (in != -1) {
        Qin = getQ(in);
    }
    std::size_t g_min_idx = -1;
    float g_maxp2 = std::numeric_limits<float>::lowest();
    float g_maxn2 = std::numeric_limits<float>::lowest();
    float obj_diff_min = std::numeric_limits<float>::max();
    for (auto j : active_set_) {
        if (y_[j] == +1) {
            if (alpha_status_[j] != Alpha::Lower) {
                float grad_diff = g_maxp1 + G_[j];
                if (G_[j] >= g_maxp2) {
                    g_maxp2 = G_[j];
                }
                if (grad_diff > 0) {
                    float obj_diff = 0.0f;
                    float quad_coeff = QD_[ip] + QD_[j] - 2.0f * Qip[j];
                    if (quad_coeff > 0) {
                        obj_diff = -std::pow(grad_diff, 2.0f) / quad_coeff;
                    } else {
                        obj_diff = -std::pow(grad_diff, 2.0f) / TAU;
                    }
                    if (obj_diff <= obj_diff_min) {
                        g_min_idx = j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        } else {
            if (alpha_status_[j] != Alpha::Upper) {
                float grad_diff = g_maxn1 - G_[j];
                if (-G_[j] >= g_maxn2) {
                    g_maxn2 = -G_[j];
                }
                if (grad_diff > 0) {
                    float obj_diff = 0.0f;
                    float quad_coeff = QD_[in] + QD_[j] - 2.0f * Qin[j];
                    if (quad_coeff > 0) {
                        obj_diff = -std::pow(grad_diff, 2.0f) / quad_coeff;
                    } else {
                        obj_diff = -std::pow(grad_diff, 2.0f) / TAU;
                    }
                    if (obj_diff <= obj_diff_min) {
                        g_min_idx = j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
    }
    if (std::max(g_maxp1 + g_maxp2, g_maxn1 + g_maxn2) < eps_ || g_min_idx == -1) {
        return {true, -1, -1};
    }
    return {false, (y_[g_min_idx] == +1) ? g_maxp_idx : g_maxn_idx, g_min_idx};
}

void SolverNU::calculateRho(Solution& sol) const {
    float res = 0.0f;
    std::size_t count_free1 = 0;
    std::size_t count_free2 = 0;
    float ub1 = std::numeric_limits<float>::max();
    float ub2 = std::numeric_limits<float>::max();
    float lb1 = std::numeric_limits<float>::lowest();
    float lb2 = std::numeric_limits<float>::lowest();
    float sum_free1 = 0.0f;
    float sum_free2 = 0.0f;
    for (auto i : active_set_) {
        if (y_[i] == +1) {
            if (alpha_status_[i] == Alpha::Upper) {
                lb1 = std::max(lb1, G_[i]);
            } else if (alpha_status_[i] == Alpha::Lower) {
                ub1 = std::min(ub1, G_[i]);
            } else {
                ++count_free1;
                sum_free1 += G_[i];
            }
        } else {
            if (alpha_status_[i] == Alpha::Upper) {
                lb2 = std::max(lb2, G_[i]);
            } else if (alpha_status_[i] == Alpha::Lower) {
                ub2 = std::min(ub2, G_[i]);
            } else {
                ++count_free2;
                sum_free2 += G_[i];
            }
        }
    }
    float r1 = 0.0f;
    float r2 = 0.0f;
    if (count_free1) {
        r1 = sum_free1 / count_free1;
    } else {
        r1 = (ub1 + lb1) / 2.0f;
    }
    if (count_free2) {
        r2 = sum_free2 / count_free2;
    } else {
        r2 = (ub2 + lb2) / 2.0f;
    }
    sol.r = (r1 + r2) / 2.0f;
    sol.rho = (r1 - r2) / 2.0f;
}

void SolverNU::doShrinking() {
    float g_max1 = std::numeric_limits<float>::lowest();
    float g_max2 = std::numeric_limits<float>::lowest();
    float g_max3 = std::numeric_limits<float>::lowest();
    float g_max4 = std::numeric_limits<float>::lowest();

    for (auto i : active_set_) {
        if (alpha_status_[i] != Alpha::Upper) {
            if (y_[i] == +1) {
                g_max1 = std::max(g_max1, -G_[i]);
            } else {
                g_max4 = std::max(g_max4, -G_[i]);
            }
        }
        if (alpha_status_[i] != Alpha::Lower) {
            if (y_[i] == +1) {
                g_max2 = std::max(g_max2, G_[i]);
            } else {
                g_max3 = std::max(g_max3, G_[i]);
            }
        }
    }
    if (!unshrink_ && std::max(g_max1 + g_max2, g_max3 + g_max4) <= eps_ * 10) {
        unshrink_ = true;
        reconstructGradient();
    }

    std::unordered_set<std::size_t> to_shrunk;
    for (auto i : active_set_) {
        if (beShrunk(i, g_max1, g_max2, g_max3, g_max4)) {
            to_shrunk.insert(i);
        }
    }
    for (auto i : to_shrunk) {
        active_set_.erase(i);
    }
}

bool SolverNU::beShrunk(std::size_t i, float g_max1, float g_max2,
                        float g_max3, float g_max4) const {
    assert(i < l_);
    if (alpha_status_[i] == Alpha::Upper) {
        if (y_[i] == +1) {
            return (-G_[i] > g_max1);
        } else {
            return (-G_[i] > g_max4);
        }
    } else if (alpha_status_[i] == Alpha::Lower) {
        if (y_[i] == +1) {
            return (G_[i] > g_max2);
        } else {
            return (G_[i] > g_max3);
        }
    }
    return false;
}

} // namespace frozenca

#endif //FROZENCA_SVMSOLVER_H
