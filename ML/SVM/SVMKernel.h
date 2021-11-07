#ifndef FROZENCA_SVMKERNEL_H
#define FROZENCA_SVMKERNEL_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <ranges>
#include <vector>

namespace frozenca {

static float dot(const std::vector<float>& x, const std::vector<float>& y) {
    return std::inner_product(x.begin(), x.end(), y.begin(), 0.0f);
}

class Kernel {
public:
    virtual ~Kernel() = default;
    [[nodiscard]] virtual float operator()(const std::vector<float>& x, const std::vector<float>& y) const = 0;
    virtual void adjustParams(std::size_t f) = 0;
};

class KernelLinear final : public Kernel {
public:
    [[nodiscard]] float operator()(const std::vector<float>& x, const std::vector<float>& y) const {
        return dot(x, y);
    }
    void adjustParams(std::size_t f) final {
        // do nothing
    }
};

class KernelPoly final : public Kernel {
private:
    float gamma_ = 0.0f;
    float coef0_ = 0.0f;
    std::size_t degree_ = 3;
public:
    [[nodiscard]] float operator()(const std::vector<float>& x, const std::vector<float>& y) const{
        return std::pow(gamma_ * dot(x, y) + coef0_, degree_);
    }
    void adjustParams(std::size_t f) {
        gamma_ = 1.0f / f;
    }
};

class KernelRBF final : public Kernel {
private:
    float gamma_ = 0.0f;
public:
    [[nodiscard]] float operator()(const std::vector<float>& x, const std::vector<float>& y) const final {
        std::vector<float> diff = x;
        std::ranges::transform(diff, y, diff.begin(), std::minus{});
        return std::exp(-gamma_ * dot(diff, diff));
    }
    void adjustParams(std::size_t f) final {
        gamma_ = 1.0f / f;
    }
};

class KernelSigmoid final : public Kernel {
private:
    float gamma_ = 0.0f;
    float coef0_ = 0.0f;
public:
    [[nodiscard]] float operator()(const std::vector<float>& x, const std::vector<float>& y) const final {
        return std::tanh(gamma_ * dot(x, y) + coef0_);
    }
    void adjustParams(std::size_t f) final {
        gamma_ = 1.0f / f;
    }
};

} // namespace frozenca

#endif //FROZENCA_SVMKERNEL_H
