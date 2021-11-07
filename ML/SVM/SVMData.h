#ifndef FROZENCA_SVMDATA_H
#define FROZENCA_SVMDATA_H

#include <array>
#include <charconv>
#include <ranges>
#include <iostream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace frozenca {

using Samples = std::vector<std::vector<float>>;
using SubSamples = std::vector<std::reference_wrapper<const std::vector<float>>>;

class SVMTrainData {
private:
    std::variant<Samples, SubSamples> X_;
public:
    std::vector<float> y_;
    std::size_t n_;
    std::size_t C_;
    std::vector<std::pair<float, float>> feature_bounds_;
    friend class SVMModel;

    // full data construct
    explicit SVMTrainData(std::istream& is) {
        std::vector<std::vector<float>> X;
        is.seekg(0, std::ios::end);
        std::streamsize len = is.tellg();
        is.seekg(0, std::ios::beg);

        constexpr std::size_t buf_len = 1024;

        std::array<char, buf_len> buffer = {0};
        float target = 0.0f;
        std::vector<float> row;
        std::size_t col_index = 0;
        while (is.getline(&buffer[0], buf_len)) {
            auto curr = buffer.begin();
            auto end = buffer.end();
            while (curr != end) {
                float result = 0;
                auto delim = std::find_if(curr, end, [](int ch) {
                    return ch == ',' || ch == '\n' || ch == '\0';
                });
                std::string_view sv(curr, delim);
                std::from_chars(sv.begin(), sv.end(), result);
                if (!col_index) {
                    y_.push_back(result);
                } else {
                    row.push_back(result);
                }
                curr = delim + 1;
                ++col_index;
                if (*delim == '\n') {
                    C_ = row.size();
                    X.push_back(std::move(row));
                    row = {};
                    col_index = 0;
                    target = 0.0f;
                }
            }
        }
        assert(!X.empty());
        n_ = y_.size();
        if (n_ != X.size() ||
            std::ranges::any_of(X, [&](const auto& row) { return row.size() != C_; })) {
            throw std::invalid_argument("Number of features of sample does not match");
        }
        X_ = std::move(X);
        scale();
    }

    SVMTrainData(Samples X, std::vector<float> y) : X_(std::move(X)), y_(std::move(y)), n_(y_.size()), C_(std::get<Samples>(X_)[0].size()) {
        scale();
    }

    // constructor for subsamples
    SVMTrainData(SubSamples X, std::vector<float> y, std::vector<std::pair<float, float>> feature_bounds)
    : X_(std::move(X)), y_(std::move(y)), n_(y_.size()), C_(std::get<SubSamples>(X_)[0].get().size()),
    feature_bounds_(std::move(feature_bounds)) {}

    [[nodiscard]] Samples getX() const {
        if (std::holds_alternative<Samples>(X_)) {
            return std::get<Samples>(X_);
        } else {
            throw std::runtime_error("getX() called for subsample set");
        }
    }

    [[nodiscard]] SubSamples getXRef() const {
        if (std::holds_alternative<SubSamples>(X_)) {
            return std::get<SubSamples>(X_);
        } else {
            throw std::runtime_error("getXRef() called for full sample set");
        }
    }

    [[nodiscard]] std::vector<float> getX(std::size_t row_index) const {
        if (std::holds_alternative<Samples>(X_)) {
            return std::get<Samples>(X_)[row_index];
        } else {
            return std::get<SubSamples>(X_)[row_index].get();
        }
    }

    [[nodiscard]] std::reference_wrapper<const std::vector<float>> getXRef(std::size_t row_index) const {
        if (std::holds_alternative<Samples>(X_)) {
            return std::ref(std::get<Samples>(X_)[row_index]);
        } else {
            return std::get<SubSamples>(X_)[row_index];
        }
    }

private:
    void scale() {
        if (std::holds_alternative<SubSamples>(X_)) {
            return;
        }
        feature_bounds_ = std::vector<std::pair<float, float>>(C_, {std::numeric_limits<float>::max(),
              std::numeric_limits<float>::lowest()});
        auto& X = std::get<Samples>(X_);
        for (std::size_t i = 0; i < n_; ++i) {
            for (std::size_t c = 0; c < C_; ++c) {
                feature_bounds_[c].first = std::min(feature_bounds_[c].first, X[i][c]);
                feature_bounds_[c].second = std::max(feature_bounds_[c].second, X[i][c]);
            }
        }

        for (std::size_t c = 0; c < C_; ++c) {
            auto [fmin, fmax] = feature_bounds_[c];
            for (std::size_t i = 0; i < n_; ++i) {
                X[i][c] = 1.0f - 2.0f * (fmax - X[i][c])/(fmax - fmin);
            }
        }
    }

};

} // namespace frozenca

#endif //FROZENCA_SVMDATA_H
