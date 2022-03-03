#include "Filter.hpp"

std::vector<cv::Mat> Filter::getFilters() {
    return filters;
}

Filter::Filter(const std::string &filename) {
    std::fstream fs;
    fs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fs.open(filename, std::ifstream::in);

    int nMatrix, mHeight, mWidth;
    float coeffInv;

    fs >> nMatrix >> mHeight >> mWidth >> coeffInv;

    for (auto i = 0u; i < nMatrix; i++) {
        auto tmpPosCoeffInv = 0;
        auto tmpNegCoeffInv = 0;

        std::vector<float> filter;
        filter.resize(mHeight * mWidth);
        for (auto j = 0u; j < mHeight; j++) {
            for (auto k = 0u; k < mWidth; k++) {
                fs >> filter[j * mWidth + k];
                if (filter[j * mWidth + k] > 0) {
                    tmpPosCoeffInv += (int) filter[j * mWidth + k];
                    filter[j * mWidth + k] /= coeffInv;
                } else {
                    tmpNegCoeffInv -= (int) filter[j * mWidth + k];
                    filter[j * mWidth + k] /= coeffInv;
                }
            }
        }
        cv::Mat tmp = cv::Mat(mHeight, mWidth, CV_32FC1, filter.data());
        filters.emplace_back(tmp.clone());

        assert(tmpPosCoeffInv == coeffInv || tmpPosCoeffInv == 0);
        assert(tmpNegCoeffInv == coeffInv || tmpNegCoeffInv == 0);
    }
}

// sh and sb both range [0;1] with sh being the top treshold (255 * sh).
// and sb being how lower than the top treshold, the low threshold is (255 * sh * sb)
cv::Mat Filter::tresholding(const cv::Mat &src, const float &sh, const float &sb) {
    cv::Mat tmp(src.size(), CV_32FC1, cv::Scalar(0));
    std::vector<std::pair<int, int>> mid;
    float test;
    double min, max;
    cv::minMaxIdx(src, &min, &max);
    float maxsh = ((float) max) * sh;
    float maxsb = ((float) max) * sh * sb;
    for (int i = 0; i < tmp.size().height; ++i)
        for (int j = 0; j < tmp.size().width; ++j) {
            test = src.at<float>(i, j);
            if (test > maxsh) {
                tmp.at<float>(i, j) = 255;
            } else if (test > maxsb) {
                mid.emplace_back(i, j);
            }
        }
    for (std::pair<int, int> &p: mid) {
        bool keep = false;
        for (int i = std::get<0>(p) - 1; i < std::get<0>(p) + 2 && !keep; ++i)
            for (int j = std::get<1>(p) - 1; j < std::get<1>(p) + 2 && !keep; ++j) {
                if (tmp.at<float>(i, j) <= 0) {
                    tmp.at<float>(std::get<0>(p), std::get<1>(p)) = 255;
                    keep = true;
                }
            }
    }
    return tmp;
}

cv::Mat Filter::ApplyFilter(const cv::Mat &src, const int &i) {
    return Convolution::convolution(src, filters[i]);
}

std::vector<cv::Mat> Filter::ApplyFilters(const cv::Mat &src, const int &nb) {
    std::vector<cv::Mat> tmp;
    for (int i = 0; i < nb; ++i)
        tmp.push_back(Convolution::convolution(src, filters[i]));

    return tmp;
}