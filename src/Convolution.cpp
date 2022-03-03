#include "Convolution.hpp"

Mat Convolution::convolutionGrayscale(const Mat &src, Mat kernel) {
    Mat tmp(src.size(), CV_32FC1, cv::Scalar(0));
    int kSize = kernel.size().width;
    int halfk = int(kSize / 2);
    int limH = tmp.size().height - halfk;
    int limW = tmp.size().width - halfk;
    float res;
    int ii, jj;
    for (int i = halfk; i < limH; ++i) {
        ii = i - halfk;
        auto *rowtmp = tmp.ptr<float>(i);
        for (int j = halfk; j < limW; ++j) {
            jj = j - halfk;
            res = 0;
            for (int ki = 0; ki < kSize; ++ki) {
                const auto *rowsrc = src.ptr<float>(ii + ki);
                const float *rowker = kernel.ptr<float>(ki);
                for (int kj = 0; kj < kSize; ++kj) {
                    res += rowsrc[jj + kj] * rowker[kj];
                }
            }
            rowtmp[j] = res;
        }
    }
    return tmp;
}

Mat Convolution::convolution(const Mat &src, Mat kernel) {
    int ch = src.channels();
    if (ch == 1) {
        return convolutionGrayscale(src, std::move(kernel));
    } else
        fprintf(stderr, "One of the matrix was not grayscale, convolution error, please use grayscale CV_32F Matrices");

    return {};
}