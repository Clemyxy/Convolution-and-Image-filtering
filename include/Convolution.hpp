#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include <typeinfo>
#include <iostream>
#include <vector>
#include <utility>

using namespace cv;

class Convolution {
private:
    static Mat convolutionGrayscale(const Mat &src, Mat kernel);

public:
    static Mat convolution(const Mat &src, Mat kernel);
};

