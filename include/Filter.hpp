#pragma once

#include <opencv2/core/core.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "Convolution.hpp"

class Filter {
private:
    std::vector<cv::Mat> filters;
public:
    Filter() = default;

    explicit Filter(const std::string &filename);

    ~Filter() = default;

    std::vector<cv::Mat> getFilters();

    //Thresholds must be between 0 and 255 and sh > sb
    static cv::Mat tresholding(const cv::Mat &src, const float &sh, const float &sb);

    cv::Mat ApplyFilter(const cv::Mat &src, const int &i);

    std::vector<cv::Mat> ApplyFilters(const cv::Mat &src, const int &nb);

};

