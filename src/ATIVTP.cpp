#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <utility>
#include "../include/Filter.hpp"

#define _USE_MATH_DEFINES

namespace fs = std::filesystem;
using namespace cv;

const static float mpi8 = -M_PI / 8;
const static float ppi8 = M_PI / 8;
const static float m7pi8 = -7 * M_PI / 8;
const static float p7pi8 = 7 * M_PI / 8;
const static float m3pi8 = -3 * M_PI / 8;
const static float p3pi8 = 3 * M_PI / 8;
const static float m5pi8 = -5 * M_PI / 8;
const static float p5pi8 = 5 * M_PI / 8;

class SlopeOutOfBoundException : public std::exception {
private:
    float m_slope;
public:
    explicit SlopeOutOfBoundException(float slope) : m_slope(slope), std::exception() {}

    [[nodiscard]] const char *what() const _GLIBCXX_TXN_SAFE_DYN _GLIBCXX_NOTHROW override {
        char *buffer;
        std::sprintf(buffer, "Error on slope parsing, value: %f", m_slope);
        return buffer;
    }
};

int parseSlope(const float &slope) {
    if ((slope >= mpi8 && slope < ppi8) || (slope >= p7pi8 || slope < m7pi8)) {
        return 0;
    } else if ((slope >= ppi8 && slope < p3pi8) || (slope >= m7pi8 && slope < m5pi8)) {
        return 1;
    } else if ((slope >= p3pi8 && slope < p5pi8) || (slope >= m5pi8 && slope < m3pi8)) {
        return 2;
    } else if ((slope >= p5pi8 && slope < p7pi8) || (slope >= m3pi8 && slope < mpi8)) {
        return 3;
    } else {
        throw SlopeOutOfBoundException(slope);
    }
}

cv::Mat applyFilter2D(const cv::Mat &src, Filter &filters, const float &htr, const float &ltr) {
    std::vector<cv::Mat> grads;
    cv::Mat amp, slope, result, mask;

    //Obtention gradients x et y
    grads = filters.ApplyFilters(src, 2);

    //Obtention amplitude
    amp = grads[0].mul(grads[0]) + grads[1].mul(grads[1]);
    amp.forEach<float>(
            [](float &value, const int *position) -> void {
                value = std::sqrt(value);
            }
    );

    result = Mat(amp.size(), CV_32FC1, Scalar(0));
    //Maximum locaux
    float res;
    for (int i = 1; i < amp.size().height - 1; ++i)
        for (int j = 1; j < amp.size().width - 1; ++j) {
            switch (parseSlope(std::atan2(grads[1].at<float>(i, j), grads[0].at<float>(i, j)))) {
                case 0:
                    res = amp.at<float>(i, j) * 2 - amp.at<float>(i, j - 1) - amp.at<float>(i, j + 1);
                    if (res > 0)
                        result.at<float>(i, j) = amp.at<float>(i, j);
                    break;
                case 1:
                    res = amp.at<float>(i, j) * 2 - amp.at<float>(i - 1, j + 1) - amp.at<float>(i + 1, j - 1);
                    if (res > 0)
                        result.at<float>(i, j) = amp.at<float>(i, j);
                    break;
                case 2:
                    res = amp.at<float>(i, j) * 2 - amp.at<float>(i - 1, j) - amp.at<float>(i + 1, j);
                    if (res > 0)
                        result.at<float>(i, j) = amp.at<float>(i, j);
                    break;
                case 3:
                    res = amp.at<float>(i, j) * 2 - amp.at<float>(i - 1, j - 1) - amp.at<float>(i + 1, j + 1);
                    if (res > 0)
                        result.at<float>(i, j) = amp.at<float>(i, j);
                    break;
                default:
                    throw std::invalid_argument("Error finding local maximum");
            }
        }

    //Hysterisis
    mask = Filter::tresholding(result, htr, ltr);
    return mask;
}

cv::Mat applyFilter4D(const cv::Mat &src, Filter &filter, const float &htr, const float &ltr) {
    std::vector<cv::Mat> grads;
    cv::Mat amp, slope, result, mask;
    slope = cv::Mat(src.size(), CV_32FC1, Scalar(0));
    amp = cv::Mat(src.size(), CV_32FC1, Scalar(0));

    //obtention des gradients et de la pente en chaque pts
    grads = filter.ApplyFilters(src, 4);
    for (int i = 0; i < src.size().height; ++i) {
        for (int j = 0; j < src.size().width; ++j) {
            float max = std::abs(grads[0].at<float>(i, j));
            float slo = 0.f;
            for (int k = 1; k < 4; ++k) {
                float num = std::abs(grads[k].at<float>(i, j));
                if (max < num) {
                    max = num;
                    slo = (float) i;
                }
            }
            amp.at<float>(i, j) = max;
            slope.at<float>(i, j) = slo * ((float) M_PI / 4.f);
        }
    }

    result = Mat(amp.size(), CV_32FC1, Scalar(0));
    //Maximum locaux
    float res;
    for (int i = 1; i < amp.size().height - 1; ++i)
        for (int j = 1; j < amp.size().width - 1; ++j) {
            switch (parseSlope(std::atan2(grads[1].at<float>(i, j), grads[0].at<float>(i, j)))) {
                case 0:
                    res = amp.at<float>(i, j) * 2 - amp.at<float>(i, j - 1) - amp.at<float>(i, j + 1);
                    if (res > 0)
                        result.at<float>(i, j) = amp.at<float>(i, j);
                    break;
                case 1:
                    res = amp.at<float>(i, j) * 2 - amp.at<float>(i - 1, j + 1) - amp.at<float>(i + 1, j - 1);
                    if (res > 0)
                        result.at<float>(i, j) = amp.at<float>(i, j);
                    break;
                case 2:
                    res = amp.at<float>(i, j) * 2 - amp.at<float>(i - 1, j) - amp.at<float>(i + 1, j);
                    if (res > 0)
                        result.at<float>(i, j) = amp.at<float>(i, j);
                    break;
                case 3:
                    res = amp.at<float>(i, j) * 2 - amp.at<float>(i - 1, j - 1) - amp.at<float>(i + 1, j + 1);
                    if (res > 0)
                        result.at<float>(i, j) = amp.at<float>(i, j);
                    break;
                default:
                    throw std::invalid_argument("Error finding local maximum");
            }
        }
    //Hysterisis
    mask = Filter::tresholding(result, htr, ltr);
    return mask;
}

class BadInputException : public std::exception {
private:
    int m_code;
    String m_input;

public:
    BadInputException(int code, String input) : m_code(code), m_input(std::move(input)), std::exception() {}

    [[nodiscard]] const char *what() const _GLIBCXX_TXN_SAFE_DYN _GLIBCXX_NOTHROW override {
        char *err;
        switch (m_code) {
            case 0:
                std::sprintf(err, "Could not open or find the -%s- image !\n", m_input.c_str());
                break;
            case 1:
                std::sprintf(err, "Could not open or find the -%s- filter !\n", m_input.c_str());
                break;
            case 2:
                std::sprintf(err, "Index -%s- is out of range !\n", m_input.c_str());
                break;
            case 3:
                std::sprintf(err, "Filter type -%s- does not exist !\n", m_input.c_str());
                break;
            case 4:
                std::sprintf(err, "Filter type -%s- can't have multi-dimension !\n", m_input.c_str());
                break;
            case 5:
                std::sprintf(err, "Filter can't have -%s- dimension !\n", m_input.c_str());
                break;
        }

        const char* usageReminder = "Arguments: \n"
                                    "\t<Image : filename>\n"
                                    "\t<Filter : filename>\n"
                                    "\t<Filter type : 'b' for Blur | 'ed' for Edge detector>\n"
                                    "\t<Filter index : integer index for select one filter or 'all' for select all>\n"
                                    "\t<High threshold : decimal number> (only for 'ed' filter)\n"
                                    "\t<Low threshold : decimal number> (only for 'ed' filter)";

        return std::strcat(err, usageReminder);
    }
};

int main(int argc, char **argv) {
    CommandLineParser parser(argc, argv, "{@img | ../data/img/ville.jpg | image}"
                                         "{@filter | ../data/filter/Prewitt_4D.txt | filter}"
                                         "{@fil_type | ed | filter type}"
                                         "{@ind | all | filter index}"
                                         "{@htr | 0.2 | high threshold}"
                                         "{@ltr | 0.9 | low threshold}");

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << parser.get<String>("@img") << std::endl;
    Mat src;
    try {
        src = imread(samples::findFile(parser.get<String>("@img")), IMREAD_COLOR);
    } catch (const std::exception &e) {
        throw BadInputException(0, parser.get<String>("@img"));
    }

    Filter filter;
    try {
        filter = Filter(parser.get<String>("@filter"));
    } catch (const std::exception &e) {
        throw BadInputException(1, parser.get<String>("@filter"));
    }

    Mat img, dst, imgRs, tmp;
    cvtColor(src, img, COLOR_BGR2GRAY);
    img.convertTo(img, CV_32FC1);

    uint dim;
    int ind;
    if(parser.get<String>("@ind") == "all") {
        dim = filter.getFilters().size();
        ind = 0u;
    } else {
        dim = 1u;
        ind = parser.get<int>("@ind");
    }

    if(ind >= filter.getFilters().size()) {
        throw BadInputException(2, parser.get<String>("@ind"));
    }

    float htr, ltr;
    auto strType = parser.get<String>("@fil_type");
    if((strType != "b") && (strType != "ed")) {
        throw BadInputException(3, strType);
    }

    bool isED = strType == "ed";
    if(isED) {
        htr = parser.get<float>("@htr");
        ltr = parser.get<float>("@ltr");
    }

    switch (dim) {
        case 1:
            tmp = filter.ApplyFilter(img, ind);
            tmp.convertTo(tmp, CV_8UC1);
            break;
        case 2:
            if(!isED)
                throw BadInputException(4, parser.get<String>("@fil_type"));

            tmp = applyFilter2D(img, filter, htr, ltr);
            break;
        case 4:
            if(!isED)
                throw BadInputException(4, parser.get<String>("@fil_type"));

            tmp = applyFilter4D(img, filter, htr, ltr);
            break;
        default:
            throw BadInputException(5, std::to_string(dim));
    }

    imshow("Source image", src);
    imshow("Conv image", tmp);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Total execution time: " << duration.count() << std::endl;
    imwrite("../data/img/result.bmp", tmp);

    waitKey(0); // Wait for a keystroke in the window

    return 0;
}
