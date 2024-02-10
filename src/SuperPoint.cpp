/**
 * @file    SuperPoint.cpp
 *
 * @author  btran
 *
 */

#include <memory>

#include <torch/script.h>
#include <torch/torch.h>

#include <torch_cpp/SuperPoint.hpp>
#include <torch_cpp/Utility.hpp>

namespace
{
cv::Mat copyRows(const cv::Mat& sr