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
cv::Mat copyRows(const cv::Mat& src, const std::vector<int>& indices);
}  // namespace

namespace _cv
{
class SuperPointImpl : public SuperPoint
{
 public:
    explicit SuperPointImpl(const SuperPoint::Param& param);

    void detectAndCompute(cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& keyPoints,
    