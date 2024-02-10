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
                          cv::OutputArray _descriptors, bool useProvidedKeypoints) CV_OVERRIDE;

    int descriptorSize() const CV_OVERRIDE
    {
        return 256;
    }

    int descriptorType() const CV_OVERRIDE
    {
        return CV_32F;
    }

 private:
    SuperPoint::Param m_param;
    torch::Device m_device;
    torch::jit::script::Module m_module;
};

cv::Ptr<SuperPoint> SuperPoint::create(const Param& param)
{
    return cv::makePtr<SuperPointImpl>(param);
}

SuperPointImpl::SuperPointImpl(const SuperPoint::Param& param)
   