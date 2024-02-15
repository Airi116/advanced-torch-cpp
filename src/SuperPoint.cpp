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
    : m_param(param)
    , m_device(torch::kCPU)
{
    if (m_param.imageHeight <= 0 || m_param.imageWidth <= 0) {
        throw std::runtime_error("dimension must be more than 0");
    }

    if (m_param.pathToWeights.empty()) {
        throw std::runtime_error("empty path to weights");
    }
    try {
        m_module = torch::jit::load(m_param.pathToWeights);
    } catch (const std::exception& e) {
        INFO_LOG("%s", e.what());
        exit(1);
    }

#if ENABLE_GPU
    if (!torch::cuda::is_available() && m_param.gpuIdx >= 0) {
        DEBUG_LOG("torch does not recog