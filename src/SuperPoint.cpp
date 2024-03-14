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
        DEBUG_LOG("torch does not recognize cuda device so fall back to cpu...");
        m_param.gpuIdx = -1;
    }
#else
    DEBUG_LOG("gpu option is not enabled...");
    m_param.gpuIdx = -1;
#endif

    if (m_param.gpuIdx >= 0) {
        torch::NoGradGuard no_grad;
        m_device = torch::Device(torch::kCUDA, m_param.gpuIdx);
    }
    DEBUG_LOG("use device: %s", m_device.str().c_str());
    m_module.eval();

    if (!m_device.is_cpu()) {
        m_module.to(m_device);
    }
}

void SuperPointImpl::detectAndCompute(cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& keyPoints,
                                      cv::OutputArray _descriptors, bool useProvidedKeypoints)
{
    cv::Mat image = _image.getMat();
    cv::Mat mask = _mask.getMat();

    if (image.empty() || image.depth() != CV_8U) {
        CV_Error(cv::Error::StsBadArg, "image is empty or has incorrect depth (!=CV_8U)");
    }

    if (!mask.empty() && mask.type() != CV_8UC1) {
        CV_Error(cv::Error::StsBadArg, "mask has incorrect type (!=CV_8UC1)");
    }

    torch::Dict<std::string, std::vector<torch::Tensor>> outputs;
    {
        cv::Mat buffer;
      