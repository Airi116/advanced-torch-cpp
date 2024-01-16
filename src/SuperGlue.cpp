/**
 * @file    SuperGlue.cpp
 *
 * @author  btran
 *
 */

#include <torch/script.h>
#include <torch/torch.h>

#include <torch_cpp/SuperGlue.hpp>
#include <torch_cpp/Utility.hpp>

namespace _cv
{
class SuperGlueImpl : public SuperGlue
{
 public:
    explicit SuperGlueImpl(const SuperGlue::Param& param);

    void match(cv::InputArray _queryDescriptors, const std::vector<cv::KeyPoint>& queryKeypoints,
               const cv::Size& querySize, cv::InputArray _trainDescriptors,
               const std::vector<cv::KeyPoint>& trainKeypoints, const cv::Size& trainSize,
               CV_OUT std::vector<cv::DMatch>& matches) const final;

 private:
    SuperGlue::Param m_param;
    torch::Device m_device;
    mutable torch::jit::script::Module m_module;
};

cv::Ptr<SuperGlue> SuperGlue::create(const Param& param)
{
    return cv::makePtr<SuperGlueImpl>(param);
}

SuperGlueImpl::SuperGlueImpl(const SuperGlue::Param& param)
    : m_param(param)
    , m_device(torch::kCPU)
{
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
    m_module.eval();
    m_module.to(m_device);
}

void SuperGlueImpl::match(cv::InputArray _queryDescriptors, const std::vector<cv::KeyPoint>& queryKeypoints,
                          const cv::Size& querySize, cv::InputArray _trainDescriptors,
                          const std::vector<cv::KeyPoint>& trai