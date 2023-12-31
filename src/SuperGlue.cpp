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
    explicit SuperGlueImpl(const SuperGlue::Param& param)