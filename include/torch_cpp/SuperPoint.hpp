/**
 * @file    SuperPoint.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <string>

#include <opencv2/opencv.hpp>

namespace _cv
{
class CV_EXPORTS_W SuperPoint : public cv::Feature2D
{
 public:
    struct Param {
        // reduce input shapes can increase speed and reduce (GPU) memory consumption
        // at the cost of accurac