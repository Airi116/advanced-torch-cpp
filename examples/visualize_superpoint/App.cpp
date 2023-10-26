
/**
 * @file    App.cpp
 *
 * @author  btran
 *
 */

#include <torch_cpp/torch_cpp.hpp>

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: [app] [path/to/superpoint/weights] [path/to/image]" << std::endl;
        return EXIT_FAILURE;
    }
    const std::string WEIGHTS_PATH = argv[1];
    const std::string IMAGE_PATH = argv[2];

    _cv::SuperPoint::Param param;
    param.pathToWeights = WEIGHTS_PATH;
    param.distThresh = 5;