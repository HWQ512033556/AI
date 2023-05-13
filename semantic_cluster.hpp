/**
 * @file semantic_cluster.hpp.h
 * @brief Descriptions(TODO.()) ...
 * @author (metoak), huwenqian@metoak.office
 * @version 0.0.1
 * @date 06/22/2022 11:30:06
 * @copyright Copyright © 2022 metoak(Beijing), LLC. all rights reserved.
 */
#ifndef SEMANTIC_CLUSTER_HPP_NDw0QrE
#define SEMANTIC_CLUSTER_HPP_NDw0QrE

/// headers
#include <unistd.h>
#include <moAI4SS/moai4ss.hpp>

/**
 * @brief  moAILabel2Ojbects analyse AI label image: extract same cluster label bounding-box
 *
 * @param mLabel the label image CV_8UC1
 * @param vBBs the output bounding-box
 *
 * @returns  0:success, other:failed
 */
/* ---------------------------------------------------------------*/
int32_t moAILabel2Objects(const cv::Mat &mLabel, std::vector<MOAIRect> &vBBs);

#endif /** end of SEMANTIC_CLUSTER_HPP_NDw0QrE define **/
