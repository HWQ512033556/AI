/**
 * @file semantic_cluster.cpp.c
 * @brief Descriptions(TODO.()) ...
 * @author (metoak), huwenqian@metoak.office
 * @version 0.0.1
 * @date 06/22/2022 11:31:14
 * @copyright Copyright © 2022 metoak(Beijing), LLC. all rights reserved.
 */

#include "semantic_cluster.hpp"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include "queue.h"
#define PRINT_CT (0)
#define DEBUG (0)

#define RGB (3)
#define EXP2_N (1)
#define OFFSET (1 << EXP2_N)
static int32_t nWidth = 0;
static int32_t nHeight = 0;
static int8_t nLowestPtsNum = 5;
struct RectInfo {
    int8_t nPixelVal = 0;
    int32_t nPtsNum = 0;
    int32_t nMinCol = INT_MAX;
    int32_t nMinRow = INT_MAX;
    int32_t nMaxCol = INT_MIN;
    int32_t nMaxRow = INT_MIN;
};
static uint8_t arrWhiteList[256] = {
    0,  // 0
    0,  // 1
    0,  // 2
    0,  // 3
    0,  // 4
    1,  // 5
    1,  // 6
    1,  // 7
    0,  // 8
    0,  // 9
    1,  // 10
    1,  // 11
    0,  // 12
};
static void GetMinMaxValue(int32_t r, int32_t c, RectInfo &rectInfo);
static void Update8neighborPoint(cv::Point neighbors[8], int32_t i, int32_t j);
static int32_t ComponentLabel(const uint8_t *pBinaryImageData, uint8_t *pLabeledImageData);
static void GetLabelRect(uint8_t *pLabeledImageData, const uint8_t *input, std::vector<RectInfo> &vLabelPtsList);
static void Label2RGB(uint8_t *pLabeledImageData, unsigned char *rgb_image, int32_t num_segment);

int32_t moAILabel2Objects(const cv::Mat &mLabel, std::vector<MOAIRect> &vBBs) {
#if PRINT_CT
    time_t tStart = clock();
#endif
    if (mLabel.cols <= 0 || mLabel.rows <= 0 || mLabel.data == nullptr || mLabel.channels() != 1) {
        printf("paramsIn error!\n");
        return -1;
    }
    vBBs.clear();
    nHeight = mLabel.rows;
    nWidth = mLabel.cols;
    const uint8_t *pBinaryData = NULL;
    pBinaryData = (const uint8_t *)mLabel.data;
    uint8_t *pLabeledImageData = NULL;
    pLabeledImageData = new uint8_t[nWidth * nHeight];
    int32_t segments = ComponentLabel(pBinaryData, pLabeledImageData);
    if (segments < 1) {
        printf("no rect pixiel > 1\n");
        delete[] pLabeledImageData;
        return -1;
    }
#if DEBUG
    unsigned char *input = new unsigned char[nWidth * nHeight * 3];
    for (int32_t i = 0; i < nHeight; i++)
        for (int32_t j = 0; j < nWidth; j++)
            for (int32_t k = 0; k < 3; k++) input[i * nWidth * 3 + j * 3 + k] = 0;
    Label2RGB(pLabeledImageData, input, segments);

    // input mat --> RGB//TODO
    cv::Mat labelMat = cv::Mat(nHeight, nWidth, CV_8UC3, (unsigned char *)input);
    cv::imshow("labelMat", labelMat);

#endif
    std::vector<RectInfo> vRectList;
    vRectList.resize(segments);
    GetLabelRect(pLabeledImageData, pBinaryData, vRectList);

    MOAIRect moAiRect;
    int32_t nLeft, nTop, nRight, nButtom, nRectWidth, nRectHeight;
    for (int32_t i = 0; i < segments; i++) {
        if (vRectList[i].nPtsNum > (nLowestPtsNum >> EXP2_N)) {
            nLeft = vRectList[i].nMinCol - EXP2_N;
            nTop = vRectList[i].nMinRow - EXP2_N;
            nRight = vRectList[i].nMaxCol + EXP2_N;
            nButtom = vRectList[i].nMaxRow + EXP2_N;
            nLeft = nLeft < 0 ? 0 : nLeft;
            nTop = nTop < 0 ? 0 : nTop;
            nRight = nRight < nWidth ? nRight : (nWidth - 1);
            nButtom = nButtom < nHeight ? nButtom : (nHeight - 1);

            nRectWidth = nRight - nLeft + 1;
            nRectHeight = nButtom - nTop + 1;
            moAiRect.rBB = cv::Rect(nLeft, nTop, nRectWidth, nRectHeight);
            moAiRect.uType = vRectList[i].nPixelVal;
            std::strcpy(moAiRect.sType, gAI4SSLabel[vRectList[i].nPixelVal].c_str());
            // gAI4SSLabel[vRectList[i].nPixelVal].copy(moAiRect.sType, 32, 0);
            float fRectAera = (float)(nRectWidth >> EXP2_N) * (nRectHeight >> EXP2_N);
            moAiRect.fConfidence = vRectList[i].nPtsNum / fRectAera;
            vBBs.push_back(moAiRect);
        }
    }

#if DEBUG
    std::cout << "current pLabelImageData number: " << segments << std::endl;
    cv::Mat resultImg = mLabel.clone();
    cv::cvtColor(resultImg * 50, resultImg, cv::COLOR_GRAY2BGR);
    for (int32_t i = 0; i < (int32_t)vBBs.size(); i++) {
        printf("rect:%d,%d,%d,%d,uType:%d,fConfidence:%f\n", vBBs[i].rBB.x, vBBs[i].rBB.y, vBBs[i].rBB.width,
               vBBs[i].rBB.height, vBBs[i].uType, vBBs[i].fConfidence);
        cv::Rect rect = vBBs[i].rBB;
        std::rand();
        cv::Scalar scalar(std::rand() % 256, std::rand() % 256, std::rand() % 256);
        cv::rectangle(resultImg, rect, scalar, 1, 8, 0);
        std::string text = std::to_string((uint8_t)vBBs[i].uType) + " " + std::to_string(vBBs[i].fConfidence);
        cv::putText(resultImg, text, cv::Point(rect.x, rect.y), 1, 1, scalar, 1);
    }
    cv::imshow("resultImg", resultImg);
#endif
#if DEBUG
    delete[] input;
#endif
    delete[] pLabeledImageData;
#if PRINT_CT
    time_t tEnd = clock();
    std::cout << difftime(tEnd, tStart) / 10000 << std::endl;
#endif
    return 0;
}

void Update8neighborPoint(cv::Point neighbors[8], int32_t i, int32_t j) {
    cv::Point up;
    up.y = i - OFFSET;
    up.x = j;
    neighbors[0] = up;
    cv::Point down;
    down.y = i + OFFSET;
    down.x = j;
    neighbors[1] = down;
    cv::Point left;
    left.y = i;
    left.x = j - OFFSET;
    neighbors[2] = left;
    cv::Point right;
    right.y = i;
    right.x = j + OFFSET;
    neighbors[3] = right;
    cv::Point upLeft;
    upLeft.y = i - OFFSET;
    upLeft.x = j - OFFSET;
    neighbors[4] = upLeft;
    cv::Point upRight;
    upRight.y = i - OFFSET;
    upRight.x = j + OFFSET;
    neighbors[5] = upRight;

    cv::Point downLeft;
    downLeft.y = i + OFFSET;
    downLeft.x = j - OFFSET;
    neighbors[6] = downLeft;
    cv::Point downRight;
    downRight.y = i + OFFSET;
    downRight.x = j + OFFSET;
    neighbors[7] = downRight;
}

// This is the function that does the work of looping over the pBinaryData image and doing the connected component
// labeling See the project description for more details
int32_t ComponentLabel(const uint8_t *pBinaryImageData, uint8_t *pLabelImageData) {
    // set pLabelImageData to 1
    int32_t nCurrentLabel = 1;
    std::memset(pLabelImageData, 0, sizeof(uint8_t) * nWidth * nHeight);
    Queue BFS(nHeight * nWidth);

    // Look through every pixel in the image
    for (int32_t i = 0; i < nHeight; i += OFFSET) {
        const uint8_t *pBinaryHeaderRow = pBinaryImageData + i * nWidth;
        uint8_t *pLabelHeaderRow = pLabelImageData + i * nWidth;
        for (int32_t j = 0; j < nWidth; j += OFFSET, pBinaryHeaderRow += OFFSET, pLabelHeaderRow += OFFSET) {
            // check if pLabelImageData is white and not labled then pLabelImageData and push to queue
            int8_t nBinaryPixel = *pBinaryHeaderRow;
            if (arrWhiteList[nBinaryPixel] && *pLabelHeaderRow == 0) {
                *pLabelHeaderRow = nCurrentLabel;
                cv::Point loc;
                loc.y = i;
                loc.x = j;
                // store position
                BFS.push(loc);
                cv::Point neighbors[8];
                while (!BFS.is_empty()) {
                    // pops out element from queue
                    cv::Point curr = BFS.pop();
                    unsigned char pixel_value = nBinaryPixel;
                    // looks at its neighbor
                    Update8neighborPoint(neighbors, curr.y, curr.x);

                    // look through neighbors
                    for (int32_t k = 0; k < 8; k++) {
                        cv::Point pixel = neighbors[k];
                        // checks if in bounds
                        if (pixel.y < 0 || pixel.x < 0 || pixel.y >= nHeight || pixel.x >= nWidth) continue;
                        // checking if white and not labeled (for neighbors) andgiven current pLabelImageData
                        int32_t nIndex = pixel.y * nWidth + pixel.x;
                        if (pBinaryImageData[nIndex] == pixel_value && pLabelImageData[nIndex] == 0) {
                            pLabelImageData[nIndex] = nCurrentLabel;
                            // store position
                            BFS.push(pixel);
                        }
                        // if black or white but labeled
                        else {
                            continue;
                        }
                    }
                }
                // if queue is empty increases current pLabelImageData by 1
                nCurrentLabel += 1;
            }
        }
    }
    return nCurrentLabel;
}

// First make num_segments number of random colors to use for coloring the labeled parts of the image.
// Then loop over the labeled image using the pLabelImageData to index into your random colors array.
// Set the rgb_pixel to the corresponding color, or set to black if the pixel was unlabeled.
void Label2RGB(uint8_t *pLabeledImageData, unsigned char *rgb_image, int32_t num_segments) {
    // colors matrix
    int32_t colors[num_segments][3];
    for (int32_t i = 0; i < num_segments; i++) {
        // creates random colors
        colors[i][0] = rand() % 256;
        colors[i][1] = rand() % 256;
        colors[i][2] = rand() % 256;
    }
    for (int32_t i = 0; i < nHeight; i += OFFSET) {
        for (int32_t j = 0; j < nWidth; j += OFFSET) {
            int32_t c = pLabeledImageData[i * nWidth + j] - 1;  //-1 so it doesn't go out of bounds
            // if pLabelImageData is zero set it to black
            if (pLabeledImageData[i * nWidth + j] == 0) {
                rgb_image[i * nWidth * 3 + j * 3 + 0] = 0;
                rgb_image[i * nWidth * 3 + j * 3 + 1] = 0;
                rgb_image[i * nWidth * 3 + j * 3 + 2] = 0;
            }
            // sets identified regions to different colors
            else {
                int32_t c = pLabeledImageData[i * nWidth + j] - 1;
                rgb_image[i * nWidth * 3 + j * 3 + 0] = colors[c][0];
                rgb_image[i * nWidth * 3 + j * 3 + 1] = colors[c][1];
                rgb_image[i * nWidth * 3 + j * 3 + 2] = colors[c][2];
            }
        }
    }
}

void GetMinMaxValue(int32_t r, int32_t c, RectInfo &rectInfo) {
    if (r <= rectInfo.nMinRow) {
        rectInfo.nMinRow = r;
    }
    if (r >= rectInfo.nMaxRow) {
        rectInfo.nMaxRow = r;
    }
    if (c <= rectInfo.nMinCol) {
        rectInfo.nMinCol = c;
    }
    if (c >= rectInfo.nMaxCol) {
        rectInfo.nMaxCol = c;
    }
}

void GetLabelRect(uint8_t *pLabeledImageData, const uint8_t *pBinaryData, std::vector<RectInfo> &vRectList) {
    for (int32_t j = 0; j < nHeight; j += OFFSET) {
        uint8_t *pLabel = pLabeledImageData + j * nWidth;
        for (int32_t i = 0; i < nWidth; i += OFFSET, pLabel += OFFSET) {
            if (*pLabel > 0) {
                if (vRectList[*pLabel].nPixelVal == 0) {
                    const uint8_t *pPixel = pBinaryData + j * nWidth + i;
                    vRectList[*pLabel].nPixelVal = *pPixel;
                }
                vRectList[*pLabel].nPtsNum++;
                GetMinMaxValue(j, i, vRectList[*pLabel]);
            }
        }
    }
}
