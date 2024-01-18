//
// Created by Junseo Youn on 2023/04/09.
// Modified by confumbit on 2024/01/06.
//

#include "DebugLogger.h"
#include "ImageCompare.h"

ImageCompare::ImageCompare()
{
    clahe = createCLAHE();
    matrix = Mat::zeros(3, 3, CV_64F);
    matrix.at<double>(0, 0) = 1;
    matrix.at<double>(1, 1) = 1;
    matrix.at<double>(2, 2) = 1;
}

ImageCompare::~ImageCompare()
{
    cvMatchers->clear();
    cvMatchers.release();
    cvMatchers = nullptr;
    cvDetector->clear();
    cvDetector.release();
    cvDetector = nullptr;
    clahe.release();
    clahe = nullptr;
    keypointsMarker.clear();
    keypointsQuery.clear();
    descriptorsMarker.release();
    descriptorsQuery.release();
    imageQuery.release();
    imageQueryAligned.release();
    imageMarker.release();
    imageMarkerMasked.release();
    imageMaskMarker.release();
    imageMaskQuery.release();
    imageEqualizedMarker.release();
    imageEqualizedQuery.release();
}

void ImageCompare::setDetector(const Ptr<Detector> &detector)
{
    this->cvDetector = detector;
}

void ImageCompare::setMatchers(const Ptr<Matcher> &matcher)
{
    this->cvMatchers = matcher;
}

bool ImageCompare::compare()
{
    std::vector<std::vector<DMatch>> createMatches;
    if (descriptorsMarker.empty() || descriptorsQuery.empty())
    {
        logger_e("descriptors is empty");
        logger_e("descriptorsMarker: %d, descriptorsQuery: %d", descriptorsMarker.empty(),
                 descriptorsQuery.empty());
        return false;
    }
    try
    {
        cvMatchers->knnMatch(descriptorsMarker, descriptorsQuery, createMatches, Constants::knn);
    }
    catch (const cv::Exception &e)
    {
        logger_e("compare - knnMatch - cv::Exception: %s", e.what());
        return false;
    }
    catch (const std::exception &e)
    {
        logger_e("compare - knnMatch - std::exception: %s", e.what());
        return false;
    }

    std::vector<DMatch> selectMatches;
    selectMatches.reserve(createMatches.size());
    for (auto &match : createMatches)
    {
        if (match[0].distance < Constants::threshold * match[1].distance)
        {
            selectMatches.emplace_back(match[0].queryIdx, match[0].trainIdx, match[0].distance);
        }
    }

    if (selectMatches.size() <= 4)
    {
        createMatches.clear();
        selectMatches.clear();
        return false;
    }

    size_t numMatches = selectMatches.size();
    auto *tarPoints = new cv::Point2f[numMatches];
    auto *qryPoints = new cv::Point2f[numMatches];
    for (size_t i = 0; i < numMatches; ++i)
    {
        const auto &match = selectMatches[i];
        tarPoints[i] = keypointsMarker[match.queryIdx].pt;
        qryPoints[i] = keypointsQuery[match.trainIdx].pt;
    }

    std::vector<cv::Point2f> mappedPointsMarker(tarPoints, tarPoints + numMatches);
    std::vector<cv::Point2f> mappedPointsQuery(qryPoints, qryPoints + numMatches);

    delete[] tarPoints;
    delete[] qryPoints;
    cv::Mat maskPoints;
    // 원근 변환 행렬 계산
    Mat H = findHomography(mappedPointsMarker, mappedPointsQuery, RANSAC, 1.0, maskPoints);

    // 원근 변환 실패시 실패 처리
    if (H.data == nullptr)
    {
        createMatches.clear();
        selectMatches.clear();
        H.release();
        mappedPointsMarker.clear();
        mappedPointsQuery.clear();
        return false;
    }

    // 원근 변환 행렬 역행렬 계산
    Mat M = H.inv();
    M /= M.at<double>(2, 2);

    // 이미지 원근 변환
    imageQueryAligned.release();
    warpPerspective(imageQuery, imageQueryAligned, M, imageQuery.size());

    // Create Mask
    imageMaskQuery.release();
    imageMaskQuery = Mat::ones(imageQuery.size(), CV_8U);

    // Create a mask to remove parts that do not correspond to the image being compared
    imageMarkerMasked.release();
    imageMarkerMasked = imageMarker.clone();
    for (int row = 0; row < imageQueryAligned.rows; row++)
    {
        for (int column = 0; column < imageQueryAligned.cols; column++)
        {
            if (imageQueryAligned.at<uint8_t>(row, column) == 0)
            {
                imageMarkerMasked.at<uint8_t>(row, column) = 0;
                imageMaskQuery.at<uint8_t>(row, column) = 0;
            }
        }
    }

    // ExportImg 사용시 아래 주석 처리
    createMatches.clear();
    selectMatches.clear();
    H.release();
    M.release();
    mappedPointsMarker.clear();
    mappedPointsQuery.clear();
    return true;
}

bool ImageCompare::setMarker(cv::Mat marker)
{
    imageMarker.release();
    imageMarker = marker.clone();
    if (imageMarker.empty())
    {
        logger_e("[ImageCompare] marker image is empty");
        return false;
    }
    keypointsMarker.clear();
    descriptorsMarker.release();
    cvDetector->detectAndCompute(imageMarker, noArray(), keypointsMarker, descriptorsMarker);
    cvMatchers->add(descriptorsMarker);
    marker.release();
    // description markers is empty
    if (descriptorsMarker.empty())
    {
        logger_e("[ImageCompare] marker descriptions is empty");
        return false;
    }
    if (descriptorsMarker.rows < Constants::knn)
    {
        return false;
    }
    return true;
}

bool ImageCompare::setQuery(cv::Mat query)
{
    imageQuery.release();
    imageQuery = query.clone();
    if (imageQuery.empty())
    {
        return false;
    }
    keypointsQuery.clear();
    descriptorsQuery.release();
    cvDetector->detectAndCompute(imageQuery, noArray(), keypointsQuery, descriptorsQuery);
    cvMatchers->clear();
    cvMatchers->add(descriptorsMarker);
    cvMatchers->add(descriptorsQuery);
    query.release();
    if (descriptorsQuery.empty())
    {
        return false;
    }
    if (descriptorsQuery.rows < Constants::knn)
    {
        return false;
    }
    return compare();
}

// The function has been modified to return the coordinates of matched features on the images
const char *ImageCompare::getConfidenceRate()
{
    if (imageMarkerMasked.empty() || imageQueryAligned.empty())
    {
        return "";
    }
    Mat res;
    clahe->apply(imageMarker, imageEqualizedMarker);
    clahe->apply(imageQueryAligned, imageEqualizedQuery);
    matchTemplate(imageEqualizedMarker, imageEqualizedQuery, res,
                  TemplateMatchModes::TM_CCORR_NORMED,
                  imageMaskQuery);

    std::stringstream ss;
    const size_t numMatches = selectMatches.size();
    auto *tarPoints = new cv::Point2f[numMatches];
    auto *qryPoints = new cv::Point2f[numMatches];
    int count = 0;
    for (size_t i = 0; i < numMatches; ++i)
    {
        const auto &match = selectMatches[i];
        tarPoints[i] = keypointsMarker[match.queryIdx].pt;
        qryPoints[i] = keypointsQuery[match.trainIdx].pt;
        char tarX;
        char tarY;
        char qryX;
        char qryY;
        ss << tarPoints[i].x << ":" << tarPoints[i].y << ";" << qryPoints[i].x << ":" << qryPoints[i].y << ";/";
        count++;
    }

    logger_i("The number of matches: %s", numCString);

    std::stringstream rs;
    rs << tarPoints[0].y << ":" << tarPoints[0].x << ";" << qryPoints[0].y << ":" << qryPoints[0].x << ";/"
       << tarPoints[1].y << ":" << tarPoints[1].x << ";" << qryPoints[1].y << ":" << qryPoints[1].x << ";/"
       << tarPoints[2].y << ":" << tarPoints[2].x << ";" << qryPoints[2].y << ":" << qryPoints[2].x << ";/"
       << tarPoints[3].y << ":" << tarPoints[3].x << ";" << qryPoints[3].y << ":" << qryPoints[3].x << ";/";
    std::string rString = rs.str();
    const char *cPointsString = rString.c_str();

    // std::string sPointsString = ss.str();
    // char *cPointsString = new char[sPointsString.length()];
    // strcpy(cPointsString, sPointsString.c_str());
    // cPointsString = (char *)malloc(sizeof sPointsString);

    logger_i("these are the feature coords: %s", cPointsString);

    delete[] tarPoints;
    delete[] qryPoints;

    // std::stringstream confRate;
    // confRate << res.at<float>(0, 0);

    return cPointsString;
}

Mat ImageCompare::getImageMarker()
{
    return imageMarker;
}

Mat ImageCompare::getImageQuery()
{
    return imageQuery;
}