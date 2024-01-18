//
// Created by Junseo Youn on 2023/04/09.
// Modified by confumbit on 2024/01/06.
//

#ifndef PIXELMATCHING_IMAGE_COMPARE_H
#define PIXELMATCHING_IMAGE_COMPARE_H

#include "Constants.h"

typedef cv::Feature2D Detector;
typedef cv::DescriptorMatcher Matcher;

class ImageCompare {

private:
    Mat matrix;
private:
    Ptr<Detector> cvDetector;
    Ptr<Matcher> cvMatchers;
    std::vector<KeyPoint> keypointsMarker, keypointsQuery;
    std::vector<DMatch> selectMatches;
    Mat descriptorsMarker, descriptorsQuery;
    Mat imageQuery, imageQueryAligned, imageMarker, imageMarkerMasked;
    Mat imageMaskMarker, imageMaskQuery;
    Mat maskPoints;
    Ptr<CLAHE> clahe;
    Mat imageEqualizedMarker, imageEqualizedQuery;

    bool compare();

public:
    ImageCompare();

    ~ImageCompare();

public:
    void setDetector(const Ptr<Detector> &detector);

    void setMatchers(const Ptr<Matcher> &matcher);

    bool setMarker(cv::Mat marker);

    bool setQuery(cv::Mat query);

    const char *getConfidenceRate();

    Mat getImageMarker();

    Mat getImageQuery();
};


#endif //PIXELMATCHING_IMAGE_COMPARE_H
