#include "PixelMatching.h"
#include "DebugLogger.h"
#include "ImageProcessor.h"

#if defined(__GNUC__)
// Attributes to prevent 'unused' function from being removed and to make it visible
#define FUNCTION_ATTRIBUTE __attribute__((visibility("default"))) __attribute__((used))
#elif defined(_MSC_VER)
// Marking a function for export
#define FUNCTION_ATTRIBUTE __declspec(dllexport)
#endif

#ifdef __cplusplus
extern "C"
{
#endif

ImageProcessor *processor = nullptr;

FUNCTION_ATTRIBUTE
bool initialize() {
    if (processor == nullptr) {
        processor = new ImageProcessor();
        processor->initialize();
        return true;
    }
    return false;
}

FUNCTION_ATTRIBUTE
int getStateCode() {
    if (processor == nullptr) {
        return -1;
    }
    return processor->getStateCode();
}

FUNCTION_ATTRIBUTE
bool setMarker(unsigned char *bytes, int width, int height, int rotation) {
    if (processor == nullptr) {
        return false;
    }
    Mat image = imdecode(Mat(1, width * height * 3, CV_8UC1, bytes), IMREAD_COLOR);
    if (rotation == 90) {
        transpose(image, image);
        flip(image, image, 1);
    } else if (rotation == 180) {
        transpose(image, image);
        flip(image, image, -1);
    } else if (rotation == 270) {
        transpose(image, image);
        flip(image, image, 0);
    }
    bool res = processor->setMarker(image);
    image.release();

    return res;
}

FUNCTION_ATTRIBUTE
bool setQuery(unsigned char *bytes, int width, int height, int rotation) {
    if (processor == nullptr) {
        return false;
    }
#ifdef __ANDROID__
    Mat image = imdecode(Mat(1, width * height * 3, CV_8UC1, bytes), IMREAD_COLOR);
#elif __APPLE__
    Mat image(height, width, CV_8UC4, bytes);
#endif
    if (rotation == 90) {
        transpose(image, image);
        flip(image, image, 1);
    } else if (rotation == 180) {
        transpose(image, image);
        flip(image, image, -1);
    } else if (rotation == 270) {
        transpose(image, image);
        flip(image, image, 0);
    }

    bool res = processor->setQuery(image);
    image.release();

    return res;
}

FUNCTION_ATTRIBUTE
double getQueryConfidenceRate() {
    if (processor == nullptr) {
        return -1;
    }
    return processor->confidence();
}

FUNCTION_ATTRIBUTE
void dispose() {
    if (processor != nullptr) {
        delete processor;
        processor = nullptr;
    }
}

FUNCTION_ATTRIBUTE
unsigned char *getMarkerQueryDifferenceImage(int *size) {
    logger_i("[PixelMatching] getMarkerQueryDifferenceImage begin");
    logger_i("[PixelMatching] getMarkerQueryDifferenceImage processor %p", processor);
    if (processor == nullptr) {
        auto *res = new unsigned char[0];
        res[0] = 0;
        return res;
    }
    try {
        Mat mkr = processor->getImageMarker();
        Mat qry = processor->getImageQuery();
        if (mkr.empty() || qry.empty()) {
            logger_i("[PixelMatching] getMarkerQueryDifferenceImage empty images");
            auto *res = new unsigned char[0];
            res[0] = 0;
            return res;
        }
        cv::Mat combine;
        cv::hconcat(mkr, qry, combine);
        cv::Mat compare;
        cv::absdiff(mkr, qry, compare);
        cv::threshold(compare, compare, 50, 255, cv::THRESH_BINARY);
        cv::hconcat(combine, compare, combine);

        std::vector<unsigned char> encoded_data;
        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
        compression_params.push_back(90);

        cv::imencode(".jpg", combine, encoded_data, compression_params);
        logger_i("[PixelMatching] getMarkerQueryDifferenceImage encoded_data size %d",
                 encoded_data.size());
        auto *img = new unsigned char[encoded_data.size()];
        memcpy(img, encoded_data.data(), encoded_data.size());
        *size = (int) encoded_data.size();
        logger_i("[PixelMatching] getMarkerQueryDifferenceImage end");

        mkr.release();
        qry.release();
        combine.release();
        compare.release();

        return img;
    } catch (std::exception &e) {
        logger_e("getMarkerQueryDifferenceImage %s", e.what());
        auto *res = new unsigned char[0];
        res[0] = 0;
        return res;
    }
}

#ifdef __cplusplus
}
#endif
