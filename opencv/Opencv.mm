//
//  opencv.mm
//  opencv
//
//  Created by Yusuf Bayindir on 10/18/24.
//

#import <opencv2/opencv.hpp>
#include <string>
#include "Opencv.h"

// 1-Basic Image Operations

cv::Mat loadImage(const std::string &filePath) {
    cv::Mat matImage = cv::imread(filePath);
    return matImage;
}

bool saveImage(const cv::Mat &image, const std::string &filePath) {
    try {
        cv::imwrite(filePath, image);
        return true;
    } catch (const cv::Exception &e) {
        // Hata durumunda false döndür
        return false;
    }
}

void saveImageToGallery(const cv::Mat &image) {
    // C++'ta doğrudan fotoğraf galerisine kaydetme işlemi yapılamaz
    // Burada dönüşüm işlemi yapılacaksa yapılabilir
}

bool saveImageToGallery(const cv::Mat& matImage, const std::string& filePath) {
    try {
        cv::imwrite(filePath, matImage);
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat resizeAndGrayColor(const cv::Mat& inputImage, int width, int height) {
    // Görüntüyü yeniden boyutlandır
    cv::Mat resizedImage;
    cv::resize(inputImage, resizedImage, cv::Size(width, height));
    
    // Görüntüyü gri tona çevir
    cv::Mat grayImage;
    cv::cvtColor(resizedImage, grayImage, cv::COLOR_BGR2GRAY);
    
    return grayImage;  // Gri tonlamaya çevrilmiş ve yeniden boyutlandırılmış görüntüyü döndür
}

cv::Mat makeBorderWithImage(const cv::Mat &inputImage, int top, int bottom, int left, int right, int borderType, const cv::Scalar &borderColor) {
    cv::Mat borderedImage;
    cv::copyMakeBorder(inputImage, borderedImage, top, bottom, left, right, borderType, borderColor);
    return borderedImage;
}

cv::Mat processAndShowImage(const cv::Mat &inputImage) {
    cv::Mat processedImage;
    cv::cvtColor(inputImage, processedImage, cv::COLOR_BGR2GRAY);
    return processedImage;
}

cv::Mat flipImage(const cv::Mat &inputImage, int flipCode) {
    cv::Mat flippedImage;
    cv::flip(inputImage, flippedImage, flipCode);
    return flippedImage;
}

cv::Mat bitwiseAndWithImage1(const cv::Mat &image1, const cv::Mat &image2) {
    cv::Mat dstMat;
    cv::bitwise_and(image1, image2, dstMat);
    return dstMat;
}

cv::Mat bitwiseNotWithImage(const cv::Mat &inputImage) {
    cv::Mat dstMat;
    cv::bitwise_not(inputImage, dstMat);
    return dstMat;
}

cv::Mat addWeightedWithImage1(const cv::Mat &image1, const cv::Mat &image2, double alpha, double beta, double gamma) {
    cv::Mat dstMat;
    cv::addWeighted(image1, alpha, image2, beta, gamma, dstMat);
    return dstMat;
}

std::vector<cv::Mat> splitImage(const cv::Mat &inputImage) {
    std::vector<cv::Mat> channels;
    cv::split(inputImage, channels);
    return channels;
}

cv::Mat mergeWithChannel1(const cv::Mat &channel1, const cv::Mat &channel2, const cv::Mat &channel3) {
    std::vector<cv::Mat> channels = {channel1, channel2, channel3};
    cv::Mat mergedMat;
    cv::merge(channels, mergedMat);
    return mergedMat;
}

// 2-Geometric Transformations

cv::Mat rotateImage(const cv::Mat &inputImage, const cv::Point2f &center, double angle, double scale) {
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, scale);
    cv::Mat rotatedImage;
    cv::warpAffine(inputImage, rotatedImage, rotationMatrix, inputImage.size());
    return rotatedImage;
}

cv::Mat applyWarpAffineToImage(const cv::Mat &inputImage, const cv::Mat &warpMat) {
    cv::Mat warpedImage;
    cv::warpAffine(inputImage, warpedImage, warpMat, inputImage.size());
    return warpedImage;
}

cv::Mat warpPerspectiveImage(const cv::Mat &inputImage, const std::vector<cv::Point2f> &srcPoints, const std::vector<cv::Point2f> &dstPoints) {
    cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(srcPoints.data(), dstPoints.data());
    cv::Mat warpedImage;
    cv::warpPerspective(inputImage, warpedImage, perspectiveMatrix, inputImage.size());
    return warpedImage;
}

std::vector<double> getAffineTransformWithSourcePoints(const std::vector<cv::Point2f>& srcPoints, const std::vector<cv::Point2f>& dstPoints) {
    if (srcPoints.size() != 3 || dstPoints.size() != 3) {
        throw std::invalid_argument("Both source and destination points must have exactly 3 points.");
    }
    
    // Affine dönüşüm matrisini hesapla
    cv::Mat affineMatrix = cv::getAffineTransform(srcPoints, dstPoints);

    // Dönüşüm matrisini 1D diziye düzleştir
    std::vector<double> matrixArray;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            matrixArray.push_back(affineMatrix.at<double>(i, j));
        }
    }
    
    return matrixArray;
}

cv::Mat applyPerspectiveTransform(const cv::Mat& inputImage, const std::vector<cv::Point2f>& srcPoints, const std::vector<cv::Point2f>& dstPoints) {
    if (srcPoints.size() != 4 || dstPoints.size() != 4) {
        throw std::invalid_argument("Both source and destination points must have exactly 4 points.");
    }
    
    // Perspektif dönüşüm matrisini hesapla
    cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(srcPoints.data(), dstPoints.data());

    // Dönüşümü uygula
    cv::Mat transformedImage;
    cv::warpPerspective(inputImage, transformedImage, perspectiveMatrix, inputImage.size());
    
    return transformedImage;
}

cv::Mat remapImage(const cv::Mat& image, const cv::Mat& mapX, const cv::Mat& mapY, int interpolation) {
    cv::Mat remappedImage;
    cv::remap(image, remappedImage, mapX, mapY, interpolation);
    return remappedImage;
}

cv::Mat transposeImage(const cv::Mat& inputImage) {
    cv::Mat transposedImage;
    cv::transpose(inputImage, transposedImage);
    return transposedImage;
}

cv::Mat pyrUpWithImage(const cv::Mat& inputImage) {
    cv::Mat outputImage;
    cv::pyrUp(inputImage, outputImage);
    return outputImage;
}

cv::Mat pyrDownWithImage(const cv::Mat& inputImage) {
    cv::Mat outputImage;
    cv::pyrDown(inputImage, outputImage);
    return outputImage;
}

void resizeWindowWithName(const std::string& windowName, int width, int height) {
    cv::resizeWindow(windowName, width, height);
}

// 3-Drawing Functions

void drawLineOnImage(cv::Mat& matImage, cv::Point start, cv::Point end, const cv::Scalar& color, int thickness) {
    cv::line(matImage, start, end, color, thickness);
}

void drawCircleOnImage(cv::Mat& matImage, cv::Point center, int radius, const cv::Scalar& color, int lineWidth) {
    cv::circle(matImage, center, radius, color, lineWidth, cv::LINE_AA);
}

void drawRectangleOnImage(cv::Mat& matImage, const cv::Point& topLeft, const cv::Point& bottomRight, const cv::Scalar& color, int lineWidth) {
    cv::rectangle(matImage, topLeft, bottomRight, color, lineWidth, cv::LINE_AA);
}

void drawEllipseOnImage(cv::Mat& matImage, const cv::Point& center, const cv::Size& axes, double angle, double startAngle, double endAngle, const cv::Scalar& color, int thickness) {
    cv::ellipse(matImage, center, axes, angle, startAngle, endAngle, color, thickness);
}

cv::Mat addTextToImage(const cv::Mat& image, const std::string& text, cv::Point position, int fontFace, double fontScale, cv::Scalar color, int thickness, int lineType) {
    // Görüntünün bir kopyasını oluştur
    cv::Mat newImage = image.clone();
    
    // Yazıyı kopyalanan görüntüye ekle
    cv::putText(newImage, text, position, fontFace, fontScale, color, thickness, lineType);
    
    // Yeni görüntüyü döndür
    return newImage;
}

cv::Mat fillPolygonOnImage(const cv::Mat& image, const std::vector<cv::Point>& points, const cv::Scalar& color) {
    cv::Mat result = image.clone(); // Orijinal resmi kopyala

    // Poligon doldurma
    cv::fillConvexPoly(result, points, color);

    return result; // Doldurulmuş resmi döndür
}

cv::Mat drawPolylinesOnImage(const cv::Mat& image, const std::vector<cv::Point>& points, const cv::Scalar& color, int lineWidth) {
    cv::Mat result = image.clone(); // Orijinal resmi kopyala

    // Poligon çizme
    const bool isClosed = false; // Kapalı değil
    cv::polylines(result, points, isClosed, color, lineWidth, cv::LINE_AA);

    return result; // Çizilmiş resmi döndür
}

// 4-Thresholding and Edge Detection

cv::Mat applyThresholdToImage(const cv::Mat& inputImage, double threshold, double maxValue, int thresholdType) {
    // Girişi gri tonlamaya çevirin
    cv::Mat grayImage;
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    
    // Eşikleme işlemini uygulayın
    cv::Mat thresholdedImage;
    cv::threshold(grayImage, thresholdedImage, threshold, maxValue, thresholdType);
    
    return thresholdedImage;
}

cv::Mat applyAdaptiveThresholdToImage(const cv::Mat& inputImage, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C) {
    // Girişi gri tonlamaya çevirin
    cv::Mat grayImage;
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    
    // Adaptif eşikleme işlemini uygulayın
    cv::Mat thresholdedImage;
    cv::adaptiveThreshold(grayImage, thresholdedImage, maxValue, adaptiveMethod, thresholdType, blockSize, C);
    
    return thresholdedImage;
}

cv::Mat applyCannyToImage(const cv::Mat& inputImage, double threshold1, double threshold2) {
    // Girişi gri tonlamaya çevirin
    cv::Mat grayImage;
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    
    // Canny kenar algılama işlemini uygulayın
    cv::Mat edges;
    cv::Canny(grayImage, edges, threshold1, threshold2);
    
    return edges;
}

cv::Mat applySobelFilter(const cv::Mat& image, int dx, int dy, int kernelSize) {
    if (image.empty()) {
        NSLog(@"Error: Input image is empty");
        return cv::Mat();
    }

    cv::Mat grayImage, gradImage;

    // Eğer görüntü renkli ise gri tonlamaya çevir
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    // Sobel filtresini uygula
    cv::Sobel(grayImage, gradImage, CV_64F, dx, dy, kernelSize);

    // CV_64F -> CV_8UC1 dönüştür
    cv::Mat absGradImage;
    cv::convertScaleAbs(gradImage, absGradImage);

    if (absGradImage.empty()) {
        NSLog(@"Error: Sobel filtered image is empty");
        return cv::Mat();
    }

    // Sonucu RGBA formatına çevir
    cv::Mat resultImage;
    cv::cvtColor(absGradImage, resultImage, cv::COLOR_GRAY2RGBA);

    return resultImage;
}

cv::Mat laplacianWithImage(const cv::Mat& inputImage, int kernelSize) {
    cv::Mat dstMat;
    cv::Laplacian(inputImage, dstMat, CV_16S, kernelSize);
    
    // CV_16S tipindeki görüntüyü CV_8U tipine dönüştürme
    cv::Mat absDstMat;
    cv::convertScaleAbs(dstMat, absDstMat);
    
    return absDstMat;
}

cv::Mat inRangeWithImage(const cv::Mat &inputMat, const cv::Scalar &lower, const cv::Scalar &upper) {
    cv::Mat mask;
    cv::inRange(inputMat, lower, upper, mask);
    return mask;
}


std::vector<cv::Point> findNonZeroWithImage(const cv::Mat &srcMat) {
    std::vector<cv::Point> points;
    cv::findNonZero(srcMat, points); // Sıfır olmayan noktaları tespit ediyor
    return points;
}

cv::Mat gaussianBlur(const cv::Mat& srcMat, const cv::Size& kernelSize, double sigma) {
    cv::Mat dstMat;

    // GaussianBlur fonksiyonunu uygulama
    cv::GaussianBlur(srcMat, dstMat, kernelSize, sigma);

    return dstMat;
}

cv::Mat medianBlur(const cv::Mat& srcMat, int kernelSize) {
    cv::Mat dstMat;

    // MedianBlur fonksiyonunu uygulama
    cv::medianBlur(srcMat, dstMat, kernelSize);

    return dstMat;
}

cv::Mat blur(const cv::Mat& srcMat, const cv::Size& kernelSize) {
    cv::Mat dstMat;

    // Blur fonksiyonunu uygulama
    cv::blur(srcMat, dstMat, kernelSize);

    return dstMat;
}

cv::Mat applyBilateralFilterToImage(const cv::Mat& srcMat, int diameter, double sigmaColor, double sigmaSpace) {
    cv::Mat filteredImage;

    // Bilateral filtre uygulama
    cv::bilateralFilter(srcMat, filteredImage, diameter, sigmaColor, sigmaSpace);

    return filteredImage;
}

cv::Mat applyFilter2DToImage(const cv::Mat& srcMat, const cv::Mat& kernelMat) {
    cv::Mat filteredImage;

    // 2D filtre uygulama
    cv::filter2D(srcMat, filteredImage, -1, kernelMat);

    return filteredImage;
}

cv::Mat applyBoxFilterToImage(const cv::Mat& srcMat, int ddepth, const cv::Size& ksize) {
    cv::Mat filteredImage;

    // Kutu filtre uygulama
    cv::boxFilter(srcMat, filteredImage, ddepth, ksize);

    return filteredImage;
}

cv::Mat applyScharrOnImage(const cv::Mat &image) {
    cv::Mat scharrX, scharrY, scharrResult;

    // X ve Y yönünde Scharr gradyanı hesapla
    cv::Scharr(image, scharrX, CV_16S, 1, 0);
    cv::Scharr(image, scharrY, CV_16S, 0, 1);

    // Gradyanları birleştir
    cv::convertScaleAbs(scharrX, scharrX);
    cv::convertScaleAbs(scharrY, scharrY);
    cv::addWeighted(scharrX, 0.5, scharrY, 0.5, 0, scharrResult);

    return scharrResult;
}

cv::Mat addImage(const cv::Mat& image1, const cv::Mat& image2) {
    cv::Mat result;
    cv::add(image1, image2, result);
    return result;
}

cv::Mat subtractImage(const cv::Mat& image1, const cv::Mat& image2) {
    cv::Mat result;
    cv::subtract(image1, image2, result);
    return result;
}

cv::Mat multiplyImage(const cv::Mat& image1, const cv::Mat& image2) {
    cv::Mat result;
    cv::multiply(image1, image2, result);
    return result;
}

cv::Mat divideImage(const cv::Mat& image1, const cv::Mat& image2) {
    cv::Mat result;
    cv::divide(image1, image2, result);
    return result;
}

// 6-Morphological Operations

cv::Mat erodeImage(const cv::Mat& image, int kernelSize) {
    cv::Mat result;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::erode(image, result, kernel);
    return result;
}

cv::Mat dilateImage(const cv::Mat& image, int kernelSize) {
    cv::Mat result;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::dilate(image, result, kernel);
    return result;
}

cv::Mat applyMorphologyEx(const cv::Mat& image, int operation, int kernelSize) {
    cv::Mat result;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::morphologyEx(image, result, operation, kernel);
    return result;
}

cv::Mat getStructuringElementWithType(int type, int kernelSize) {
    return cv::getStructuringElement(type, cv::Size(kernelSize, kernelSize));
}

// 7-Image Contours and Shape Analysis

std::vector<std::vector<cv::Point>> findContoursInImage(const cv::Mat &image) {
    cv::Mat grayImage, edgeImage;

    // Görüntüyü gri tonlamaya dönüştür ve kenarlarını belirginleştir
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::Canny(grayImage, edgeImage, 100, 200);

    // Konturları bul
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edgeImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    return contours;
}

void drawContoursOnImage(cv::Mat &image, const std::vector<std::vector<cv::Point>> &contours, const cv::Scalar &color, int thickness) {
    cv::drawContours(image, contours, -1, color, thickness);
}

double arcLengthOfContour(const std::vector<cv::Point> &contour, bool isClosed) {
    return cv::arcLength(contour, isClosed);
}

double contourAreaOfContour(const std::vector<cv::Point> &contour) {
    return cv::contourArea(contour);
}

void approxPolyDPOfContour(const std::vector<cv::Point> &contour, std::vector<cv::Point> &approxContour, double epsilon, bool isClosed) {
    cv::approxPolyDP(contour, approxContour, epsilon, isClosed);
}

void convexHullOfContour(const std::vector<cv::Point> &contour, std::vector<cv::Point> &hull) {
    cv::convexHull(contour, hull);
}

bool isContourConvex(const std::vector<cv::Point> &contour) {
    return cv::isContourConvex(contour);
}

cv::Rect boundingRectOfContour(const std::vector<cv::Point> &contour) {
    return cv::boundingRect(contour);
}

cv::RotatedRect minAreaRectOfContour(const std::vector<cv::Point> &contour) {
    return cv::minAreaRect(contour);
}

cv::RotatedRect fitEllipseOfContour(const std::vector<cv::Point> &contour) {
    return cv::fitEllipse(contour);
}

cv::Vec4f fitLineOfContour(const std::vector<cv::Point>& contour) {
    cv::Vec4f line;
    cv::fitLine(contour, line, cv::DIST_L2, 0, 0.01, 0.01);
    return line;
}

std::vector<cv::Point2f> goodFeaturesToTrackInImage(const cv::Mat& grayImage, int maxCorners, double qualityLevel, double minDistance) {
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(grayImage, corners, maxCorners, qualityLevel, minDistance);
    return corners;
}

std::vector<cv::Vec2f> houghLinesInImage(const cv::Mat& edges, double rho, double theta, int threshold) {
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(edges, lines, rho, theta, threshold);
    return lines;
}

std::vector<cv::Vec3f> houghCirclesInImage(const cv::Mat& gray, double dp, double minDist, double param1, double param2, int minRadius, int maxRadius) {
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, dp, minDist, param1, param2, minRadius, maxRadius);
    return circles;
}

cv::Mat cornerHarrisInImage(const cv::Mat& gray, int blockSize, int ksize, double k) {
    cv::Mat dst;
    cv::cornerHarris(gray, dst, blockSize, ksize, k);
    
    // Normalizasyon
    cv::Mat dstNorm;
    cv::normalize(dst, dstNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    
    return dstNorm;
}

std::vector<cv::KeyPoint> detectORBKeypointsInImage(const cv::Mat& gray, int nFeatures, cv::Mat& descriptors) {
    // ORB dedektörü oluştur
    cv::Ptr<cv::ORB> orb = cv::ORB::create(nFeatures);
    
    // Anahtar noktaları tespit et
    std::vector<cv::KeyPoint> keypoints;
    orb->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
    
    return keypoints;
}

std::vector<cv::KeyPoint> detectSIFTKeypointsInImage(const cv::Mat& gray, int nFeatures, cv::Mat& descriptors) {
    // SIFT dedektörü oluştur
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(nFeatures);
    
    // Anahtar noktaları tespit et
    std::vector<cv::KeyPoint> keypoints;
    sift->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
    
    return keypoints;
}

std::vector<cv::DMatch> matchKeypointsWithBFMatcherDescriptors1(const cv::Mat& descriptors1, const cv::Mat& descriptors2) {
    // BFMatcher oluştur
    cv::BFMatcher matcher(cv::NORM_L2, true);
    
    // Eşleşmeleri bul
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    
    return matches;
}

std::vector<cv::DMatch> matchKeypointsWithFlannMatcherDescriptors1(const cv::Mat& descriptors1, const cv::Mat& descriptors2) {
    // FLANN tabanlı eşleştirici oluştur
    cv::FlannBasedMatcher matcher;

    // Eşleşmeleri bul
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    
    return matches;
}

cv::Mat drawKeypointsOnImage(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints) {
    cv::Mat outputImage;
    cv::drawKeypoints(image, keypoints, outputImage, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return outputImage;
}

cv::Mat matchTemplateInImage(const cv::Mat& image, const cv::Mat& templateImage, cv::Point& matchLocation) {
    // Sonuç matrisini oluştur
    cv::Mat result;
    cv::matchTemplate(image, templateImage, result, cv::TM_CCOEFF_NORMED);
    
    // En iyi eşleşmenin konumunu bul
    double minVal, maxVal;
    cv::Point minLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &matchLocation);
    
    return result; // Elde edilen sonuç matrisini döndür
}

// 9-Optical Flow

cv::Mat calculateOpticalFlowFromImage(const cv::Mat& prevImage, const cv::Mat& nextImage) {
    // Gri tonlamaya çevir
    cv::Mat grayPrev, grayNext;
    cv::cvtColor(prevImage, grayPrev, cv::COLOR_BGR2GRAY);
    cv::cvtColor(nextImage, grayNext, cv::COLOR_BGR2GRAY);
    
    // Optik akışı hesapla
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(grayPrev, grayNext, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    
    // Akış vektörlerini görüntüye çiz
    cv::Mat outputImage = nextImage.clone();
    for (int y = 0; y < outputImage.rows; y += 5) {
        for (int x = 0; x < outputImage.cols; x += 5) {
            const cv::Point2f flowAtPoint = flow.at<cv::Point2f>(y, x);
            cv::line(outputImage, cv::Point(x, y), cv::Point(cvRound(x + flowAtPoint.x), cvRound(y + flowAtPoint.y)), cv::Scalar(0, 255, 0), 1);
            cv::circle(outputImage, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
        }
    }
    
    return outputImage; // Çizilmiş görüntüyü döndür
}

cv::Mat calculateOpticalFlowPyrLKFromImage(const cv::Mat& prevImage, const cv::Mat& nextImage, const std::vector<cv::Point2f>& points) {
    // Gri tonlamaya çevir
    cv::Mat grayPrev, grayNext;
    cv::cvtColor(prevImage, grayPrev, cv::COLOR_BGR2GRAY);
    cv::cvtColor(nextImage, grayNext, cv::COLOR_BGR2GRAY);
    
    // Optik akışı hesapla
    std::vector<cv::Point2f> nextPoints;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(grayPrev, grayNext, points, nextPoints, status, err);
    
    // Akış vektörlerini görüntüye çiz
    cv::Mat outputImage = nextImage.clone();
    for (size_t i = 0; i < points.size(); i++) {
        if (status[i]) {
            cv::line(outputImage, points[i], nextPoints[i], cv::Scalar(0, 255, 0), 1);
            cv::circle(outputImage, nextPoints[i], 2, cv::Scalar(0, 0, 255), -1);
        }
    }
    
    return outputImage; // Çizilmiş görüntüyü döndür
}

cv::Mat calculateMotionGradientFromImage(const cv::Mat& image) {
    // Gri tonlamaya çevir
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    
    // Hareket gradyanı hesapla
    cv::Mat flow, magnitude;
    cv::calcOpticalFlowFarneback(gray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    
    // Akışın büyüklüğünü hesapla
    cv::magnitude(flow(cv::Rect(0, 0, flow.cols, flow.rows)).at<float>(0,0),
                  flow(cv::Rect(0, 0, flow.cols, flow.rows)).at<float>(1,0),
                  magnitude);
    
    // Sonuç görüntüsünü normalize et
    cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX);
    cv::Mat outputImage;
    magnitude.convertTo(outputImage, CV_8U);
    
    return outputImage; // Normalize edilmiş görüntüyü döndür
}

double calculateGlobalOrientationFromImage(const cv::Mat& image) {
    // Gri tonlamaya çevir
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    
    // Hareket gradyanı hesapla
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(gray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    
    // Akışın yönünü hesapla
    cv::Mat orientation;
    cv::phase(flow(cv::Rect(0, 0, flow.cols, flow.rows)),
              flow(cv::Rect(0, 0, flow.cols, flow.rows)),
              orientation, true);
    
    // Ortalama yönelimi hesapla
    cv::Scalar meanOrientation = cv::mean(orientation);
    
    return meanOrientation[0]; // Global yönelimi döndür
}

// 10-Camera Calibration and 3D Vision

bool findChessboardCornersInImage(const cv::Mat& image, const cv::Size& boardSize, std::vector<cv::Point2f>& corners) {
    // Gri tonlamaya çevir
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    
    // Satranç tahtası köşelerini bul
    bool found = cv::findChessboardCorners(gray, boardSize, corners);
    
    // Köşeleri doğrulama
    if (found) {
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
    }
    
    return found; // Bulunduysa true, bulunmadıysa false döndür
}

double calibrateCameraWithObjectPoints(const std::vector<std::vector<cv::Point3f>>& objPoints,
                                  const std::vector<std::vector<cv::Point2f>>& imgPoints,
                                  cv::Size imgSize,
                                  cv::Mat& cameraMatrix,
                                  cv::Mat& distCoeffs,
                                  std::vector<cv::Mat>& rvecs,
                                  std::vector<cv::Mat>& tvecs) {
    
    // Kalibrasyon parametrelerini hesapla
    double rms = cv::calibrateCamera(objPoints, imgPoints, imgSize, cameraMatrix, distCoeffs, rvecs, tvecs);
    return rms;
}

void undistortImage(const cv::Mat& inputImage, cv::Mat& outputImage,
                    const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs) {
    // Görüntüyü düzelt
    cv::undistort(inputImage, outputImage, cameraMatrix, distCoeffs);
}

bool solvePnPWithObjectPoints(const std::vector<cv::Point3f>& objPoints,
              const std::vector<cv::Point2f>& imgPoints,
              const cv::Mat& camMatrix,
              const cv::Mat& distCoeffs,
              cv::Mat& rvec,
              cv::Mat& tvec) {
    return cv::solvePnP(objPoints, imgPoints, camMatrix, distCoeffs, rvec, tvec);
}

void projectPointsWithObjectPoints(const std::vector<cv::Point3f>& objPoints,
                   const cv::Mat& rvec,
                   const cv::Mat& tvec,
                   const cv::Mat& camMatrix,
                   const cv::Mat& distCoeffs,
                   std::vector<cv::Point2f>& imgPoints) {
    cv::projectPoints(objPoints, rvec, tvec, camMatrix, distCoeffs, imgPoints);
}

std::vector<std::tuple<cv::Mat, cv::Mat, cv::Mat>> decomposeHomography(const cv::Mat &homography) {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F); // Genellikle K birim matristir
    std::vector<cv::Mat> rotations, translations, normals;
    
    cv::decomposeHomographyMat(homography, K, rotations, translations, normals);

    std::vector<std::tuple<cv::Mat, cv::Mat, cv::Mat>> results;
    for (size_t i = 0; i < rotations.size(); i++) {
        results.emplace_back(rotations[i], translations[i], normals[i]);
    }
    return results;
}

cv::Mat findEssentialMatrixWithPoints1(const std::vector<cv::Point2f>& pts1,
                                const std::vector<cv::Point2f>& pts2,
                                const cv::Mat& camMatrix) {
    return cv::findEssentialMat(pts1, pts2, camMatrix, cv::RANSAC);
}

std::tuple<cv::Mat, cv::Mat, cv::Mat> decomposeEssentialMatWithEssentialMat(const cv::Mat& essentialMat) {
    cv::Mat R1, R2, t;
    cv::decomposeEssentialMat(essentialMat, R1, R2, t);
    return std::make_tuple(R1, R2, t);
}

// 11-Video Processing

cv::Mat captureFrameFromCameraIndex(int cameraIndex) {
    cv::VideoCapture videoCapture(cameraIndex);  // VideoCapture nesnesini oluştur

    if (!videoCapture.isOpened()) {  // Kamera açık mı kontrol et
        return cv::Mat(); // Boş bir mat döner
    }

    cv::Mat frame;
    if (videoCapture.read(frame)) {  // Çerçeveyi oku
        return frame;
    }
    
    return cv::Mat(); // Eğer çerçeve okunamazsa boş bir mat döner
}

bool writeVideoFromImages(const std::vector<cv::Mat>& images, const std::string& filePath, int fps) {
    if (images.empty()) {
        std::cout << "No images to write." << std::endl;
        return false;
    }

    cv::Size frameSize(images[0].cols, images[0].rows);
    cv::VideoWriter videoWriter(filePath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frameSize, true);

    if (!videoWriter.isOpened()) {
        std::cout << "Failed to open video writer." << std::endl;
        return false;
    }

    for (const auto& matImage : images) {
        videoWriter.write(matImage);
    }
    
    videoWriter.release();
    return true;
}


#import "Opencv.h"
#import <Foundation/Foundation.h>
#import <vector>
#import <opencv2/opencv.hpp>

@implementation Opencv

// 1-Basic Image Operations

+ (UIImage *)loadImage:(NSString *)filePath {
    cv::Mat matImage = cv::imread([filePath UTF8String]);
    if (matImage.empty()) {
        return nil;
    }
    // C++ fonksiyonunu çağırarak cv::Mat formatında görüntüyü yükle
    cv::Mat matToUIImage = loadImage([filePath UTF8String]);
    return [self MatToUIImage:matImage];
}

+ (BOOL)saveImage:(UIImage *)image toFilePath:(NSString *)filePath {
    // UIImage'i cv::Mat formatına çevir
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];
    
    // C++ fonksiyonunu çağırarak resmi belirtilen dosya yoluna kaydet
    return saveImage(matImage, [filePath UTF8String]);
}

+ (void)saveImageToGallery:(UIImage *)image {
    // C++ işlemi olmadığı için doğrudan iOS API'sini çağırıyoruz
    UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil);
}

+ (UIImage *)processAndShowImage:(UIImage *)image {
    // UIImage'i cv::Mat formatına çevir
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];
    
    // C++ işlem fonksiyonunu çağırarak görüntüyü gri tonlamaya çevir
    cv::Mat processedImage = processAndShowImage(matImage);
    
    // İşlenmiş cv::Mat görüntüyü UIImage formatına çevir ve döndür
    return [self MatToUIImage:processedImage];
}

+ (UIImage *)resizeAndGrayColor:(UIImage *)image
                         toSize:(CGSize)size {
    cv::Mat cvImage;
    [self UIImageToMat:image mat:cvImage];
    
    cv::Mat resizedImage;
    cv::resize(cvImage, resizedImage, cv::Size(size.width, size.height));
    
    cv::Mat convertedImage;
    cv::cvtColor(resizedImage, convertedImage, cv::COLOR_BGR2GRAY);
    
    // C++ fonksiyonunu çağırarak işleme gönder
    cv::Mat processedImage = resizeAndGrayColor(cvImage, static_cast<int>(size.width), static_cast<int>(size.height));
    
    UIImage *resultImage = [self MatToUIImage:convertedImage];
    return resultImage;
}

+ (UIImage *)makeBorderWithImage:(UIImage *)image top:(int)top bottom:(int)bottom left:(int)left right:(int)right borderType:(int)borderType color:(UIColor *)color {
    // UIImage to cv::Mat
    cv::Mat cvImage;
    [self UIImageToMat:image mat:cvImage];
    
    // Convert UIColor to cv::Scalar
    CGFloat red, green, blue, alpha;
    [color getRed:&red green:&green blue:&blue alpha:&alpha];
    cv::Scalar borderColor = cv::Scalar(blue * 255, green * 255, red * 255, alpha * 255);
    
    // C++ fonksiyonunu çağırarak border ekle
    cv::Mat borderedImage = makeBorderWithImage(cvImage, top, bottom, left, right, borderType, borderColor);
    
    // cv::Mat to UIImage
    return [self MatToUIImage:borderedImage];
}

+ (UIImage *)flipImage:(UIImage *)image flipCode:(int)flipCode {
    //    0: X ekseninde çevirme (dikey olarak çevirme)
    //    1: Y ekseninde çevirme (yatay olarak çevirme)
    //    -1: Hem X hem de Y ekseninde çevirme
    
    // UIImage'ı cv::Mat'e çevir
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];
    
    // C++ fonksiyonunu çağırarak görüntüyü çevir
    cv::Mat flippedImage = flipImage(matImage, flipCode);
    
    // cv::Mat'i UIImage'e çevir
    return [self MatToUIImage:flippedImage];
}

+ (UIImage *)bitwiseAndWithImage1:(UIImage *)image1 image2:(UIImage *)image2 {
    // UIImage'ları cv::Mat'e dönüştürme
    cv::Mat mat1, mat2;
    [self UIImageToMat:image1 mat:mat1];
    [self UIImageToMat:image2 mat:mat2];
    
    // İki görüntünün aynı boyutta olup olmadığını kontrol edin
    if (mat1.size() != mat2.size()) {
        NSLog(@"Error: Images must be of the same size for bitwise operations.");
        return nil;
    }
    
    // C++ fonksiyonunu çağırarak bitwise AND işlemi yap
    cv::Mat dstMat = bitwiseAndWithImage1(mat1, mat2);
    
    // Sonucu UIImage'a dönüştürme
    return [self MatToUIImage:dstMat];
}

+ (UIImage *)bitwiseNotWithImage:(UIImage *)image {
    // UIImage'ı cv::Mat'e dönüştür
    cv::Mat srcMat;
    [self UIImageToMat:image mat:srcMat];
    
    // C++ fonksiyonunu çağırarak bitwise NOT işlemi yap
    cv::Mat dstMat = bitwiseNotWithImage(srcMat);
    
    // Sonucu UIImage'a dönüştürme
    return [self MatToUIImage:dstMat];
}

+ (UIImage *)addWeightedWithImage1:(UIImage *)image1
                            image2:(UIImage *)image2
                             alpha:(double)alpha
                              beta:(double)beta
                             gamma:(double)gamma {
    cv::Mat mat1, mat2;
    [self UIImageToMat:image1 mat:mat1];
    [self UIImageToMat:image2 mat:mat2];
    
    // İki görüntünün aynı boyutta olup olmadığını kontrol edin
    if (mat1.size() != mat2.size()) {
        NSLog(@"Error: Images must be of the same size for weighted addition.");
        return nil;
    }
    
    // C++ fonksiyonunu çağırarak ağırlıklı toplama işlemi yap
    cv::Mat dstMat = addWeightedWithImage1(mat1, mat2, alpha, beta, gamma);
    
    // Sonucu UIImage'a dönüştürme
    return [self MatToUIImage:dstMat];
}

+ (NSArray<UIImage *> *)splitImage:(UIImage *)image {
    // UIImage'den cv::Mat'e dönüşüm
    cv::Mat mat = [self cvMatFromUIImage:image];
    
    // C++ fonksiyonunu çağırarak görüntüyü bileşenlerine ayır
    std::vector<cv::Mat> channels = splitImage(mat);
    
    // cv::Mat'leri UIImage'ye çevirme
    NSMutableArray<UIImage *> *resultImages = [NSMutableArray array];
    for (const auto& channel : channels) {
        [resultImages addObject:[self MatToUIImage:channel]];
    }
    
    return resultImages;
}

+ (UIImage *)mergeWithChannel1:(UIImage *)channel1
                      channel2:(UIImage *)channel2
                      channel3:(UIImage *)channel3 {
    // Her kanalı cv::Mat'e dönüştürme
    cv::Mat mat1, mat2, mat3;
    [self UIImageToMat:channel1 mat:mat1];
    [self UIImageToMat:channel2 mat:mat2];
    [self UIImageToMat:channel3 mat:mat3];
    
    // Her kanalın aynı boyutta olup olmadığını kontrol et
    if (mat1.size() != mat2.size() || mat2.size() != mat3.size()) {
        NSLog(@"Error: All channels must be of the same size for merging.");
        return nil;
    }
    
    // C++ fonksiyonunu çağırarak kanalları birleştir
    cv::Mat mergedMat = mergeWithChannel1(mat1, mat2, mat3);
    
    // Sonucu UIImage'a dönüştür
    return [self MatToUIImage:mergedMat];
}

// 2-Geometric Transformations

+ (UIImage *)rotateImage:(UIImage *)image center:(CGPoint)center angle:(double)angle scale:(double)scale {
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];
    
    // C++ fonksiyonunu çağırarak görüntüyü döndür
    cv::Point2f cvCenter(center.x, center.y);
    cv::Mat rotatedImage = rotateImage(matImage, cvCenter, angle, scale);
    
    // Sonucu UIImage'a dönüştür
    return [self MatToUIImage:rotatedImage];
}

+ (UIImage *)applyWarpAffineToImage:(UIImage *)image matrix:(NSArray<NSNumber *> *)matrix {
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];
    
    // Matristen dönüşüm matrisini oluştur
    cv::Mat warpMat = (cv::Mat_<double>(2, 3) <<
                       matrix[0].doubleValue, matrix[1].doubleValue, matrix[2].doubleValue,
                       matrix[3].doubleValue, matrix[4].doubleValue, matrix[5].doubleValue
                       );
    
    // C++ fonksiyonunu çağırarak warp işlemini uygula
    cv::Mat warpedImage = applyWarpAffineToImage(matImage, warpMat);
    
    // Sonucu UIImage'a dönüştür
    return [self MatToUIImage:warpedImage];
}

+ (UIImage *)warpPerspectiveImage:(UIImage *)image
                        srcPoints:(NSArray<NSValue *> *)srcPoints
                        dstPoints:(NSArray<NSValue *> *)dstPoints {
    
    // UIImage'den cv::Mat'e dönüşüm
    cv::Mat mat = [self UIImageToMat:image];
    
    // Kaynak ve hedef noktalarını cv::Point2f dizisine çevir
    std::vector<cv::Point2f> src(4), dst(4);
    
    for (int i = 0; i < 4; i++) {
        CGPoint point = [srcPoints[i] CGPointValue];
        src[i] = cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y)); // Dönüşüm burada
    }
    
    for (int i = 0; i < 4; i++) {
        CGPoint point = [dstPoints[i] CGPointValue];
        dst[i] = cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y)); // Dönüşüm burada
    }
    
    // C++ fonksiyonunu çağırarak perspektif dönüşüm uygula
    cv::Mat warpedImage = warpPerspectiveImage(mat, src, dst);
    
    // Sonucu UIImage'a dönüştür
    return [self MatToUIImage:warpedImage];
}

+ (nullable NSArray<NSNumber *> *)getAffineTransformWithSourcePoints:(NSArray<NSValue *> *)sourcePoints
                                                   destinationPoints:(NSArray<NSValue *> *)destinationPoints {
    if (sourcePoints.count != 3 || destinationPoints.count != 3) {
        NSLog(@"Error: Both source and destination points must have exactly 3 points.");
        return nil;
    }
    
    // Kaynak ve hedef noktaları cv::Point2f dizisine çevir
    std::vector<cv::Point2f> srcPoints, dstPoints;
    for (int i = 0; i < 3; i++) {
        CGPoint srcCGPoint = [sourcePoints[i] CGPointValue];
        CGPoint dstCGPoint = [destinationPoints[i] CGPointValue];
        
        srcPoints.emplace_back(srcCGPoint.x, srcCGPoint.y);
        dstPoints.emplace_back(dstCGPoint.x, dstCGPoint.y);
    }
    
    // C++ fonksiyonunu çağır
    std::vector<double> matrixArray;
    try {
        matrixArray = getAffineTransformWithSourcePoints(srcPoints, dstPoints);
    } catch (const std::invalid_argument& e) {
        NSLog(@"%s", e.what());
        return nil;
    }
    
    // Dönüşüm matrisini NSArray<NSNumber *> formatına çevir
    NSMutableArray<NSNumber *> *resultArray = [NSMutableArray array];
    for (double value : matrixArray) {
        [resultArray addObject:@(value)];
    }
    
    return resultArray;
}

+ (UIImage *)applyPerspectiveTransform:(UIImage *)image srcPoints:(NSArray<NSValue *> *)srcPoints dstPoints:(NSArray<NSValue *> *)dstPoints {
    // UIImage'den cv::Mat'e dönüşüm
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];
    
    // Dönüşüm noktalarını al
    std::vector<cv::Point2f> src(4), dst(4);
    for (int i = 0; i < 4; i++) {
        src[i] = cv::Point2f([srcPoints[i] CGPointValue].x, [srcPoints[i] CGPointValue].y);
        dst[i] = cv::Point2f([dstPoints[i] CGPointValue].x, [dstPoints[i] CGPointValue].y);
    }
    
    // C++ fonksiyonunu çağır
    cv::Mat transformedImage;
    try {
        transformedImage = applyPerspectiveTransform(matImage, src, dst);
    } catch (const std::invalid_argument& e) {
        NSLog(@"%s", e.what());
        return nil;
    }
    
    // cv::Mat'i UIImage'a dönüştür
    return [self MatToUIImage:transformedImage];
}

+ (UIImage *)remapImage:(UIImage *)image
               withMapX:(NSArray<NSArray<NSNumber *> *> *)mapX
                   mapY:(NSArray<NSArray<NSNumber *> *> *)mapY
          interpolation:(int)interpolation {
    
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];
    
    int rows = (int)mapX.count;
    int cols = (int)[mapX[0] count];
    cv::Mat mapXMat(rows, cols, CV_32F);
    cv::Mat mapYMat(rows, cols, CV_32F);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mapXMat.at<float>(i, j) = [mapX[i][j] floatValue];
            mapYMat.at<float>(i, j) = [mapY[i][j] floatValue];
        }
    }
    
    cv::Mat remappedImage = remapImage(matImage, mapXMat, mapYMat, interpolation);
    
    return [self MatToUIImage:remappedImage];
}


+ (UIImage *)transposeImage:(UIImage *)image {
    // UIImage'dan cv::Mat'e dönüşüm
    cv::Mat mat;
    [self UIImageToMat:image mat:mat];
    
    // C++ fonksiyonunu çağır
    cv::Mat transposedMat;
    try {
        transposedMat = transposeImage(mat);
    } catch (const std::exception& e) {
        NSLog(@"Error: %s", e.what());
        return nil;
    }
    
    // cv::Mat'i UIImage'a dönüştür
    return [self MatToUIImage:transposedMat];
}

+ (nullable UIImage *)pyrUpWithImage:(UIImage *)image {
    // UIImage'dan cv::Mat'e dönüşüm
    cv::Mat srcMat;
    [self UIImageToMat:image mat:srcMat];
    
    if (srcMat.empty()) {
        NSLog(@"Error: Source image is empty.");
        return nil;
    }
    
    // C++ fonksiyonunu çağır
    cv::Mat dstMat;
    try {
        dstMat = pyrUpWithImage(srcMat);
    } catch (const std::exception& e) {
        NSLog(@"Error: %s", e.what());
        return nil;
    }
    
    // Sonucu UIImage'a dönüştür
    return [self MatToUIImage:dstMat];
}

+ (nullable UIImage *)pyrDownWithImage:(UIImage *)image {
    // UIImage'dan cv::Mat'e dönüşüm
    cv::Mat srcMat;
    [self UIImageToMat:image mat:srcMat];
    
    if (srcMat.empty()) {
        NSLog(@"Error: Source image is empty.");
        return nil;
    }
    
    // C++ fonksiyonunu çağır
    cv::Mat dstMat;
    try {
        dstMat = pyrDownWithImage(srcMat);
    } catch (const std::exception& e) {
        NSLog(@"Error: %s", e.what());
        return nil;
    }
    
    // Sonucu UIImage'a dönüştür
    return [self MatToUIImage:dstMat];
}

+ (void)resizeWindowWithName:(NSString *)windowName width:(int)width height:(int)height {
    std::string windowNameStr = [windowName UTF8String];
    
    // C++ fonksiyonunu çağır
    resizeWindowWithName(windowNameStr, width, height);
}

// 3-Drawing Functions

+ (UIImage *)drawLineOnImage:(UIImage *)image start:(CGPoint)start end:(CGPoint)end color:(UIColor *)color thickness:(int)thickness {
    // UIImage'den cv::Mat'e dönüşüm
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];
    
    // UIColor'dan cv::Scalar'a dönüşüm
    cv::Scalar lineColor;
    [self UIColorToScalar:color scalar:lineColor];
    
    // Başlangıç ve bitiş noktalarını cv::Point'e dönüştür
    cv::Point cvStart(static_cast<int>(start.x), static_cast<int>(start.y));
    cv::Point cvEnd(static_cast<int>(end.x), static_cast<int>(end.y));
    
    // C++ fonksiyonunu çağır
    drawLineOnImage(matImage, cvStart, cvEnd, lineColor, thickness);
    
    // cv::Mat'ten UIImage'e dönüşüm
    return [self MatToUIImage:matImage];
}

+ (UIImage *)drawCircleOnImage:(UIImage *)image
                       atPoint:(CGPoint)center
                    withRadius:(int)radius
                      andColor:(UIColor *)color
                     lineWidth:(int)lineWidth {
    // UIImage'den cv::Mat'e dönüşüm
    cv::Mat mat;
    [self UIImageToMat:image mat:mat];
    
    // UIColor'dan cv::Scalar'a dönüşüm
    CGFloat r, g, b, a;
    [color getRed:&r green:&g blue:&b alpha:&a];
    cv::Scalar circleColor(b * 255, g * 255, r * 255); // OpenCV BGR formatını kullanır
    
    // Başlangıç noktası ve daireyi çizmek için C++ fonksiyonunu çağır
    cv::Point centerPoint(cvRound(center.x), cvRound(center.y));
    drawCircleOnImage(mat, centerPoint, radius, circleColor, lineWidth);
    
    // cv::Mat'ten UIImage'e dönüşüm
    return [self MatToUIImage:mat];
}

+ (UIImage *)drawRectangleOnImage:(UIImage *)image
                        fromPoint:(CGPoint)topLeft
                          toPoint:(CGPoint)bottomRight
                        withColor:(UIColor *)color
                        lineWidth:(int)lineWidth {
    // UIImage'den cv::Mat'e dönüşüm
    cv::Mat mat;
    [self UIImageToMat:image mat:mat];
    
    // UIColor'dan cv::Scalar'a dönüşüm
    CGFloat r, g, b, a;
    [color getRed:&r green:&g blue:&b alpha:&a];
    cv::Scalar rectangleColor(b * 255, g * 255, r * 255); // OpenCV BGR formatını kullanır
    
    // Dikdörtgeni çizmek için C++ fonksiyonunu çağır
    cv::Point topLeftPoint(cvRound(topLeft.x), cvRound(topLeft.y));
    cv::Point bottomRightPoint(cvRound(bottomRight.x), cvRound(bottomRight.y));
    drawRectangleOnImage(mat, topLeftPoint, bottomRightPoint, rectangleColor, lineWidth);
    
    // cv::Mat'ten UIImage'e dönüşüm
    return [self MatToUIImage:mat];
}

+ (cv::Mat)UIImageToMat:(UIImage *)image {
    if (!image.CGImage) {
        NSLog(@"UIImage does not contain a valid CGImage.");
        return cv::Mat(); // Boş bir cv::Mat döndür
    }
    
    CGFloat width = image.size.width;
    CGFloat height = image.size.height;
    int widthInt = (int)round(width);
    int heightInt = (int)round(height);
    
    cv::Mat mat(heightInt, widthInt, CV_8UC4);
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef contextRef = CGBitmapContextCreate(mat.data,
                                                    widthInt,
                                                    heightInt,
                                                    8,
                                                    mat.step[0],
                                                    colorSpace,
                                                    kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
    
    if (!contextRef) {
        NSLog(@"Failed to create CGContextRef.");
        CGColorSpaceRelease(colorSpace);
        return cv::Mat(); // Boş bir cv::Mat döndür
    }
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, width, height), image.CGImage);
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    
    cv::Mat bgrMat;
    cv::cvtColor(mat, bgrMat, cv::COLOR_RGBA2BGR);
    
    return bgrMat;
}

// cv::Mat'i UIImage'e dönüştürme
+ (UIImage *)MatToUIImage:(const cv::Mat &)mat {
    if (mat.empty()) {
        NSLog(@"Empty cv::Mat received.");
        return nil;
    }
    
    cv::Mat rgbMat;
    cv::cvtColor(mat, rgbMat, cv::COLOR_BGR2RGB);
    
    NSData *data = [NSData dataWithBytes:rgbMat.data length:rgbMat.elemSize() * rgbMat.total()];
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    if (!provider) {
        NSLog(@"Failed to create CGDataProvider.");
        CGColorSpaceRelease(colorSpace);
        return nil;
    }
    
    CGImageRef imageRef = CGImageCreate(rgbMat.cols, rgbMat.rows, 8, 24, rgbMat.step[0], colorSpace, kCGImageAlphaNone | kCGBitmapByteOrderDefault, provider, NULL, false, kCGRenderingIntentDefault);
    
    UIImage *image = [UIImage imageWithCGImage:imageRef];
    
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return image;
}

+ (UIImage *)drawEllipseOnImage:(UIImage *)image
                         center:(CGPoint)center
                           axes:(CGSize)axes
                          angle:(double)angle
                     startAngle:(double)startAngle
                       endAngle:(double)endAngle
                          color:(UIColor *)color
                      thickness:(int)thickness {
    // UIImage'den cv::Mat'e dönüşüm
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];
    
    // UIColor'dan cv::Scalar'a dönüşüm
    cv::Scalar ellipseColor;
    [self UIColorToScalar:color scalar:ellipseColor];
    
    // OpenCV için merkez ve boyutları ayarlama
    cv::Point cvCenter(cvRound(center.x), cvRound(center.y));
    cv::Size cvAxes(cvRound(axes.width), cvRound(axes.height));
    
    // C++ fonksiyonunu çağırarak elips çiz
    drawEllipseOnImage(matImage, cvCenter, cvAxes, angle, startAngle, endAngle, ellipseColor, thickness);
    
    // cv::Mat'ten UIImage'e dönüşüm
    return [self MatToUIImage:matImage];
}

+ (void)UIColorToScalar:(UIColor *)color scalar:(cv::Scalar&)scalar {
    CGFloat r, g, b, a;
    [color getRed:&r green:&g blue:&b alpha:&a];
    
    // OpenCV BGR formatına dönüştür
    scalar = cv::Scalar(b * 255, g * 255, r * 255, a * 255);
}

+ (UIImage *)addTextToUIImage:(UIImage *)image
                         text:(NSString *)text
                     position:(CGPoint)position
                     fontFace:(int)fontFace
                    fontScale:(double)fontScale
                        color:(UIColor *)color
                    thickness:(int)thickness
                     lineType:(int)lineType {
    // UIImage -> cv::Mat dönüştürme
    cv::Mat cvImage = [self cvMatFromUIImage:image];
    
    // UIColor -> cv::Scalar dönüştürme
    CGFloat red, green, blue, alpha;
    [color getRed:&red green:&green blue:&blue alpha:&alpha];
    cv::Scalar cvColor(blue * 255, green * 255, red * 255); // OpenCV BGR formatını kullanır
    
    // C++ fonksiyonunu çağırarak görüntü üzerine yazıyı ekle
    cv::Mat resultImage = addTextToImage(cvImage, [text UTF8String], cv::Point(position.x, position.y), fontFace, fontScale, cvColor, thickness, lineType);
    
    // Sonuç olarak işlenmiş cv::Mat'i tekrar UIImage'a dönüştür
    return [self UIImageFromCVMat:resultImage];
}

+ (UIImage *)fillPolygonOnImage:(UIImage *)image
                     withPoints:(NSArray<NSValue *> *)points
                       andColor:(UIColor *)color {
    // UIImage'den cv::Mat'e dönüşüm
    cv::Mat mat = [self UIImageToMat:image];
    
    // Çizgi rengini UIColor'dan cv::Scalar'a çevirme
    CGFloat r, g, b, a;
    [color getRed:&r green:&g blue:&b alpha:&a];
    cv::Scalar fillColor(b * 255, g * 255, r * 255); // OpenCV BGR kullanır
    
    // Noktaları cv::Point dizisine dönüştürme
    std::vector<cv::Point> pts;
    for (NSValue *pointValue in points) {
        CGPoint point = [pointValue CGPointValue];
        pts.push_back(cv::Point(cvRound(point.x), cvRound(point.y)));
    }
    
    // C++ fonksiyonunu çağırarak poligon doldurma işlemini yap
    cv::Mat filledMat = fillPolygonOnImage(mat, pts, fillColor);
    
    // cv::Mat'ten UIImage'e dönüşüm
    return [self MatToUIImage:filledMat];
}

+ (UIImage *)drawPolylinesOnImage:(UIImage *)image
                       withPoints:(NSArray<NSValue *> *)points
                         andColor:(UIColor *)color
                        lineWidth:(int)lineWidth {
    // UIImage'den cv::Mat'e dönüşüm
    cv::Mat mat = [self UIImageToMat:image];
    
    // Çizgi rengini UIColor'dan cv::Scalar'a çevirme
    CGFloat r, g, b, a;
    [color getRed:&r green:&g blue:&b alpha:&a];
    cv::Scalar polylineColor(b * 255, g * 255, r * 255); // OpenCV BGR kullanır
    
    // Noktaları cv::Point dizisine dönüştürme
    std::vector<cv::Point> pts;
    for (NSValue *pointValue in points) {
        CGPoint point = [pointValue CGPointValue];
        pts.push_back(cv::Point(cvRound(point.x), cvRound(point.y)));
    }
    
    // C++ fonksiyonunu çağırarak poligon çizme işlemini yap
    cv::Mat drawnMat = drawPolylinesOnImage(mat, pts, polylineColor, lineWidth);
    
    // cv::Mat'ten UIImage'e dönüşüm
    return [self MatToUIImage:drawnMat];
}

// 4-Thresholding and Edge Detection

+ (UIImage *)applyThresholdToImage:(UIImage *)image
                         threshold:(double)threshold
                          maxValue:(double)maxValue
                     thresholdType:(int)thresholdType {
    // UIImage'ı cv::Mat formatına çevirin
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];
    
    // Giriş görüntüsünün boş olup olmadığını kontrol edin
    if (matImage.empty()) {
        NSLog(@"Error: Source image is empty.");
        return nil;
    }
    
    // C++ fonksiyonunu çağırarak eşikleme işlemini gerçekleştir
    cv::Mat thresholdedImage = applyThresholdToImage(matImage, threshold, maxValue, thresholdType);
    
    // cv::Mat'ten UIImage formatına geri dönüştürün
    return [self MatToUIImage:thresholdedImage];
}

+ (UIImage *)applyAdaptiveThresholdToImage:(UIImage *)image
                                  maxValue:(double)maxValue
                            adaptiveMethod:(int)adaptiveMethod
                             thresholdType:(int)thresholdType
                                 blockSize:(int)blockSize
                                         C:(double)C {
    // UIImage'ı cv::Mat formatına çevirin
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];
    
    // Giriş görüntüsünün boş olup olmadığını kontrol edin
    if (matImage.empty()) {
        NSLog(@"Error: Source image is empty.");
        return nil;
    }
    
    // C++ fonksiyonunu çağırarak adaptif eşikleme işlemini gerçekleştir
    cv::Mat thresholdedImage = applyAdaptiveThresholdToImage(matImage, maxValue, adaptiveMethod, thresholdType, blockSize, C);
    
    // cv::Mat'ten UIImage formatına geri dönüştürün
    return [self MatToUIImage:thresholdedImage];
}

+ (UIImage *)applyCannyToImage:(UIImage *)image threshold1:(double)threshold1 threshold2:(double)threshold2 {
    // UIImage'ı cv::Mat formatına çevirin
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];
    
    // Giriş görüntüsünün boş olup olmadığını kontrol edin
    if (matImage.empty()) {
        NSLog(@"Error: Source image is empty.");
        return nil;
    }
    
    // C++ fonksiyonunu çağırarak Canny kenar algılama işlemini gerçekleştir
    cv::Mat edges = applyCannyToImage(matImage, threshold1, threshold2);
    
    // cv::Mat'ten UIImage formatına geri dönüştürün
    return [self MatToUIImage:edges];
}

+ (UIImage *)applySobelToUIImage:(UIImage *)image
                              dx:(int)dx
                              dy:(int)dy
                      kernelSize:(int)kernelSize {
    // UIImage -> cv::Mat dönüştürme
    cv::Mat cvImage = [self cvMatFromUIImage:image];
    
    // Sobel filtresini uygula
    cv::Mat resultImage = applySobelFilter(cvImage, dx, dy, kernelSize);
    if (resultImage.empty()) {
        NSLog(@"Error: Resulting cv::Mat is empty after Sobel filter");
        return nil;
    }
    
    // Sonuç olarak işlenmiş cv::Mat'i tekrar UIImage'a dönüştür
    return [self UIImageFromCVMat:resultImage];
}

+ (nullable UIImage *)laplacianWithImage:(UIImage *)image kernelSize:(int)kernelSize {
    // UIImage'ı cv::Mat formatına çevirin
    cv::Mat srcMat;
    [self UIImageToMat:image mat:srcMat];
    
    if (srcMat.empty()) {
        NSLog(@"Error: Source image is empty.");
        return nil;
    }
    
    // C++ fonksiyonunu çağırarak Laplacian işlemini gerçekleştir
    cv::Mat dstMat = laplacianWithImage(srcMat, kernelSize);
    
    // cv::Mat'ten UIImage formatına geri dönüştürün
    return [self MatToUIImage:dstMat];
}

+ (UIImage *)inRangeWithImage:(UIImage *)image lowerBound:(NSArray<NSNumber *> *)lower upperBound:(NSArray<NSNumber *> *)upper {
    cv::Mat mat = [self cvMatFromUIImage:image];
    
    if (mat.empty()) {
        NSLog(@"Hata: Görüntü cv::Mat'e dönüştürülürken boş çıktı.");
        return nil; // Geçerli bir UIImage döndüremeyeceğinden nil döndürün
    }
    
    // Alt ve üst sınırları ayarlayın
    cv::Scalar lowerScalar([lower[0] doubleValue], [lower[1] doubleValue], [lower[2] doubleValue]);
    cv::Scalar upperScalar([upper[0] doubleValue], [upper[1] doubleValue], [upper[2] doubleValue]);
    
    // Mask oluşturmak için cv::inRange kullanın
    cv::Mat mask;
    cv::inRange(mat, lowerScalar, upperScalar, mask);
    
    // Maskeyi UIImage'ye dönüştürüp döndürün
    return [self UIImageFromCVMat:mask];
}

+ (NSArray<NSValue *> *)findNonZeroWithImage:(UIImage *)image {
    // UIImage'ı cv::Mat türüne dönüştürme
    cv::Mat srcMat;
    UIImageToMat(image, srcMat);
    
    if (srcMat.empty()) {
        NSLog(@"Error: Source image is empty.");
        return @[];
    }
    
    // Renkli görüntüyü gri tonlamalı hale getirme
    if (srcMat.channels() > 1) {
        cv::cvtColor(srcMat, srcMat, cv::COLOR_BGR2GRAY);
    }
    
    // Sıfır olmayan pikselleri tespit etmek için C++ fonksiyonunu çağırma
    std::vector<cv::Point> nonZeroPoints = findNonZeroWithImage(srcMat);
    
    // C++ vektörünü NSArray türüne dönüştürme
    NSMutableArray<NSValue *> *resultArray = [NSMutableArray array];
    for (const auto& point : nonZeroPoints) {
        [resultArray addObject:[NSValue valueWithCGPoint:CGPointMake(point.x, point.y)]];
    }
    
    return [resultArray copy];
}

// 5-Image Filtering

+ (UIImage *)gaussianBlur:(UIImage *)image
           withKernelSize:(CGSize)kernelSize
                    sigma:(double)sigma {
    // UIImage'ı cv::Mat'e dönüştürme
    cv::Mat srcMat;
    [self UIImageToMat:image mat:srcMat];
    
    // C++ fonksiyonunu çağırarak Gaussian bulanıklık uygulama
    cv::Mat dstMat = gaussianBlur(srcMat, cv::Size(kernelSize.width, kernelSize.height), sigma);
    
    // Sonucu UIImage'a dönüştürme
    return [self MatToUIImage:dstMat];
}

+ (UIImage *)medianBlur:(UIImage *)image
         withKernelSize:(int)kernelSize {
    // UIImage'ı cv::Mat'e dönüştürme
    cv::Mat srcMat;
    [self UIImageToMat:image mat:srcMat];
    
    // C++ fonksiyonunu çağırarak median bulanıklık uygulama
    cv::Mat dstMat = medianBlur(srcMat, kernelSize);
    
    // Sonucu UIImage'a dönüştürme
    return [self MatToUIImage:dstMat];
}

+ (UIImage *)blur:(UIImage *)image
   withKernelSize:(CGSize)kernelSize {
    // UIImage'ı cv::Mat'e dönüştürme
    cv::Mat srcMat;
    [self UIImageToMat:image mat:srcMat];
    
    // C++ fonksiyonunu çağırarak bulanıklık uygulama
    cv::Mat dstMat = blur(srcMat, cv::Size(kernelSize.width, kernelSize.height));
    
    // Sonucu UIImage'a dönüştürme
    return [self MatToUIImage:dstMat];
}

+ (UIImage *)applyBilateralFilterToImage:(UIImage *)image
                                diameter:(int)diameter
                              sigmaColor:(double)sigmaColor
                              sigmaSpace:(double)sigmaSpace {
    // UIImage'ı cv::Mat'e dönüştürme
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];
    
    // C++ fonksiyonunu çağırarak bilateral filtre uygulama
    cv::Mat filteredImage = applyBilateralFilterToImage(matImage, diameter, sigmaColor, sigmaSpace);
    
    // Sonucu UIImage'a dönüştürme
    return [self MatToUIImage:filteredImage];
}

+ (UIImage *)applyFilter2DToImage:(UIImage *)image
                           kernel:(NSArray<NSArray<NSNumber *> *> *)kernel {
    // UIImage'ı cv::Mat'e dönüştürme
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];
    
    // Çekirdek matrisini oluşturma
    int rows = (int)kernel.count;
    int cols = (int)[kernel[0] count];
    cv::Mat kernelMat(rows, cols, CV_32F);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            kernelMat.at<float>(i, j) = [kernel[i][j] floatValue];
        }
    }
    
    // C++ fonksiyonunu çağırarak filtre uygulama
    cv::Mat filteredImage = applyFilter2DToImage(matImage, kernelMat);
    
    // Sonucu UIImage'a dönüştürme
    return [self MatToUIImage:filteredImage];
}

+ (UIImage *)applyBoxFilterToImage:(UIImage *)image
                            ddepth:(int)ddepth
                             ksize:(CGSize)ksize {
    // UIImage'ı cv::Mat'e dönüştürme
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];
    
    // C++ fonksiyonunu çağırarak kutu filtre uygulama
    cv::Mat filteredImage = applyBoxFilterToImage(matImage, ddepth, cv::Size(ksize.width, ksize.height));
    
    // Sonucu UIImage'a dönüştürme
    return [self MatToUIImage:filteredImage];
}

+ (UIImage *)applyScharrOnImage:(UIImage *)image {
    cv::Mat cvImage;
    UIImageToMat(image, cvImage); // UIImage'ı cv::Mat'e dönüştürme
    
    cv::Mat scharrResult = applyScharrOnImage(cvImage); // C++ fonksiyonunu çağırma
    
    return MatToUIImage(scharrResult); // Sonucu UIImage'a çevirme ve döndürme
}

+ (UIImage *)addImage:(UIImage *)image1
            withImage:(UIImage *)image2 {
    cv::Mat cvImage1, cvImage2;
    
    // UIImage'ları cv::Mat formatına dönüştür
    UIImageToMat(image1, cvImage1);
    UIImageToMat(image2, cvImage2);
    
    // Görsellerin boyutlarının aynı olduğundan emin olun
    if (cvImage1.size() != cvImage2.size()) {
        NSLog(@"Error: Images must be the same size");
        return nil;
    }
    
    // C++ fonksiyonunu çağırarak görselleri topla
    cv::Mat result = addImage(cvImage1, cvImage2);
    
    // Sonucu UIImage'a çevir ve döndür
    return MatToUIImage(result);
}

+ (UIImage *)subtractImage:(UIImage *)image1
                 fromImage:(UIImage *)image2 {
    cv::Mat cvImage1, cvImage2;
    
    // UIImage'ları cv::Mat formatına dönüştür
    UIImageToMat(image1, cvImage1);
    UIImageToMat(image2, cvImage2);
    
    // Görsellerin boyutlarının aynı olduğundan emin olun
    if (cvImage1.size() != cvImage2.size()) {
        NSLog(@"Error: Images must be the same size");
        return nil;
    }
    
    // C++ fonksiyonunu çağırarak görselleri çıkar
    cv::Mat result = subtractImage(cvImage1, cvImage2);
    
    // Sonucu UIImage'a çevir ve döndür
    return MatToUIImage(result);
}

+ (UIImage *)multiplyImage:(UIImage *)image1
                 withImage:(UIImage *)image2 {
    cv::Mat cvImage1, cvImage2;
    
    // UIImage'ları cv::Mat formatına dönüştür
    UIImageToMat(image1, cvImage1);
    UIImageToMat(image2, cvImage2);
    
    // Görsellerin boyutlarının aynı olduğundan emin olun
    if (cvImage1.size() != cvImage2.size()) {
        NSLog(@"Error: Images must be the same size");
        return nil;
    }
    
    // C++ fonksiyonunu çağırarak görselleri çarp
    cv::Mat result = multiplyImage(cvImage1, cvImage2);
    
    // Sonucu UIImage'a çevir ve döndür
    return MatToUIImage(result);
}

+ (UIImage *)divideImage:(UIImage *)image1 byImage:(UIImage *)image2 {
    cv::Mat cvImage1, cvImage2;
    
    // UIImage'ları cv::Mat formatına dönüştür
    UIImageToMat(image1, cvImage1);
    UIImageToMat(image2, cvImage2);
    
    // Görsellerin boyutlarının aynı olduğundan emin olun
    if (cvImage1.size() != cvImage2.size()) {
        NSLog(@"Error: Images must be the same size");
        return nil;
    }
    
    // C++ fonksiyonunu çağırarak görselleri böl
    cv::Mat result = divideImage(cvImage1, cvImage2);
    
    // Sonucu UIImage'a çevir ve döndür
    return MatToUIImage(result);
}

// 6-Morphological Operations

+ (UIImage *)erodeImage:(UIImage *)image withKernelSize:(int)kernelSize {
    cv::Mat cvImage;
    
    // UIImage'ı cv::Mat formatına dönüştür
    UIImageToMat(image, cvImage);
    
    // C++ fonksiyonunu çağırarak görüntüyü aşındır
    cv::Mat result = erodeImage(cvImage, kernelSize);
    
    // Sonucu UIImage'a çevir ve döndür
    return MatToUIImage(result);
}

+ (UIImage *)dilateImage:(UIImage *)image withKernelSize:(int)kernelSize {
    cv::Mat cvImage;
    
    // UIImage'ı cv::Mat formatına dönüştür
    UIImageToMat(image, cvImage);
    
    // C++ fonksiyonunu çağırarak görüntüyü genişlet
    cv::Mat result = dilateImage(cvImage, kernelSize);
    
    // Sonucu UIImage'a çevir ve döndür
    return MatToUIImage(result);
}

+ (UIImage *)applyMorphologyEx:(UIImage *)image withOperation:(MorphType)operation kernelSize:(int)kernelSize {
    cv::Mat cvImage;
    
    // UIImage'ı cv::Mat formatına dönüştür
    UIImageToMat(image, cvImage);
    
    // C++ fonksiyonunu çağırarak morfolojik işlemi uygula
    cv::Mat result = applyMorphologyEx(cvImage, operation, kernelSize);
    
    // Sonucu UIImage'a çevir ve döndür
    return MatToUIImage(result);
}

+ (UIImage *)getStructuringElementWithType:(ElementType)type kernelSize:(int)kernelSize {
    // C++ fonksiyonunu çağırarak çekirdek oluştur
    cv::Mat kernel = getStructuringElementWithType(type, kernelSize);
    
    // Kernel'i görüntü olarak döndür
    cv::Mat kernelImage;
    cv::normalize(kernel, kernelImage, 0, 255, cv::NORM_MINMAX);
    kernelImage.convertTo(kernelImage, CV_8U);
    
    // Sonucu UIImage'a çevir ve döndür
    return MatToUIImage(kernelImage);
}

// 7-Image Contours and Shape Analysis

+ (nullable NSString *)findContoursInImage:(UIImage *)image {
    cv::Mat cvImage;
    
    // UIImage'ı cv::Mat formatına dönüştür
    UIImageToMat(image, cvImage);
    
    // C++ fonksiyonunu çağırarak konturları bul
    std::vector<std::vector<cv::Point>> contours = findContoursInImage(cvImage);
    
    // Konturları JSON formatına çevir
    NSMutableArray *contoursArray = [NSMutableArray array];
    for (const auto& contour : contours) {
        NSMutableArray *contourPoints = [NSMutableArray array];
        for (const auto& point : contour) {
            NSDictionary *pointDict = @{@"x": @(point.x), @"y": @(point.y)};
            [contourPoints addObject:pointDict];
        }
        [contoursArray addObject:contourPoints];
    }
    
    // JSON formatında döndür
    NSError *error;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:contoursArray options:0 error:&error];
    if (!jsonData) {
        NSLog(@"Error converting contours to JSON: %@", error.localizedDescription);
        return nil;
    }
    return [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
}

+ (UIImage *)drawContoursOnImage:(UIImage *)image
                    withContours:(NSString *)contoursJSON
                           color:(UIColor *)color
                       thickness:(int)thickness {
    cv::Mat cvImage;
    UIImageToMat(image, cvImage);
    
    // JSON'u parse et ve konturları yükle
    NSError *error;
    NSData *jsonData = [contoursJSON dataUsingEncoding:NSUTF8StringEncoding];
    NSArray *contoursArray = [NSJSONSerialization JSONObjectWithData:jsonData options:0 error:&error];
    if (error) {
        NSLog(@"Error parsing JSON: %@", error.localizedDescription);
        return nil;
    }
    
    std::vector<std::vector<cv::Point>> contours;
    for (NSArray *contourArray in contoursArray) {
        std::vector<cv::Point> contour;
        for (NSDictionary *pointDict in contourArray) {
            int x = [pointDict[@"x"] intValue];
            int y = [pointDict[@"y"] intValue];
            contour.push_back(cv::Point(x, y));
        }
        contours.push_back(contour);
    }
    
    // UIColor'ı cv::Scalar'a çevir
    CGFloat red, green, blue, alpha;
    [color getRed:&red green:&green blue:&blue alpha:&alpha];
    cv::Scalar cvColor(red * 255, green * 255, blue * 255, alpha * 255);
    
    // C++ fonksiyonunu çağırarak konturları çiz
    drawContoursOnImage(cvImage, contours, cvColor, thickness);
    
    // Sonucu UIImage'a çevir ve döndür
    return MatToUIImage(cvImage);
}

+ (double)arcLengthOfContour:(NSString *)contourJSON isClosed:(BOOL)isClosed {
    // JSON'u parse et ve kontur noktalarını yükle
    NSError *error;
    NSData *jsonData = [contourJSON dataUsingEncoding:NSUTF8StringEncoding];
    NSArray *contourArray = [NSJSONSerialization JSONObjectWithData:jsonData options:0 error:&error];
    if (error) {
        NSLog(@"Error parsing JSON: %@", error.localizedDescription);
        return -1.0;
    }
    
    // Kontur noktalarını oluştur
    std::vector<cv::Point> contour;
    for (NSDictionary *pointDict in contourArray) {
        int x = [pointDict[@"x"] intValue];
        int y = [pointDict[@"y"] intValue];
        contour.push_back(cv::Point(x, y));
    }
    
    // C++ fonksiyonunu çağırarak çevreyi hesapla
    double arcLength = arcLengthOfContour(contour, isClosed);
    return arcLength;
}

+ (double)contourAreaOfContour:(NSString *)contourJSON {
    // JSON'u parse et ve kontur noktalarını yükle
    NSError *error;
    NSData *jsonData = [contourJSON dataUsingEncoding:NSUTF8StringEncoding];
    NSArray *contourArray = [NSJSONSerialization JSONObjectWithData:jsonData options:0 error:&error];
    if (error) {
        NSLog(@"Error parsing JSON: %@", error.localizedDescription);
        return -1.0;
    }
    
    // Kontur noktalarını oluştur
    std::vector<cv::Point> contour;
    for (NSDictionary *pointDict in contourArray) {
        int x = [pointDict[@"x"] intValue];
        int y = [pointDict[@"y"] intValue];
        contour.push_back(cv::Point(x, y));
    }
    
    // C++ fonksiyonunu çağırarak alanı hesapla
    double area = contourAreaOfContour(contour);
    return area;
}

+ (nullable NSString *)approxPolyDPOfContour:(NSString *)contourJSON epsilon:(double)epsilon isClosed:(BOOL)isClosed {
    // JSON'u parse et ve kontur noktalarını yükle
    NSError *error;
    NSData *jsonData = [contourJSON dataUsingEncoding:NSUTF8StringEncoding];
    NSArray *contourArray = [NSJSONSerialization JSONObjectWithData:jsonData options:0 error:&error];
    if (error) {
        NSLog(@"Error parsing JSON: %@", error.localizedDescription);
        return nil;
    }
    
    // Kontur noktalarını oluştur
    std::vector<cv::Point> contour;
    for (NSDictionary *pointDict in contourArray) {
        int x = [pointDict[@"x"] intValue];
        int y = [pointDict[@"y"] intValue];
        contour.push_back(cv::Point(x, y));
    }
    
    // Yaklaşık poligon noktalarını hesapla
    std::vector<cv::Point> approxContour;
    approxPolyDPOfContour(contour, approxContour, epsilon, isClosed);
    
    // Yaklaşık poligon noktalarını JSON formatına çevir
    NSMutableArray *approxContourArray = [NSMutableArray array];
    for (const auto& point : approxContour) {
        NSDictionary *pointDict = @{@"x": @(point.x), @"y": @(point.y)};
        [approxContourArray addObject:pointDict];
    }
    
    // JSON formatında döndür
    NSData *approxJSONData = [NSJSONSerialization dataWithJSONObject:approxContourArray options:0 error:&error];
    if (error) {
        NSLog(@"Error converting approx contour to JSON: %@", error.localizedDescription);
        return nil;
    }
    return [[NSString alloc] initWithData:approxJSONData encoding:NSUTF8StringEncoding];
}

+ (nullable NSString *)convexHullOfContour:(NSString *)contourJSON {
    // JSON'u parse et ve kontur noktalarını yükle
    NSError *error;
    NSData *jsonData = [contourJSON dataUsingEncoding:NSUTF8StringEncoding];
    NSArray *contourArray = [NSJSONSerialization JSONObjectWithData:jsonData options:0 error:&error];
    if (error) {
        NSLog(@"Error parsing JSON: %@", error.localizedDescription);
        return nil;
    }
    
    // Kontur noktalarını oluştur
    std::vector<cv::Point> contour;
    for (NSDictionary *pointDict in contourArray) {
        int x = [pointDict[@"x"] intValue];
        int y = [pointDict[@"y"] intValue];
        contour.push_back(cv::Point(x, y));
    }
    
    // Konveks hull'ü hesapla
    std::vector<cv::Point> hull;
    convexHullOfContour(contour, hull);
    
    // Konveks hull noktalarını JSON formatına çevir
    NSMutableArray *hullArray = [NSMutableArray array];
    for (const auto& point : hull) {
        NSDictionary *pointDict = @{@"x": @(point.x), @"y": @(point.y)};
        [hullArray addObject:pointDict];
    }
    
    // JSON formatında döndür
    NSData *hullJSONData = [NSJSONSerialization dataWithJSONObject:hullArray options:0 error:&error];
    if (error) {
        NSLog(@"Error converting hull to JSON: %@", error.localizedDescription);
        return nil;
    }
    return [[NSString alloc] initWithData:hullJSONData encoding:NSUTF8StringEncoding];
}

+ (BOOL)isContourConvex:(NSString *)contourJSON {
    // JSON'u parse et ve kontur noktalarını yükle
    NSError *error;
    NSData *jsonData = [contourJSON dataUsingEncoding:NSUTF8StringEncoding];
    NSArray *contourArray = [NSJSONSerialization JSONObjectWithData:jsonData options:0 error:&error];
    if (error) {
        NSLog(@"Error parsing JSON: %@", error.localizedDescription);
        return NO;
    }
    
    // Kontur noktalarını oluştur
    std::vector<cv::Point> contour;
    for (NSDictionary *pointDict in contourArray) {
        int x = [pointDict[@"x"] intValue];
        int y = [pointDict[@"y"] intValue];
        contour.push_back(cv::Point(x, y));
    }
    
    // Konturun konveks olup olmadığını kontrol et
    return isContourConvex(contour);
}

+ (NSDictionary *)boundingRectOfContour:(NSString *)contourJSON {
    // JSON'u parse et ve kontur noktalarını yükle
    NSError *error;
    NSData *jsonData = [contourJSON dataUsingEncoding:NSUTF8StringEncoding];
    NSArray *contourArray = [NSJSONSerialization JSONObjectWithData:jsonData options:0 error:&error];
    if (error) {
        NSLog(@"Error parsing JSON: %@", error.localizedDescription);
        return nil;
    }
    
    // Kontur noktalarını oluştur
    std::vector<cv::Point> contour;
    for (NSDictionary *pointDict in contourArray) {
        int x = [pointDict[@"x"] intValue];
        int y = [pointDict[@"y"] intValue];
        contour.push_back(cv::Point(x, y));
    }
    
    // Konturun etrafındaki en küçük dikdörtgeni hesapla
    cv::Rect boundingRect = boundingRectOfContour(contour);
    
    // Dikdörtgen bilgilerini sözlüğe çevir
    NSDictionary *boundingRectDict = @{
        @"x": @(boundingRect.x),
        @"y": @(boundingRect.y),
        @"width": @(boundingRect.width),
        @"height": @(boundingRect.height)
    };
    
    return boundingRectDict;
}

+ (nullable NSDictionary *)minAreaRectOfContour:(NSString *)contourJSON {
    // JSON'u parse et ve kontur noktalarını yükle
    NSError *error;
    NSData *jsonData = [contourJSON dataUsingEncoding:NSUTF8StringEncoding];
    NSArray *contourArray = [NSJSONSerialization JSONObjectWithData:jsonData options:0 error:&error];
    if (error) {
        NSLog(@"Error parsing JSON: %@", error.localizedDescription);
        return nil;
    }
    
    // Kontur noktalarını oluştur
    std::vector<cv::Point> contour;
    for (NSDictionary *pointDict in contourArray) {
        int x = [pointDict[@"x"] intValue];
        int y = [pointDict[@"y"] intValue];
        contour.push_back(cv::Point(x, y));
    }
    
    // Minimum alanlı dikdörtgeni hesapla
    cv::RotatedRect minRect = minAreaRectOfContour(contour);
    
    // Sonucu JSON formatına çevir
    NSDictionary *minRectDict = @{
        @"center": @{@"x": @(minRect.center.x), @"y": @(minRect.center.y)},
        @"size": @{@"width": @(minRect.size.width), @"height": @(minRect.size.height)},
        @"angle": @(minRect.angle)
    };
    
    return minRectDict;
}

+ (nullable NSDictionary *)fitEllipseOfContour:(NSString *)contourJSON {
    // JSON'u parse et ve kontur noktalarını yükle
    NSError *error;
    NSData *jsonData = [contourJSON dataUsingEncoding:NSUTF8StringEncoding];
    NSArray *contourArray = [NSJSONSerialization JSONObjectWithData:jsonData options:0 error:&error];
    if (error) {
        NSLog(@"Error parsing JSON: %@", error.localizedDescription);
        return nil;
    }
    
    // Kontur noktalarını oluştur
    std::vector<cv::Point> contour;
    for (NSDictionary *pointDict in contourArray) {
        int x = [pointDict[@"x"] intValue];
        int y = [pointDict[@"y"] intValue];
        contour.push_back(cv::Point(x, y));
    }
    
    // Yeterli nokta olup olmadığını kontrol et (fitEllipse en az 5 noktaya ihtiyaç duyar)
    if (contour.size() < 5) {
        NSLog(@"Not enough points to fit an ellipse");
        return nil;
    }
    
    // Elipsi uydur
    cv::RotatedRect ellipse = fitEllipseOfContour(contour);
    
    // Sonucu JSON formatına çevir
    NSDictionary *ellipseDict = @{
        @"center": @{@"x": @(ellipse.center.x), @"y": @(ellipse.center.y)},
        @"size": @{@"majorAxis": @(ellipse.size.width), @"minorAxis": @(ellipse.size.height)},
        @"angle": @(ellipse.angle)
    };
    
    return ellipseDict;
}

+ (nullable NSDictionary *)fitLineOfContour:(NSString *)contourJSON {
    // JSON'u parse et ve kontur noktalarını yükle
    NSError *error;
    NSData *jsonData = [contourJSON dataUsingEncoding:NSUTF8StringEncoding];
    NSArray *contourArray = [NSJSONSerialization JSONObjectWithData:jsonData options:0 error:&error];
    if (error) {
        NSLog(@"Error parsing JSON: %@", error.localizedDescription);
        return nil;
    }
    
    // Kontur noktalarını oluştur
    std::vector<cv::Point> contour;
    for (NSDictionary *pointDict in contourArray) {
        int x = [pointDict[@"x"] intValue];
        int y = [pointDict[@"y"] intValue];
        contour.push_back(cv::Point(x, y));
    }
    
    // Konturda yeterli nokta olup olmadığını kontrol et
    if (contour.size() < 2) {
        NSLog(@"Not enough points to fit a line");
        return nil;
    }
    
    // C++ fonksiyonunu çağır
    cv::Vec4f line = fitLineOfContour(contour);
    
    // Sonucu JSON formatına çevir
    NSDictionary *lineDict = @{
        @"direction": @{@"vx": @(line[0]), @"vy": @(line[1])},
        @"point": @{@"x": @(line[2]), @"y": @(line[3])}
    };
    
    return lineDict;
}

// 8-Feature Detection and Matching

+ (nullable NSArray<NSDictionary *> *)goodFeaturesToTrackInImage:(UIImage *)image
                                                      maxCorners:(int)maxCorners
                                                    qualityLevel:(double)qualityLevel
                                                     minDistance:(double)minDistance {
    
    // UIImage'i cv::Mat formatına çevir
    cv::Mat mat;
    UIImageToMat(image, mat);
    
    // Gri tonlamaya dönüştür
    cv::Mat gray;
    cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
    
    // C++ fonksiyonunu çağır
    std::vector<cv::Point2f> corners = goodFeaturesToTrackInImage(gray, maxCorners, qualityLevel, minDistance);
    
    // Sonuçları JSON formatına dönüştür
    NSMutableArray<NSDictionary *> *pointsArray = [NSMutableArray array];
    for (const auto& corner : corners) {
        NSDictionary *pointDict = @{@"x": @(corner.x), @"y": @(corner.y)};
        [pointsArray addObject:pointDict];
    }
    
    return pointsArray;
}

+ (nullable NSArray<NSDictionary *> *)houghLinesInImage:(UIImage *)image
                                                    rho:(double)rho
                                                  theta:(double)theta
                                              threshold:(int)threshold {
    
    // UIImage'i cv::Mat formatına çevir
    cv::Mat mat;
    UIImageToMat(image, mat);
    
    // Gri tonlamaya dönüştür
    cv::Mat gray;
    cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
    
    // Kenar tespiti için Canny uygula
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    
    // C++ fonksiyonunu çağır
    std::vector<cv::Vec2f> lines = houghLinesInImage(edges, rho, theta, threshold);
    
    // Sonuçları JSON formatına dönüştür
    NSMutableArray<NSDictionary *> *linesArray = [NSMutableArray array];
    for (const auto& line : lines) {
        NSDictionary *lineDict = @{@"rho": @(line[0]), @"theta": @(line[1])};
        [linesArray addObject:lineDict];
    }
    
    return linesArray;
}

+ (nullable NSArray<NSDictionary *> *)houghCirclesInImage:(UIImage *)image
                                                       dp:(double)dp
                                                  minDist:(double)minDist
                                                   param1:(double)param1
                                                   param2:(double)param2
                                                minRadius:(int)minRadius
                                                maxRadius:(int)maxRadius {
    
    // UIImage'i cv::Mat formatına çevir
    cv::Mat mat;
    UIImageToMat(image, mat);
    
    // Gri tonlamaya dönüştür
    cv::Mat gray;
    cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
    
    // C++ fonksiyonunu çağır
    std::vector<cv::Vec3f> circles = houghCirclesInImage(gray, dp, minDist, param1, param2, minRadius, maxRadius);
    
    // Sonuçları JSON formatına dönüştür
    NSMutableArray<NSDictionary *> *circlesArray = [NSMutableArray array];
    for (const auto& circle : circles) {
        NSDictionary *circleDict = @{
            @"centerX": @(circle[0]),
            @"centerY": @(circle[1]),
            @"radius": @(circle[2])
        };
        [circlesArray addObject:circleDict];
    }
    
    return circlesArray;
}

+ (nullable UIImage *)cornerHarrisInImage:(UIImage *)image
                                blockSize:(int)blockSize
                                    ksize:(int)ksize
                                        k:(double)k {
    
    // UIImage'i cv::Mat formatına çevir
    cv::Mat mat;
    UIImageToMat(image, mat);
    
    // Gri tonlamaya dönüştür
    cv::Mat gray;
    cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
    
    // C++ fonksiyonunu çağır
    cv::Mat dstNorm = cornerHarrisInImage(gray, blockSize, ksize, k);
    
    // Köşe noktalarını belirgin hale getirmek için işaretleme
    for (int y = 0; y < dstNorm.rows; y++) {
        for (int x = 0; x < dstNorm.cols; x++) {
            if ((int)dstNorm.at<float>(y, x) > 200) {
                cv::circle(mat, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), 2, 8, 0);
            }
        }
    }
    
    // cv::Mat'i UIImage formatına geri çevir ve döndür
    return MatToUIImage(mat);
}

+ (nullable NSArray<NSDictionary *> *)detectORBKeypointsInImage:(UIImage *)image
                                                      nFeatures:(int)nFeatures {
    
    // UIImage'i cv::Mat formatına çevir
    cv::Mat mat;
    UIImageToMat(image, mat);
    
    // Gri tonlamaya dönüştür
    cv::Mat gray;
    cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
    
    // Deskriptör matrisini tanımla
    cv::Mat descriptors;
    
    // C++ fonksiyonunu çağır
    std::vector<cv::KeyPoint> keypoints = detectORBKeypointsInImage(gray, nFeatures, descriptors);
    
    // Anahtar noktaları JSON formatına dönüştür
    NSMutableArray<NSDictionary *> *keypointsArray = [NSMutableArray array];
    for (const auto& kp : keypoints) {
        NSDictionary *keypointDict = @{
            @"x": @(kp.pt.x),
            @"y": @(kp.pt.y),
            @"size": @(kp.size),
            @"angle": @(kp.angle),
            @"response": @(kp.response),
            @"octave": @(kp.octave)
        };
        [keypointsArray addObject:keypointDict];
    }
    
    return keypointsArray;
}

+ (nullable NSArray<NSDictionary *> *)detectSIFTKeypointsInImage:(UIImage *)image
                                                       nFeatures:(int)nFeatures {
    
    // UIImage'i cv::Mat formatına çevir
    cv::Mat mat;
    UIImageToMat(image, mat);
    
    // Gri tonlamaya dönüştür
    cv::Mat gray;
    cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
    
    // Deskriptör matrisini tanımla
    cv::Mat descriptors;
    
    // C++ fonksiyonunu çağır
    std::vector<cv::KeyPoint> keypoints = detectSIFTKeypointsInImage(gray, nFeatures, descriptors);
    
    // Anahtar noktaları JSON formatına dönüştür
    NSMutableArray<NSDictionary *> *keypointsArray = [NSMutableArray array];
    for (const auto& kp : keypoints) {
        NSDictionary *keypointDict = @{
            @"x": @(kp.pt.x),
            @"y": @(kp.pt.y),
            @"size": @(kp.size),
            @"angle": @(kp.angle),
            @"response": @(kp.response),
            @"octave": @(kp.octave)
        };
        [keypointsArray addObject:keypointDict];
    }
    
    return keypointsArray;
}

+ (NSArray<NSDictionary *> *)matchKeypointsWithBFMatcherDescriptors1:(NSArray<NSArray<NSNumber *> *> *)descriptors1
                                                        descriptors2:(NSArray<NSArray<NSNumber *> *> *)descriptors2 {
    
    // Descriptor'ları cv::Mat formatına çevir
    cv::Mat mat1(descriptors1.count, descriptors1[0].count, CV_32F);
    cv::Mat mat2(descriptors2.count, descriptors2[0].count, CV_32F);
    
    for (int i = 0; i < descriptors1.count; i++) {
        for (int j = 0; j < descriptors1[i].count; j++) {
            mat1.at<float>(i, j) = descriptors1[i][j].floatValue;
        }
    }
    
    for (int i = 0; i < descriptors2.count; i++) {
        for (int j = 0; j < descriptors2[i].count; j++) {
            mat2.at<float>(i, j) = descriptors2[i][j].floatValue;
        }
    }
    
    // C++ fonksiyonunu çağır
    std::vector<cv::DMatch> matches = matchKeypointsWithBFMatcherDescriptors1(mat1, mat2);
    
    // Eşleşmeleri JSON formatına dönüştür
    NSMutableArray<NSDictionary *> *matchesArray = [NSMutableArray array];
    for (const auto& match : matches) {
        NSDictionary *matchDict = @{
            @"queryIdx": @(match.queryIdx),
            @"trainIdx": @(match.trainIdx),
            @"distance": @(match.distance)
        };
        [matchesArray addObject:matchDict];
    }
    
    return matchesArray;
}

+ (NSArray<NSDictionary *> *)matchKeypointsWithFlannMatcherDescriptors1:(NSArray<NSArray<NSNumber *> *> *)descriptors1
                                                           descriptors2:(NSArray<NSArray<NSNumber *> *> *)descriptors2 {
    
    // Descriptor'ları cv::Mat formatına çevir
    cv::Mat mat1(descriptors1.count, descriptors1[0].count, CV_32F);
    cv::Mat mat2(descriptors2.count, descriptors2[0].count, CV_32F);
    
    for (int i = 0; i < descriptors1.count; i++) {
        for (int j = 0; j < descriptors1[i].count; j++) {
            mat1.at<float>(i, j) = descriptors1[i][j].floatValue;
        }
    }
    
    for (int i = 0; i < descriptors2.count; i++) {
        for (int j = 0; j < descriptors2[i].count; j++) {
            mat2.at<float>(i, j) = descriptors2[i][j].floatValue;
        }
    }
    
    // C++ fonksiyonunu çağır
    std::vector<cv::DMatch> matches = matchKeypointsWithFlannMatcherDescriptors1(mat1, mat2);
    
    // Eşleşmeleri JSON formatına dönüştür
    NSMutableArray<NSDictionary *> *matchesArray = [NSMutableArray array];
    for (const auto& match : matches) {
        NSDictionary *matchDict = @{
            @"queryIdx": @(match.queryIdx),
            @"trainIdx": @(match.trainIdx),
            @"distance": @(match.distance)
        };
        [matchesArray addObject:matchDict];
    }
    
    return matchesArray;
}

+ (UIImage *)drawKeypointsOnImage:(UIImage *)image keypoints:(NSArray<NSValue *> *)keypoints {
    // UIImage'den cv::Mat'e çevir
    cv::Mat mat;
    UIImageToMat(image, mat);
    
    // Anahtar noktaları cv::KeyPoint formatına dönüştür
    std::vector<cv::KeyPoint> cvKeypoints;
    for (NSValue *value in keypoints) {
        CGPoint point = [value CGPointValue];
        // Anahtar noktaların boyutunu ve açısını varsayılan değerlerle oluşturuyoruz
        cv::KeyPoint keypoint(point.x, point.y, 1); // 1: boyut
        cvKeypoints.push_back(keypoint);
    }
    
    // Anahtar noktaları çizmek için C++ fonksiyonunu çağır
    cv::Mat outputImage = drawKeypointsOnImage(mat, cvKeypoints);
    
    // cv::Mat'ten UIImage'e çevir
    return MatToUIImage(outputImage);
}

+ (UIImage *)matchTemplateInImage:(UIImage *)image
                    templateImage:(UIImage *)templateImage {
    // UIImage'den cv::Mat'e çevir
    cv::Mat matImage, matTemplate;
    UIImageToMat(image, matImage);
    UIImageToMat(templateImage, matTemplate);
    
    // C++ fonksiyonunu çağır
    cv::Point matchLocation;
    matchTemplateInImage(matImage, matTemplate, matchLocation);
    
    // Şablonun bulunduğu alanı çerçevele
    cv::rectangle(matImage, matchLocation,
                  cv::Point(matchLocation.x + matTemplate.cols, matchLocation.y + matTemplate.rows),
                  cv::Scalar(0, 255, 0), 2);
    
    // cv::Mat'ten UIImage'e çevir
    return MatToUIImage(matImage);
}

// 9-Optical Flow

+ (UIImage *)calculateOpticalFlowFromImage:(UIImage *)prevImage
                                   toImage:(UIImage *)nextImage {
    // UIImage'den cv::Mat'e çevir
    cv::Mat matPrev, matNext;
    UIImageToMat(prevImage, matPrev);
    UIImageToMat(nextImage, matNext);
    
    // C++ fonksiyonunu çağır
    cv::Mat outputImage = calculateOpticalFlowFromImage(matPrev, matNext);
    
    // cv::Mat'ten UIImage'e çevir
    return MatToUIImage(outputImage);
}

+ (UIImage *)calculateOpticalFlowPyrLKFromImage:(UIImage *)prevImage
                                        toImage:(UIImage *)nextImage
                                      keypoints:(NSArray<NSValue *> *)keypoints {
    // UIImage'den cv::Mat'e çevir
    cv::Mat matPrev, matNext;
    UIImageToMat(prevImage, matPrev);
    UIImageToMat(nextImage, matNext);
    
    // Anahtar noktaları cv::Point2f formatına dönüştür
    std::vector<cv::Point2f> points;
    for (NSValue *value in keypoints) {
        CGPoint point = [value CGPointValue];
        points.push_back(cv::Point2f(point.x, point.y));
    }
    
    // C++ fonksiyonunu çağır
    cv::Mat outputImage = calculateOpticalFlowPyrLKFromImage(matPrev, matNext, points);
    
    // cv::Mat'ten UIImage'e çevir
    return MatToUIImage(outputImage);
}

+ (UIImage *)calculateMotionGradientFromImage:(UIImage *)image {
    // UIImage'den cv::Mat'e çevir
    cv::Mat mat;
    UIImageToMat(image, mat);
    
    // C++ fonksiyonunu çağır
    cv::Mat outputImage = calculateMotionGradientFromImage(mat);
    
    // cv::Mat'ten UIImage'e çevir
    return MatToUIImage(outputImage);
}

+ (CGFloat)calculateGlobalOrientationFromImage:(UIImage *)image {
    // UIImage'den cv::Mat'e çevir
    cv::Mat mat;
    UIImageToMat(image, mat);
    
    // C++ fonksiyonunu çağır
    double globalOrientation = calculateGlobalOrientationFromImage(mat);
    
    return static_cast<CGFloat>(globalOrientation);
}

// 10-Camera Calibration and 3D Vision

+ (NSArray<NSValue *> *)findChessboardCornersInImage:(UIImage *)image
                                           boardSize:(CGSize)boardSize {
    // UIImage'den cv::Mat'e çevir
    cv::Mat mat;
    UIImageToMat(image, mat);
    
    // Satranç tahtası köşelerini bulacak vektörü oluştur
    std::vector<cv::Point2f> corners;
    cv::Size patternSize(static_cast<int>(boardSize.width), static_cast<int>(boardSize.height));
    
    // C++ fonksiyonunu çağır
    bool found = findChessboardCornersInImage(mat, patternSize, corners);
    
    // Sonuçları NSValue dizisi olarak döndür
    NSMutableArray<NSValue *> *result = [NSMutableArray array];
    if (found) {
        for (const auto &corner : corners) {
            [result addObject:[NSValue valueWithCGPoint:CGPointMake(corner.x, corner.y)]];
        }
    }
    
    return result; // Eğer köşe bulunmadıysa, boş bir dizi döndürülür.
}

+ (NSDictionary<NSString *, NSValue *> *)calibrateCameraWithObjectPoints:(NSArray<NSArray<NSValue *> *> *)objectPoints
                                                             imagePoints:(NSArray<NSArray<NSValue *> *> *)imagePoints
                                                               imageSize:(CGSize)imageSize {
    // Objeleri cv::Point3f formatına dönüştür
    std::vector<std::vector<cv::Point3f>> objPoints;
    for (NSArray<NSValue *> *points in objectPoints) {
        std::vector<cv::Point3f> cvPoints;
        for (NSValue *pointValue in points) {
            CGPoint point = [pointValue CGPointValue];
            cvPoints.emplace_back(point.x, point.y, 0); // Z eksenini sıfır yapıyoruz
        }
        objPoints.push_back(cvPoints);
    }
    
    // Görüntü noktalarını cv::Point2f formatına dönüştür
    std::vector<std::vector<cv::Point2f>> imgPoints;
    for (NSArray<NSValue *> *points in imagePoints) {
        std::vector<cv::Point2f> cvPoints;
        for (NSValue *pointValue in points) {
            CGPoint point = [pointValue CGPointValue];
            cvPoints.emplace_back(point.x, point.y);
        }
        imgPoints.push_back(cvPoints);
    }
    
    // Kalibrasyon parametrelerini ayarla
    cv::Size imgSize(static_cast<int>(imageSize.width), static_cast<int>(imageSize.height));
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    
    // C++ fonksiyonunu çağır
    double rms = calibrateCameraWithObjectPoints(objPoints, imgPoints, imgSize, cameraMatrix, distCoeffs, rvecs, tvecs);
    
    // Sonuçları NSDictionary formatında döndür
    NSDictionary *result = @{
        @"cameraMatrix": [NSValue valueWithBytes:cameraMatrix.data objCType:@encode(cv::Mat)],
        @"distCoeffs": [NSValue valueWithBytes:distCoeffs.data objCType:@encode(cv::Mat)],
        @"rms": @(rms)
    };
    
    return result;
}

+ (UIImage *)undistortImage:(UIImage *)image
           withCameraMatrix:(NSArray<NSNumber *> *)cameraMatrix
                 distCoeffs:(NSArray<NSNumber *> *)distCoeffs {
    // UIImage'den cv::Mat'e çevir
    cv::Mat mat;
    UIImageToMat(image, mat);
    
    // Gri tonlamaya çevir
    cv::Mat gray;
    cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
    
    // Kamera matrisini ve distorsiyon katsayılarını al
    cv::Mat cameraMat(3, 3, CV_64F);
    for (int i = 0; i < 9; i++) {
        cameraMat.at<double>(i / 3, i % 3) = [cameraMatrix[i] doubleValue];
    }
    
    cv::Mat distCoeffsMat(1, (int)distCoeffs.count, CV_64F);
    for (int i = 0; i < distCoeffs.count; i++) {
        distCoeffsMat.at<double>(0, i) = [distCoeffs[i] doubleValue];
    }
    
    // C++ fonksiyonunu çağır
    cv::Mat undistorted;
    undistortImage(gray, undistorted, cameraMat, distCoeffsMat);
    
    // Düzgünleştirilmiş görüntüyü UIImage'a çevir ve döndür
    return MatToUIImage(undistorted);
}

+ (NSDictionary<NSString *, NSValue *> *)solvePnPWithObjectPoints:(NSArray<NSValue *> *)objectPoints
                                                      imagePoints:(NSArray<NSValue *> *)imagePoints
                                                     cameraMatrix:(NSArray<NSNumber *> *)cameraMatrix
                                                       distCoeffs:(NSArray<NSNumber *> *)distCoeffs {
    // 3D objeleri cv::Point3f formatına dönüştür
    std::vector<cv::Point3f> objPoints;
    for (NSValue *value in objectPoints) {
        CGPoint point = [value CGPointValue];
        objPoints.emplace_back(point.x, point.y, 0); // Z ekseni için varsayılan değer 0
    }
    
    // 2D görüntü noktalarını cv::Point2f formatına dönüştür
    std::vector<cv::Point2f> imgPoints;
    for (NSValue *value in imagePoints) {
        CGPoint point = [value CGPointValue];
        imgPoints.emplace_back(point.x, point.y);
    }
    
    // Kamera matrisini oluştur
    cv::Mat camMatrix(3, 3, CV_64F);
    for (int i = 0; i < 9; i++) {
        camMatrix.at<double>(i / 3, i % 3) = [cameraMatrix[i] doubleValue];
    }
    
    // Distorsiyon katsayılarını oluştur
    cv::Mat distCoeffMat(1, (int)distCoeffs.count, CV_64F);
    for (int i = 0; i < distCoeffs.count; i++) {
        distCoeffMat.at<double>(0, i) = [distCoeffs[i] doubleValue];
    }
    
    // Çıktı rotasyon ve translasyon vektörleri
    cv::Mat rvec, tvec;
    
    // C++ fonksiyonunu çağır
    bool success = solvePnPWithObjectPoints(objPoints, imgPoints, camMatrix, distCoeffMat, rvec, tvec);
    
    if (!success) {
        return @{};
    }
    
    // Rotasyon ve translasyon vektörlerini NSDictionary olarak döndür
    return @{
        @"rvec": [NSValue valueWithBytes:rvec.data objCType:@encode(cv::Mat)],
        @"tvec": [NSValue valueWithBytes:tvec.data objCType:@encode(cv::Mat)]
    };
}

+ (NSArray<NSDictionary *> *)decomposeHomographyMatrix:(NSArray<NSNumber *> *)homographyMatrix
                                          cameraMatrix:(NSArray<NSNumber *> *)cameraMatrix {
    // Homografi matrisini oluştur
    cv::Mat homography(3, 3, CV_64F);
    for (int i = 0; i < 9; i++) {
        homography.at<double>(i / 3, i % 3) = [homographyMatrix[i] doubleValue];
    }
    
    // Kamera matrisini oluştur
    cv::Mat camMatrix(3, 3, CV_64F);
    for (int i = 0; i < 9; i++) {
        camMatrix.at<double>(i / 3, i % 3) = [cameraMatrix[i] doubleValue];
    }
    
    // Çıktı için rotasyon, translasyon ve normalleri içeren vektörler
    std::vector<cv::Mat> rotations, translations, normals;
    
    // Homografi matrisini decompose et
    bool success = cv::decomposeHomographyMat(homography, camMatrix, rotations, translations, normals);
    
    if (!success) return @[];
    
    // Rotasyon, translasyon ve normal değerlerini NSDictionary olarak kaydet
    NSMutableArray<NSDictionary *> *output = [NSMutableArray array];
    for (size_t i = 0; i < rotations.size(); i++) {
        NSDictionary *decomposition = @{
            @"rotation": @[
                @(rotations[i].at<double>(0, 0)), @(rotations[i].at<double>(0, 1)), @(rotations[i].at<double>(0, 2)),
                @(rotations[i].at<double>(1, 0)), @(rotations[i].at<double>(1, 1)), @(rotations[i].at<double>(1, 2)),
                @(rotations[i].at<double>(2, 0)), @(rotations[i].at<double>(2, 1)), @(rotations[i].at<double>(2, 2))
            ],
            @"translation": @[
                @(translations[i].at<double>(0, 0)), @(translations[i].at<double>(1, 0)), @(translations[i].at<double>(2, 0))
            ],
            @"normal": @[
                @(normals[i].at<double>(0, 0)), @(normals[i].at<double>(1, 0)), @(normals[i].at<double>(2, 0))
            ]
        };
        [output addObject:decomposition];
    }
    
    return output;
}

+ (NSArray<NSNumber *> *)findEssentialMatrixWithPoints1:(NSArray<NSValue *> *)points1
                                                points2:(NSArray<NSValue *> *)points2
                                           cameraMatrix:(NSArray<NSNumber *> *)cameraMatrix {
    // Noktaları cv::Point2f formatına dönüştür
    std::vector<cv::Point2f> pts1;
    for (NSValue *value in points1) {
        CGPoint point = [value CGPointValue];
        pts1.emplace_back(point.x, point.y);
    }
    
    std::vector<cv::Point2f> pts2;
    for (NSValue *value in points2) {
        CGPoint point = [value CGPointValue];
        pts2.emplace_back(point.x, point.y);
    }
    
    // Kamera matrisini oluştur
    cv::Mat camMatrix(3, 3, CV_64F);
    for (int i = 0; i < 9; i++) {
        camMatrix.at<double>(i / 3, i % 3) = [cameraMatrix[i] doubleValue];
    }
    
    // C++ fonksiyonunu çağırarak Essential matrisi bul
    cv::Mat essentialMat = findEssentialMatrixWithPoints1(pts1, pts2, camMatrix);
    
    // Essential matrisini NSArray<NSNumber *> olarak döndür
    NSMutableArray<NSNumber *> *outputMatrix = [NSMutableArray arrayWithCapacity:essentialMat.rows * essentialMat.cols];
    for (int i = 0; i < essentialMat.rows; i++) {
        for (int j = 0; j < essentialMat.cols; j++) {
            [outputMatrix addObject:@(essentialMat.at<double>(i, j))];
        }
    }
    return outputMatrix;
}
+ (NSDictionary *)decomposeEssentialMatrix:(NSArray<NSNumber *> *)essentialMatrix {
    // Essential matrisini oluştur
    cv::Mat essentialMat(3, 3, CV_64F);
    for (int i = 0; i < 9; i++) {
        essentialMat.at<double>(i / 3, i % 3) = [essentialMatrix[i] doubleValue];
    }
    
    // Rotasyon ve çeviri bileşenlerini saklamak için Mat nesneleri
    cv::Mat rotation1, rotation2, translation;
    
    // Essential matrisini ayrıştır
    cv::decomposeEssentialMat(essentialMat, rotation1, rotation2, translation);
    
    // Rotasyon ve çeviri vektörlerini NSDictionary olarak kaydet
    NSDictionary *decomposition = @{
        @"rotation1": @[
            @(rotation1.at<double>(0, 0)), @(rotation1.at<double>(0, 1)), @(rotation1.at<double>(0, 2)),
            @(rotation1.at<double>(1, 0)), @(rotation1.at<double>(1, 1)), @(rotation1.at<double>(1, 2)),
            @(rotation1.at<double>(2, 0)), @(rotation1.at<double>(2, 1)), @(rotation1.at<double>(2, 2))
        ],
        @"rotation2": @[
            @(rotation2.at<double>(0, 0)), @(rotation2.at<double>(0, 1)), @(rotation2.at<double>(0, 2)),
            @(rotation2.at<double>(1, 0)), @(rotation2.at<double>(1, 1)), @(rotation2.at<double>(1, 2)),
            @(rotation2.at<double>(2, 0)), @(rotation2.at<double>(2, 1)), @(rotation2.at<double>(2, 2))
        ],
        @"translation": @[
            @(translation.at<double>(0)), @(translation.at<double>(1)), @(translation.at<double>(2))
        ]
    };
    
    return decomposition;
}

+ (NSArray<NSDictionary *> *)decomposeHomography:(NSArray<NSNumber *> *)homographyMatrix {
    // 3x3 homografi matrisini OpenCV formatında hazırlayın
    cv::Mat homography(3, 3, CV_64F);
    for (int i = 0; i < 9; i++) {
        homography.at<double>(i / 3, i % 3) = [homographyMatrix[i] doubleValue];
    }
    
    // C++ yardımcı işlevini çağırın
    auto decomposedResults = decomposeHomography(homography);
    
    // Sonuçları Swift'in anlayacağı formata çevirin
    NSMutableArray<NSDictionary *> *result = [NSMutableArray array];
    for (const auto &decomposition : decomposedResults) {
        NSMutableDictionary *decompositionDict = [NSMutableDictionary dictionary];
        
        // Dönüşüm matrisini Swift için diziye çevirin
        NSMutableArray *rotationArray = [NSMutableArray array];
        cv::Mat rotation = std::get<0>(decomposition);
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                [rotationArray addObject:@(rotation.at<double>(r, c))];
            }
        }
        decompositionDict[@"rotation"] = rotationArray;
        
        // Translation vektörünü diziye çevirin
        NSMutableArray *translationArray = [NSMutableArray array];
        cv::Mat translation = std::get<1>(decomposition);
        for (int t = 0; t < 3; t++) {
            [translationArray addObject:@(translation.at<double>(t, 0))];
        }
        decompositionDict[@"translation"] = translationArray;
        
        // Normal vektörünü diziye çevirin
        NSMutableArray *normalArray = [NSMutableArray array];
        cv::Mat normal = std::get<2>(decomposition);
        for (int n = 0; n < 3; n++) {
            [normalArray addObject:@(normal.at<double>(n, 0))];
        }
        decompositionDict[@"normal"] = normalArray;
        
        [result addObject:decompositionDict];
    }
    return result;
}

+ (NSArray<NSNumber *> *)findEssentialMatWithPoints1:(NSArray<NSArray<NSNumber *> *> *)points1
                                             points2:(NSArray<NSArray<NSNumber *> *> *)points2 {
    
    // Swift'ten gelen NSArray'leri cv::Mat'e dönüştürme
    cv::Mat points1Mat = cv::Mat((int)points1.count, (int)points1[0].count, CV_64F);
    cv::Mat points2Mat = cv::Mat((int)points2.count, (int)points2[0].count, CV_64F);
    
    for (int i = 0; i < points1.count; i++) {
        for (int j = 0; j < points1[0].count; j++) {
            points1Mat.at<double>(i, j) = [points1[i][j] doubleValue];
            points2Mat.at<double>(i, j) = [points2[i][j] doubleValue];
        }
    }
    cv::Mat essentialMat = cv::findEssentialMat(points1Mat, points2Mat);
    
    // cv::Mat'i NSArray'e dönüştürme
    NSMutableArray<NSNumber *> *result = [NSMutableArray array];
    for (int i = 0; i < essentialMat.rows; i++) {
        for (int j = 0; j < essentialMat.cols; j++) {
            [result addObject:@(essentialMat.at<double>(i, j))];
        }
    }
    return result;
}

+ (NSArray<NSArray<NSNumber *> *> *)decomposeEssentialMatWithEssentialMat:(NSArray<NSArray<NSNumber *> *> *)essentialMatArray {
    // Swift'ten gelen NSArray'leri cv::Mat'e dönüştürme
    cv::Mat essentialMat = cv::Mat((int)essentialMatArray.count, (int)essentialMatArray[0].count, CV_64F);
    
    for (int i = 0; i < essentialMatArray.count; i++) {
        for (int j = 0; j < essentialMatArray[0].count; j++) {
            essentialMat.at<double>(i, j) = [essentialMatArray[i][j] doubleValue];
        }
    }
    
    // C++ fonksiyonunu çağırarak dekompoze et
    cv::Mat R1, R2, t;
    std::tie(R1, R2, t) = decomposeEssentialMatWithEssentialMat(essentialMat);
    
    // cv::Mat'leri NSArray'e dönüştürme
    NSMutableArray<NSArray<NSNumber *> *> *result = [NSMutableArray array];
    
    // R1
    NSMutableArray<NSNumber *> *r1Array = [NSMutableArray array];
    for (int i = 0; i < R1.rows; i++) {
        for (int j = 0; j < R1.cols; j++) {
            [r1Array addObject:@(R1.at<double>(i, j))];
        }
    }
    [result addObject:r1Array];
    
    // R2
    NSMutableArray<NSNumber *> *r2Array = [NSMutableArray array];
    for (int i = 0; i < R2.rows; i++) {
        for (int j = 0; j < R2.cols; j++) {
            [r2Array addObject:@(R2.at<double>(i, j))];
        }
    }
    [result addObject:r2Array];
    
    // t
    NSMutableArray<NSNumber *> *tArray = [NSMutableArray array];
    for (int i = 0; i < t.rows; i++) {
        for (int j = 0; j < t.cols; j++) {
            [tArray addObject:@(t.at<double>(i, j))];
        }
    }
    [result addObject:tArray];
    
    return result;
}
// 11-Video Processing

+ (UIImage *)captureFrameFromCameraIndex:(int)cameraIndex {
    cv::Mat frame = captureFrameFromCameraIndex(cameraIndex); // C++ fonksiyonunu çağır
    
    if (frame.empty()) {  // Eğer çerçeve boşsa nil döner
        return nil;
    }
    
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);  // BGR'den RGB'ye dönüştür
    
    NSData *data = [NSData dataWithBytes:frame.data length:frame.elemSize() * frame.total()];
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGImageRef cgImage = CGImageCreate(frame.cols, frame.rows, 8, 24, frame.step[0], colorSpace, kCGImageAlphaNone | kCGBitmapByteOrderDefault, provider, NULL, NO, kCGRenderingIntentDefault);
    
    UIImage *image = [UIImage imageWithCGImage:cgImage];
    
    // Bellek yönetimi
    CGImageRelease(cgImage);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return image;
}

+ (BOOL)writeVideoFromImages:(NSArray<UIImage *> *)images toFilePath:(NSString *)filePath fps:(int)fps {
    if (images.count == 0) {
        NSLog(@"No images to write.");
        return NO;
    }
    
    std::vector<cv::Mat> matImages;
    
    for (UIImage *image in images) {
        cv::Mat matImage;
        [self UIImageToMat:image mat:matImage];
        matImages.push_back(matImage);
    }
    
    std::string cppFilePath = [filePath UTF8String];
    return writeVideoFromImages(matImages, cppFilePath, fps);
}




















//// UIImage'ı cv::Mat formatına çevirme
//+ (cv::Mat)cvMatFromUIImage:(UIImage *)image {
//    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
//    CGFloat width = image.size.width;
//    CGFloat height = image.size.height;
//
//    cv::Mat cvMat(height, width, CV_8UC3); // 4 kanallı RGBA formatı
//
//    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data, width, height, 8, cvMat.step[0], colorSpace, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrderDefault);
//
//    if (contextRef == NULL) {
//        NSLog(@"Error: Failed to create CGContext");
//        return cv::Mat(); // Boş matris döndür
//    }
//
//    CGContextDrawImage(contextRef, CGRectMake(0, 0, width, height), image.CGImage);
//    CGContextRelease(contextRef);
//
//    // RGBA formatını BGR'ye çevir (OpenCV'nin varsayılanı BGR'dir)
//    cv::cvtColor(cvMat, cvMat, cv::COLOR_RGBA2BGR);
//
//    return cvMat;
//}

//+ (cv::Mat)cvMatFromUIImage:(UIImage *)image {
//    if (!image) {
//        NSLog(@"Hata: Dönüştürmek için geçerli bir UIImage sağlanmadı.");
//        return cv::Mat(); // Boş bir cv::Mat döndürür
//    }
//    
//    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
//    CGFloat cols = image.size.width;
//    CGFloat rows = image.size.height;
//    
//    // cv::Mat türünü CV_8UC3 yapıyoruz (RGB formatı için)
//    cv::Mat mat(rows, cols, CV_8UC3);
//    CGContextRef contextRef = CGBitmapContextCreate(mat.data, cols, rows, 8, mat.step[0], colorSpace, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
//    
//    if (!contextRef) {
//        NSLog(@"Hata: CGContext oluşturulamadı.");
//        CGColorSpaceRelease(colorSpace);
//        return cv::Mat(); // Boş bir cv::Mat döndürür
//    }
//    
//    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
//    CGContextRelease(contextRef);
//    CGColorSpaceRelease(colorSpace);
//    
//    if (mat.empty()) {
//        NSLog(@"Hata: UIImage cv::Mat'e dönüştürülemedi.");
//    }
//    
//    return mat;
//}

//+ (cv::Mat)cvMatFromUIImage:(UIImage *)image {
//    if (!image) {
//        NSLog(@"Hata: Geçersiz UIImage.");
//        return cv::Mat();
//    }
//
//    // RGB renk alanı oluşturuyoruz
//    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
//    CGFloat width = image.size.width;
//    CGFloat height = image.size.height;
//
//    // CV_8UC3 formatında (RGB) cv::Mat oluştur
//    cv::Mat mat(height, width, CV_8UC3);
//
//    // Bitmap Context oluşturma
//    CGContextRef contextRef = CGBitmapContextCreate(mat.data,
//                                                    width,
//                                                    height,
//                                                    8,
//                                                    mat.step[0],
//                                                    colorSpace,
//                                                    kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
//
//    CGColorSpaceRelease(colorSpace);
//
//    if (!contextRef) {
//        NSLog(@"Hata: CGContext oluşturulamadı.");
//        return cv::Mat();
//    }
//
//    // Görüntüyü CGContext’e çiz
//    CGContextDrawImage(contextRef, CGRectMake(0, 0, width, height), image.CGImage);
//    CGContextRelease(contextRef);
//
//    if (mat.empty()) {
//        NSLog(@"Hata: UIImage cv::Mat'e dönüştürülemedi.");
//    }
//
//    return mat;
//}

+ (cv::Mat)cvMatFromUIImage:(UIImage *)image {
    // UIImage'yi doğrudan CGImage'ye çevir
    CGImageRef imageRef = image.CGImage;
    if (!imageRef) {
        NSLog(@"Hata: UIImage'den CGImage alınamadı.");
        return cv::Mat();
    }

    // CGImage boyutlarını alın
    size_t width = CGImageGetWidth(imageRef);
    size_t height = CGImageGetHeight(imageRef);

    // CV_8UC3 formatında cv::Mat oluştur
    cv::Mat mat(height, width, CV_8UC3);

    // Renk uzayı ayarı
    CGContextRef contextRef = CGBitmapContextCreate(mat.data,
                                                    width,
                                                    height,
                                                    8,
                                                    mat.step[0],
                                                    CGColorSpaceCreateDeviceRGB(),
                                                    kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);

    if (!contextRef) {
        NSLog(@"Hata: CGContext oluşturulamadı.");
        return cv::Mat();
    }

    // Görüntüyü CGContext'e çizme
    CGContextDrawImage(contextRef, CGRectMake(0, 0, width, height), imageRef);
    CGContextRelease(contextRef);

    if (mat.empty()) {
        NSLog(@"Hata: UIImage cv::Mat'e dönüştürülemedi.");
    }

    return mat;
}




+ (UIImage *)UIImageFromCVMat:(cv::Mat)cvMat {
    if (cvMat.empty()) {
        NSLog(@"Error: cv::Mat is empty");
        return nil;
    }
    
    // BGR -> RGBA'ya çevrilecek (OpenCV'den iOS'a uyum için)
    if (cvMat.channels() == 3) {
        cv::cvtColor(cvMat, cvMat, cv::COLOR_BGR2RGBA); // BGR'den RGBA'ya dönüşüm
    }
    
    if (cvMat.type() != CV_8UC4) {
        NSLog(@"Error: Unsupported cv::Mat type after conversion: %d", cvMat.type());
        return nil;
    }
    
    size_t dataLength = cvMat.elemSize() * cvMat.total();
    NSData *data = [NSData dataWithBytes:cvMat.data length:dataLength];
    
    if (data == nil || data.length != dataLength) {
        NSLog(@"Error: Failed to create NSData from cv::Mat data");
        return nil;
    }
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    if (provider == NULL) {
        NSLog(@"Error: Failed to create CGDataProviderRef");
        return nil;
    }
    
    CGImageRef imageRef = CGImageCreate(cvMat.cols, cvMat.rows, 8, 32, cvMat.step[0], colorSpace, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault, provider, NULL, false, kCGRenderingIntentDefault);
    
    if (imageRef == NULL) {
        NSLog(@"Error: Failed to create CGImageRef");
        CGDataProviderRelease(provider);
        CGColorSpaceRelease(colorSpace);
        return nil;
    }
    
    UIImage *image = [UIImage imageWithCGImage:imageRef];
    
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return image;
}

// UIImage'ten cv::Mat'e dönüştürme fonksiyonu
void UIImageToMat(UIImage *image, cv::Mat &mat) {
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // RGBA
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data, cols, rows, 8, cvMat.step[0], colorSpace, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    
    mat = cvMat;
}

// cv::Mat'ten UIImage'e dönüştürme fonksiyonu
UIImage *MatToUIImage(const cv::Mat &mat) {
    NSData *data = [NSData dataWithBytes:mat.data length:mat.elemSize() * mat.total()];
    CGColorSpaceRef colorSpace;
    
    if (mat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGImageRef imageRef = CGImageCreate(mat.cols, mat.rows, 8, 8 * mat.elemSize(), mat.step[0], colorSpace, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault, provider, NULL, false, kCGRenderingIntentDefault);
    UIImage *image = [UIImage imageWithCGImage:imageRef];
    
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return image;
}

+ (void)UIImageToMat:(const UIImage *)image mat:(cv::Mat&)mat {
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    int widthInt = (int)round(cols);
    int heightInt = (int)round(rows);
    
    cv::Mat tmp(rows, cols, CV_8UC4); // RGBA formatı
    CGContextRef contextRef = CGBitmapContextCreate(tmp.data, cols, rows, 8, tmp.step[0], colorSpace, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrderDefault);
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    cv::cvtColor(tmp, mat, cv::COLOR_RGBA2BGR); // RGBA'dan BGR'a dönüşüm
}


// resizeAndGrayColor bu olmadan çalışmıyor.
cv::Mat UIImageToMat(UIImage *image) {
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    cv::Mat mat(rows, cols, CV_8UC4); // 4 kanal (RGBA) olarak oluşturulur
    CGContextRef contextRef = CGBitmapContextCreate(mat.data, cols, rows, 8, mat.step[0], colorSpace, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    return mat;
}


@end
