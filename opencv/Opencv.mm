//
//  opencv.mm
//  opencv
//
//  Created by Yusuf Bayindir on 10/18/24.
//

#import <opencv2/opencv.hpp>
#include <string>
#include "Opencv.h"

cv::Mat addTextToImage(const cv::Mat& image, const std::string& text, cv::Point position, int fontFace, double fontScale, cv::Scalar color, int thickness, int lineType) {
    // Görüntünün bir kopyasını oluştur
    cv::Mat newImage = image.clone();
    
    // Yazıyı kopyalanan görüntüye ekle
    cv::putText(newImage, text, position, fontFace, fontScale, color, thickness, lineType);
    
    // Yeni görüntüyü döndür
    return newImage;
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
    return [self MatToUIImage:matImage];
}

+ (BOOL)saveImage:(UIImage *)image toFilePath:(NSString *)filePath {
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];

    try {
        cv::imwrite([filePath UTF8String], matImage);
        return YES;
    } catch (const cv::Exception &e) {
        NSLog(@"OpenCV error: %s", e.what());
        return NO;
    }
}

+ (void)saveImageToGallery:(UIImage *)image {
    UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil);
}


+ (UIImage *)resizeAndGrayColor:(UIImage *)image
                            toSize:(CGSize)size {
    cv::Mat cvImage;
    [self UIImageToMat:image mat:cvImage];

    cv::Mat resizedImage;
    cv::resize(cvImage, resizedImage, cv::Size(size.width, size.height));

    cv::Mat convertedImage;
    cv::cvtColor(resizedImage, convertedImage, cv::COLOR_BGR2GRAY);

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
    
    // Apply copyMakeBorder
    cv::Mat borderedImage;
    cv::copyMakeBorder(cvImage, borderedImage, top, bottom, left, right, borderType, borderColor);
    
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

    // cv::flip uygulaması
    cv::Mat flippedImage;
    cv::flip(matImage, flippedImage, flipCode);

    // cv::Mat'i UIImage'e çevir
    return [self MatToUIImage:flippedImage];
}

+ (UIImage *)processAndShowImage:(UIImage *)image {
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];

    // Burada görüntü işlemleri yapılabilir, örneğin gri tonlama.
    cv::Mat processedImage;
    cv::cvtColor(matImage, processedImage, cv::COLOR_BGR2GRAY);

    return [self MatToUIImage:processedImage];
}

+ (NSArray<UIImage *> *)splitImage:(UIImage *)image {
    // UIImage'den cv::Mat'e dönüşüm
    cv::Mat mat;
    CGImageRef imageRef = image.CGImage;
    mat = [self cvMatFromUIImage:image];

    // Bileşenlere ayırma
    std::vector<cv::Mat> channels;
    cv::split(mat, channels);

    // cv::Mat'leri UIImage'ye çevirme
    NSMutableArray<UIImage *> *resultImages = [NSMutableArray array];
    for (const auto& channel : channels) {
        [resultImages addObject:[self MatToUIImage:channel]];
    }

    return resultImages;
}


// 2-Geometric Transformations

+ (UIImage *)rotateImage:(UIImage *)image center:(CGPoint)center angle:(double)angle scale:(double)scale {
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];

    cv::Point2f cvCenter(center.x, center.y);
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(cvCenter, angle, scale);

    cv::Mat rotatedImage;
    cv::warpAffine(matImage, rotatedImage, rotationMatrix, matImage.size());

    return [self MatToUIImage:rotatedImage];
}

+ (UIImage *)applyWarpAffineToImage:(UIImage *)image matrix:(NSArray<NSNumber *> *)matrix {
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];

    // Matristen dönüşüm matrisini oluştur
    cv::Mat warpMat = (cv::Mat_<double>(2, 3) << matrix[0].doubleValue, matrix[1].doubleValue, matrix[2].doubleValue,
                                                  matrix[3].doubleValue, matrix[4].doubleValue, matrix[5].doubleValue);

    cv::Mat warpedImage;
    cv::warpAffine(matImage, warpedImage, warpMat, matImage.size());

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
    
    // Perspektif dönüşüm matrisini al
    cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(src.data(), dst.data());
    
    // Görüntüyü perspektif olarak dönüştür
    cv::Mat warpedImage;
    cv::warpPerspective(mat, warpedImage, perspectiveMatrix, mat.size());
    
    return [self MatToUIImage:warpedImage];
}

+ (UIImage *)applyPerspectiveTransform:(UIImage *)image srcPoints:(NSArray<NSValue *> *)srcPoints dstPoints:(NSArray<NSValue *> *)dstPoints {
   cv::Mat matImage;
   [self UIImageToMat:image mat:matImage];

   // Dönüşüm noktalarını al
   cv::Point2f src[4];
   cv::Point2f dst[4];

   for (int i = 0; i < 4; i++) {
       src[i] = cv::Point2f([srcPoints[i] CGPointValue].x, [srcPoints[i] CGPointValue].y);
       dst[i] = cv::Point2f([dstPoints[i] CGPointValue].x, [dstPoints[i] CGPointValue].y);
   }

   // Perspektif dönüşüm matrisini hesapla
   cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(src, dst);

   // Perspektif dönüşümünü uygula
   cv::Mat transformedImage;
   cv::warpPerspective(matImage, transformedImage, perspectiveMatrix, matImage.size());

   return [self MatToUIImage:transformedImage];
}

+ (UIImage *)transposeImage:(UIImage *)image {
    // UIImage'ı cv::Mat formatına çevirin
    cv::Mat mat;
    UIImageToMat(image, mat);

    // Transpoz fonksiyonunu uygulayın
    cv::Mat transposedMat;
    cv::transpose(mat, transposedMat);

    // cv::Mat'ten UIImage formatına geri dönüştürün
    return MatToUIImage(transposedMat);
}

// 3-Drawing Functions

+ (UIImage *)drawLineOnImage:(UIImage *)image start:(CGPoint)start end:(CGPoint)end color:(UIColor *)color thickness:(int)thickness {
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];

    cv::Scalar lineColor;
    [self UIColorToScalar:color scalar:lineColor];

    cv::Point cvStart(start.x, start.y);
    cv::Point cvEnd(end.x, end.y);

    cv::line(matImage, cvStart, cvEnd, lineColor, thickness);

    return [self MatToUIImage:matImage];
}

+ (UIImage *)drawCircleOnImage:(UIImage *)image
                     atPoint:(CGPoint)center
                    withRadius:(int)radius
                     andColor:(UIColor *)color
                     lineWidth:(int)lineWidth {
    // UIImage'den cv::Mat'e dönüşüm
    cv::Mat mat = [self UIImageToMat:image];

    // Çizgi rengini UIColor'dan cv::Scalar'a çevirme
    CGFloat r, g, b, a;
    [color getRed:&r green:&g blue:&b alpha:&a];
    cv::Scalar circleColor(b * 255, g * 255, r * 255); // OpenCV BGR kullanır

    // Daire çizme
    cv::Point centerPoint(cvRound(center.x), cvRound(center.y));
    cv::circle(mat, centerPoint, radius, circleColor, lineWidth, cv::LINE_AA);

    // cv::Mat'ten UIImage'e dönüşüm
    return [self MatToUIImage:mat];
}

+ (UIImage *)drawRectangleOnImage:(UIImage *)image
                         fromPoint:(CGPoint)topLeft
                         toPoint:(CGPoint)bottomRight
                         withColor:(UIColor *)color
                         lineWidth:(int)lineWidth {
    // UIImage'den cv::Mat'e dönüşüm
    cv::Mat mat = [self UIImageToMat:image];

    // Çizgi rengini UIColor'dan cv::Scalar'a çevirme
    CGFloat r, g, b, a;
    [color getRed:&r green:&g blue:&b alpha:&a];
    cv::Scalar rectangleColor(b * 255, g * 255, r * 255); // OpenCV BGR kullanır

    // Dikdörtgen çizme
    cv::Point topLeftPoint(cvRound(topLeft.x), cvRound(topLeft.y));
    cv::Point bottomRightPoint(cvRound(bottomRight.x), cvRound(bottomRight.y));
    cv::rectangle(mat, topLeftPoint, bottomRightPoint, rectangleColor, lineWidth, cv::LINE_AA);

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

+ (UIImage *)drawEllipseOnImage:(UIImage *)image center:(CGPoint)center axes:(CGSize)axes angle:(double)angle startAngle:(double)startAngle endAngle:(double)endAngle color:(UIColor *)color thickness:(int)thickness {
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];

    cv::Scalar ellipseColor;
    [self UIColorToScalar:color scalar:ellipseColor];

    cv::Point cvCenter(center.x, center.y);
    cv::Size cvAxes(axes.width, axes.height);

    cv::ellipse(matImage, cvCenter, cvAxes, angle, startAngle, endAngle, ellipseColor, thickness);

    return [self MatToUIImage:matImage];
}

+ (void)UIColorToScalar:(UIColor *)color scalar:(cv::Scalar&)scalar {
    CGFloat r, g, b, a;
    [color getRed:&r green:&g blue:&b alpha:&a];
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

    // Çokgeni doldurma
    std::vector<std::vector<cv::Point>> fillPts;
    fillPts.push_back(pts);
    cv::fillPoly(mat, fillPts, fillColor);

    // cv::Mat'ten UIImage'e dönüşüm
    return [self MatToUIImage:mat];
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

    // Poligonu çizme
    const int isClosed = 0; // Kapalı değil
    cv::polylines(mat, pts, isClosed, polylineColor, lineWidth, cv::LINE_AA);

    // cv::Mat'ten UIImage'e dönüşüm
    return [self MatToUIImage:mat];
}

// 4-Thresholding and Edge Detection

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

// 5-Image Filtering

+ (UIImage *)gaussianBlur:(UIImage *)image withKernelSize:(CGSize)kernelSize sigma:(double)sigma {
    // UIImage'ı cv::Mat'e dönüştürme
    cv::Mat srcMat;
    UIImageToMat(image, srcMat);

    // Çıktı matrisi
    cv::Mat dstMat;

    // GaussianBlur fonksiyonunu uygulama
    cv::GaussianBlur(srcMat, dstMat, cv::Size(kernelSize.width, kernelSize.height), sigma);

    // Sonucu UIImage'a dönüştürme
    return MatToUIImage(dstMat);
}

+ (UIImage *)medianBlur:(UIImage *)image withKernelSize:(int)kernelSize {
   // UIImage'ı cv::Mat'e dönüştürme
   cv::Mat srcMat;
   UIImageToMat(image, srcMat);

   // Çıktı matrisi
   cv::Mat dstMat;

   // MedianBlur fonksiyonunu uygulama
   cv::medianBlur(srcMat, dstMat, kernelSize);

   // Sonucu UIImage'a dönüştürme
   return MatToUIImage(dstMat);
}

+ (UIImage *)blur:(UIImage *)image withKernelSize:(CGSize)kernelSize {
    // UIImage'ı cv::Mat'e dönüştürme
    cv::Mat srcMat;
    UIImageToMat(image, srcMat);

    // Çıktı matrisi
    cv::Mat dstMat;

    // Blur fonksiyonunu uygulama
    cv::blur(srcMat, dstMat, cv::Size(kernelSize.width, kernelSize.height));

    // Sonucu UIImage'a dönüştürme
    return MatToUIImage(dstMat);
}

+ (UIImage *)applyBilateralFilterToImage:(UIImage *)image diameter:(int)diameter sigmaColor:(double)sigmaColor sigmaSpace:(double)sigmaSpace {
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];

    cv::Mat filteredImage;
    cv::bilateralFilter(matImage, filteredImage, diameter, sigmaColor, sigmaSpace);

    return [self MatToUIImage:filteredImage];
}

+ (UIImage *)applyFilter2DToImage:(UIImage *)image kernel:(NSArray<NSArray<NSNumber *> *> *)kernel {
    cv::Mat matImage;
    [self UIImageToMat:image mat:matImage];

    // Çekirdek matrisini oluştur
    int rows = (int)kernel.count;
    int cols = (int)[kernel[0] count];
    cv::Mat kernelMat(rows, cols, CV_32F);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            kernelMat.at<float>(i, j) = [kernel[i][j] floatValue];
        }
    }

    cv::Mat filteredImage;
    cv::filter2D(matImage, filteredImage, -1, kernelMat);

    return [self MatToUIImage:filteredImage];
}


// UIImage'ı cv::Mat formatına çevirme
+ (cv::Mat)cvMatFromUIImage:(UIImage *)image {
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat width = image.size.width;
    CGFloat height = image.size.height;

    cv::Mat cvMat(height, width, CV_8UC4); // 4 kanallı RGBA formatı

    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data, width, height, 8, cvMat.step[0], colorSpace, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrderDefault);

    if (contextRef == NULL) {
        NSLog(@"Error: Failed to create CGContext");
        return cv::Mat(); // Boş matris döndür
    }

    CGContextDrawImage(contextRef, CGRectMake(0, 0, width, height), image.CGImage);
    CGContextRelease(contextRef);

    // RGBA formatını BGR'ye çevir (OpenCV'nin varsayılanı BGR'dir)
    cv::cvtColor(cvMat, cvMat, cv::COLOR_RGBA2BGR);

    return cvMat;
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


@end
