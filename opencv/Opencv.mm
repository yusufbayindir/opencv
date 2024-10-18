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

#import "Opencv.h"
#import <Foundation/Foundation.h>
#import <vector>
#import <opencv2/opencv.hpp>

@implementation Opencv
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
