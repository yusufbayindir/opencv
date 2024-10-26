//
//  opencv.h
//  opencv
//
//  Created by Yusuf Bayindir on 10/18/24.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface Opencv : NSObject

// 1-Basic Image Operations

+ (UIImage *)loadImage:(NSString *)filePath;

+ (BOOL)saveImage:(UIImage *)image
       toFilePath:(NSString *)filePath;

+ (void)saveImageToGallery:(UIImage *)image;

// resize ve gray color işlemini yapar.
+ (UIImage *)resizeAndGrayColor:(UIImage *)image
                            toSize:(CGSize)size;

+ (UIImage *)makeBorderWithImage:(UIImage *)image
                             top:(int)top
                          bottom:(int)bottom
                            left:(int)left
                           right:(int)right
                      borderType:(int)borderType
                           color:(UIColor *)color;

+ (UIImage *)flipImage:(UIImage *)image
              flipCode:(int)flipCode;

+ (UIImage *)processAndShowImage:(UIImage *)image;

+ (NSArray<UIImage *> *)splitImage:(UIImage *)image;

// 2-Geometric Transformations

+ (UIImage *)rotateImage:(UIImage *)image
                  center:(CGPoint)center
                   angle:(double)angle
                   scale:(double)scale;

+ (UIImage *)applyWarpAffineToImage:(UIImage *)image
                             matrix:(NSArray<NSNumber *> *)matrix;

+ (UIImage *)warpPerspectiveImage:(UIImage *)image
                       srcPoints:(NSArray<NSValue *> *)srcPoints
                       dstPoints:(NSArray<NSValue *> *)dstPoints;

+ (UIImage *)applyPerspectiveTransform:(UIImage *)image
                             srcPoints:(NSArray<NSValue *> *)srcPoints
                             dstPoints:(NSArray<NSValue *> *)dstPoints;

+ (UIImage *)transposeImage:(UIImage *)image;

// 3-Drawing Functions

+ (UIImage *)drawLineOnImage:(UIImage *)image
                       start:(CGPoint)start
                         end:(CGPoint)end
                       color:(UIColor *)color
                   thickness:(int)thickness;

+ (UIImage *)drawCircleOnImage:(UIImage *)image
                     atPoint:(CGPoint)center
                     withRadius:(int)radius
                     andColor:(UIColor *)color
                     lineWidth:(int)lineWidth;

+ (UIImage *)drawRectangleOnImage:(UIImage *)image
                         fromPoint:(CGPoint)topLeft
                         toPoint:(CGPoint)bottomRight
                         withColor:(UIColor *)color
                         lineWidth:(int)lineWidth;

+ (UIImage *)drawEllipseOnImage:(UIImage *)image
                         center:(CGPoint)center
                           axes:(CGSize)axes
                          angle:(double)angle
                     startAngle:(double)startAngle
                       endAngle:(double)endAngle
                          color:(UIColor *)color
                      thickness:(int)thickness;

+ (UIImage *)addTextToUIImage:(UIImage *)image
                         text:(NSString *)text
                     position:(CGPoint)position
                     fontFace:(int)fontFace
                    fontScale:(double)fontScale
                        color:(UIColor *)color
                    thickness:(int)thickness
                     lineType:(int)lineType;

+ (UIImage *)fillPolygonOnImage:(UIImage *)image
                     withPoints:(NSArray<NSValue *> *)points
                     andColor:(UIColor *)color;

+ (UIImage *)drawPolylinesOnImage:(UIImage *)image
                     withPoints:(NSArray<NSValue *> *)points
                     andColor:(UIColor *)color
                     lineWidth:(int)lineWidth;

// 4-Thresholding and Edge Detection

+ (UIImage *)applySobelToUIImage:(UIImage *)image
                             dx:(int)dx
                             dy:(int)dy
                     kernelSize:(int)kernelSize;

// 5-Image Filtering

+ (UIImage *)gaussianBlur:(UIImage *)image
           withKernelSize:(CGSize)kernelSize
                    sigma:(double)sigma;

+ (UIImage *)medianBlur:(UIImage *)image
         withKernelSize:(int)kernelSize;

+ (UIImage *)blur:(UIImage *)image
   withKernelSize:(CGSize)kernelSize;

+ (UIImage *)applyBilateralFilterToImage:(UIImage *)image
                                diameter:(int)diameter
                              sigmaColor:(double)sigmaColor
                              sigmaSpace:(double)sigmaSpace;

+ (UIImage *)applyFilter2DToImage:(UIImage *)image
                           kernel:(NSArray<NSArray<NSNumber *> *> *)kernel;


@end
NS_ASSUME_NONNULL_END
