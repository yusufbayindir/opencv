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

+ (UIImage *)processAndShowImage:(UIImage *)image;

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

+ (UIImage *)bitwiseAndWithImage1:(UIImage *)image1 image2:(UIImage *)image2;

+ (UIImage *)bitwiseOrImage:(UIImage *)image1 withImage:(UIImage *)image2;

+ (UIImage *)bitwiseNotWithImage:(UIImage *)image;

+ (UIImage *)addWeightedWithImage1:(UIImage *)image1
                            image2:(UIImage *)image2
                             alpha:(double)alpha
                              beta:(double)beta
                             gamma:(double)gamma;

+ (NSArray<UIImage *> *)splitImage:(UIImage *)image;

//+ (UIImage *)mergeWithChannel1:(UIImage *)channel1
//                      channel2:(UIImage *)channel2
//                      channel3:(UIImage *)channel3;
+ (UIImage *)mergeChannels:(UIImage *)imageR G:(UIImage *)imageG B:(UIImage *)imageB;

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

+ (nullable NSArray<NSNumber *> *)getAffineTransformWithSourcePoints:(NSArray<NSValue *> *)sourcePoints
                                                   destinationPoints:(NSArray<NSValue *> *)destinationPoints;

+ (UIImage *)applyPerspectiveTransform:(UIImage *)image
                             srcPoints:(NSArray<NSValue *> *)srcPoints
                             dstPoints:(NSArray<NSValue *> *)dstPoints;

+ (UIImage *)remapImage:(UIImage *)image;


+ (UIImage *)transposeImage:(UIImage *)image;

+ (nullable UIImage *)pyrUpWithImage:(UIImage *)image;

+ (nullable UIImage *)pyrDownWithImage:(UIImage *)image;

+ (void)resizeWindowWithName:(NSString *)windowName
                       width:(int)width
                      height:(int)height;

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

+ (UIImage *)applyThresholdToImage:(UIImage *)image
                         threshold:(double)threshold
                          maxValue:(double)maxValue
                     thresholdType:(int)thresholdType;

+ (UIImage *)applyAdaptiveThresholdToImage:(UIImage *)image
                                  maxValue:(double)maxValue
                            adaptiveMethod:(int)adaptiveMethod
                             thresholdType:(int)thresholdType
                                 blockSize:(int)blockSize
                                         C:(double)C;

+ (UIImage *)applyCannyToImage:(UIImage *)image threshold1:(double)threshold1 threshold2:(double)threshold2;

//+ (UIImage *)applySobelToUIImage:(UIImage *)image
//                              dx:(int)dx
//                              dy:(int)dy
//                      kernelSize:(int)kernelSize;

+ (UIImage *)applySobelFilterToImage:(UIImage *)image ddepth:(int)ddepth dx:(int)dx dy:(int)dy ksize:(int)ksize;

+ (nullable UIImage *)laplacianWithImage:(UIImage *)image
                              kernelSize:(int)kernelSize;

+ (UIImage *)inRangeWithImage:(UIImage *)image
              lowerBound:(NSArray<NSNumber *> *)lower
              upperBound:(NSArray<NSNumber *> *)upper;

+ (NSArray<NSValue *> *)findNonZeroWithImage:(UIImage *)image;

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

+ (UIImage *)applyBoxFilterToImage:(UIImage *)image
                            ddepth:(int)ddepth
                             ksize:(CGSize)ksize;

+ (UIImage *)applyScharrOnImage:(UIImage *)image;

+ (UIImage *)addImage:(UIImage *)image1
            withImage:(UIImage *)image2;

+ (UIImage *)subtractImage:(UIImage *)image1
                 fromImage:(UIImage *)image2;

+ (UIImage *)multiplyImage:(UIImage *)image1
                 withImage:(UIImage *)image2;

+ (UIImage *)divideImage:(UIImage *)image1
                 byImage:(UIImage *)image2;

// 6-Morphological Operations

+ (UIImage *)erodeImage:(UIImage *)image
         withKernelSize:(int)kernelSize;

+ (UIImage *)dilateImage:(UIImage *)image
          withKernelSize:(int)kernelSize;

typedef NS_ENUM(NSInteger, MorphType) {
    MorphOpening = 2,
    MorphClosing = 3,
    MorphGradient = 4,
    MorphTopHat = 5,
    MorphBlackHat = 6
};

+ (UIImage *)applyMorphologyEx:(UIImage *)image
                 withOperation:(MorphType)operation
                    kernelSize:(int)kernelSize;

typedef NS_ENUM(NSInteger, ElementType) {
    ElementRect = 0,
    ElementCross = 1,
    ElementEllipse = 2
};

//+ (UIImage *)getStructuringElementWithType:(ElementType)type
//                                kernelSize:(int)kernelSize;
+ (NSArray<NSNumber *> *)getStructuringElementWithShape:(int)shape size:(CGSize)size;

// 7-Image Contours and Shape Analysis

+ (nullable NSString *)findContoursInImage:(UIImage *)image;

+ (UIImage *)drawContoursOnImage:(UIImage *)image
                    withContours:(NSString *)contoursJSON
                           color:(UIColor *)color
                       thickness:(int)thickness;

+ (double)arcLengthOfContour:(NSString *)contourJSON
                    isClosed:(BOOL)isClosed;

+ (double)contourAreaOfContour:(NSString *)contourJSON;

+ (nullable NSString *)approxPolyDPOfContour:(NSString *)contourJSON
                                     epsilon:(double)epsilon
                                    isClosed:(BOOL)isClosed;

+ (nullable NSString *)convexHullOfContour:(NSString *)contourJSON;

+ (BOOL)isContourConvex:(NSString *)contourJSON;

+ (NSDictionary *)boundingRectOfContour:(NSString *)contourJSON;

+ (nullable NSDictionary *)minAreaRectOfContour:(NSString *)contourJSON;

+ (nullable NSDictionary *)fitEllipseOfContour:(NSString *)contourJSON;

+ (nullable NSDictionary *)fitLineOfContour:(NSString *)contourJSON;

// 8-Feature Detection and Matching

+ (nullable NSArray<NSDictionary *> *)goodFeaturesToTrackInImage:(UIImage *)image
                                                      maxCorners:(int)maxCorners
                                                    qualityLevel:(double)qualityLevel
                                                     minDistance:(double)minDistance;

+ (nullable NSArray<NSDictionary *> *)houghLinesInImage:(UIImage *)image
                                                    rho:(double)rho
                                                  theta:(double)theta
                                              threshold:(int)threshold;

+ (nullable NSArray<NSDictionary *> *)houghCirclesInImage:(UIImage *)image
                                                       dp:(double)dp
                                                  minDist:(double)minDist
                                                   param1:(double)param1
                                                   param2:(double)param2
                                                minRadius:(int)minRadius
                                                maxRadius:(int)maxRadius;

+ (nullable UIImage *)cornerHarrisInImage:(UIImage *)image
                                blockSize:(int)blockSize
                                    ksize:(int)ksize
                                        k:(double)k;

+ (nullable NSArray<NSDictionary *> *)detectORBKeypointsInImage:(UIImage *)image
                                                      nFeatures:(int)nFeatures;

+ (nullable NSArray<NSDictionary *> *)detectSIFTKeypointsInImage:(UIImage *)image
                                                       nFeatures:(int)nFeatures;

+ (NSArray<NSDictionary *> *)matchKeypointsWithBFMatcherDescriptors1:(NSArray<NSArray<NSNumber *> *> *)descriptors1
                                                        descriptors2:(NSArray<NSArray<NSNumber *> *> *)descriptors2;

+ (NSArray<NSDictionary *> *)matchKeypointsWithFlannMatcherDescriptors1:(NSArray<NSArray<NSNumber *> *> *)descriptors1
                                                           descriptors2:(NSArray<NSArray<NSNumber *> *> *)descriptors2;

+ (UIImage *)drawKeypointsOnImage:(UIImage *)image
                        keypoints:(NSArray<NSValue *> *)keypoints;

+ (UIImage *)matchTemplateInImage:(UIImage *)image
                    templateImage:(UIImage *)templateImage;

// 9-Optical Flow

+ (UIImage *)calculateOpticalFlowFromImage:(UIImage *)prevImage
                                   toImage:(UIImage *)nextImage;

+ (UIImage *)calculateOpticalFlowPyrLKFromImage:(UIImage *)prevImage
                                        toImage:(UIImage *)nextImage
                                      keypoints:(NSArray<NSValue *> *)keypoints;

+ (UIImage *)calculateMotionGradient:(UIImage *)image;

+ (CGFloat)calculateGlobalOrientationFromImage:(UIImage *)image;

// 10-Camera Calibration and 3D Vision

+ (NSArray<NSValue *> *)findChessboardCornersInImage:(UIImage *)image
                                           boardSize:(CGSize)boardSize;

+ (NSDictionary<NSString *, NSValue *> *)calibrateCameraWithObjectPoints:(NSArray<NSArray<NSValue *> *> *)objectPoints
                                                             imagePoints:(NSArray<NSArray<NSValue *> *> *)imagePoints
                                                               imageSize:(CGSize)imageSize;

//+ (UIImage *)undistortImage:(UIImage *)image
//           withCameraMatrix:(NSArray<NSNumber *> *)cameraMatrix
//                 distCoeffs:(NSArray<NSNumber *> *)distCoeffs;

+ (NSArray<NSNumber *> *)undistortWithImage:(NSArray<NSNumber *> *)image imageSize:(CGSize)imageSize cameraMatrix:(NSArray<NSNumber *> *)cameraMatrix distCoeffs:(NSArray<NSNumber *> *)distCoeffs;

+ (NSDictionary<NSString *, NSValue *> *)solvePnPWithObjectPoints:(NSArray<NSValue *> *)objectPoints
                                                      imagePoints:(NSArray<NSValue *> *)imagePoints
                                                     cameraMatrix:(NSArray<NSNumber *> *)cameraMatrix
                                                       distCoeffs:(NSArray<NSNumber *> *)distCoeffs;

+ (NSArray<NSValue *> *)projectPointsWithObjectPoints:(NSArray<NSValue *> *)objectPoints
                                          rotationVec:(NSArray<NSNumber *> *)rvec
                                       translationVec:(NSArray<NSNumber *> *)tvec
                                         cameraMatrix:(NSArray<NSNumber *> *)cameraMatrix
                                           distCoeffs:(NSArray<NSNumber *> *)distCoeffs;

+ (NSArray<NSNumber *> *)findHomographyWithSourcePoints:(NSArray<NSValue *> *)srcPoints
                                      destinationPoints:(NSArray<NSValue *> *)dstPoints;

+ (NSArray<NSDictionary *> *)decomposeHomographyMatrix:(NSArray<NSNumber *> *)homographyMatrix
                                          cameraMatrix:(NSArray<NSNumber *> *)cameraMatrix;

+ (NSArray<NSNumber *> *)findEssentialMatrixWithPoints1:(NSArray<NSValue *> *)points1
                                                points2:(NSArray<NSValue *> *)points2
                                           cameraMatrix:(NSArray<NSNumber *> *)cameraMatrix;

+ (NSDictionary *)decomposeEssentialMatrix:(NSArray<NSNumber *> *)essentialMatrix;

+ (NSArray<NSDictionary *> *)decomposeHomography:(NSArray<NSNumber *> *)homographyMatrix;

+ (NSArray<NSNumber *> *)findEssentialMatWithPoints1:(NSArray<NSArray<NSNumber *> *> *)points1
                                             points2:(NSArray<NSArray<NSNumber *> *> *)points2;

+ (NSArray<NSArray<NSNumber *> *> *)decomposeEssentialMatWithEssentialMat:(NSArray<NSArray<NSNumber *> *> *)essentialMat;

// 11-Video Processing

+ (UIImage *)captureFrameFromCameraIndex:(int)cameraIndex;

+ (BOOL)writeVideoFromImages:(NSArray<UIImage *> *)images
                  toFilePath:(NSString *)filePath
                         fps:(int)fps;


@end
NS_ASSUME_NONNULL_END
