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

+ (UIImage *)addTextToUIImage:(UIImage *)image
                         text:(NSString *)text
                     position:(CGPoint)position
                     fontFace:(int)fontFace
                    fontScale:(double)fontScale
                        color:(UIColor *)color
                    thickness:(int)thickness
                     lineType:(int)lineType;

@end

NS_ASSUME_NONNULL_END
