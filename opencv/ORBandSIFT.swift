//
//  ORB.swift
//  opencv
//
//  Created by Emrehan Kaya on 29.10.2024.
//

import SwiftUI

struct ORBandSIFT: View {
    @State private var keypoints: [CGPoint] = []

    var body: some View {
        VStack {
            if let image = UIImage(named: "izmir.jpg") {
                Image(uiImage: drawKeypoints(on: image))
                    .resizable()
                    .scaledToFit()
                    .frame(width: 300, height: 300)
            }
        }
    }
    
    func processImage() {
        if let image = UIImage(named: "izmir.jpg") {
            let nFeatures = 500
            
            // Anahtar noktaları tespit et
//            if let keypointsArray = Opencv.detectORBKeypoints(in: image, nFeatures: Int32(nFeatures)) {
            // Anahtar noktaları tespit et
            if let keypointsArray = Opencv.detectSIFTKeypoints(in: image, nFeatures: Int32(nFeatures)) {
                self.keypoints = keypointsArray.compactMap {
                    if let x = $0["x"] as? CGFloat, let y = $0["y"] as? CGFloat {
                        return CGPoint(x: x, y: y)
                    }
                    return nil
                }
            }
        }
    }
    
    func drawKeypoints(on image: UIImage) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: image.size)
        return renderer.image { context in
            image.draw(at: .zero)
            
            context.cgContext.setStrokeColor(UIColor.red.cgColor)
            context.cgContext.setLineWidth(2.0)
            
            for point in keypoints {
                context.cgContext.addArc(center: point, radius: 4.0, startAngle: 0, endAngle: .pi * 2, clockwise: true)
                context.cgContext.strokePath()
            }
        }
    }
    
    init() {
        processImage()
    }
}
