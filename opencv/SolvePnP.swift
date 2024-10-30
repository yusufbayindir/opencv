//
//  SolvePnP.swift
//  opencv
//
//  Created by Emrehan Kaya on 29.10.2024.
//

import SwiftUI

import SwiftUI

struct SolvePnP: View {
    @State private var rotationVector: [Double] = []
    @State private var translationVector: [Double] = []

    var body: some View {
        VStack {
            Text("Rotation Vector: \(rotationVector)")
            Text("Translation Vector: \(translationVector)")
        }
        .onAppear {
            performSolvePnP()
        }
    }
    
    func performSolvePnP() {
        let objectPoints: [CGPoint] = [/* 3D nokta listesi */]
        let imagePoints: [CGPoint] = [/* 2D görüntü noktaları */]
        
        let cameraMatrix: [NSNumber] = [1.0, 0.0, 320.0,
                                        0.0, 1.0, 240.0,
                                        0.0, 0.0, 1.0]
        let distCoeffs: [NSNumber] = [0.1, -0.1, 0.0, 0.0, 0.0]

        let nsObjectPoints = objectPoints.map { NSValue(cgPoint: $0) }
        let nsImagePoints = imagePoints.map { NSValue(cgPoint: $0) }

        if let result = Opencv.solvePnP(withObjectPoints: nsObjectPoints,
                                                               imagePoints: nsImagePoints,
                                                               cameraMatrix: cameraMatrix,
                                                               distCoeffs: distCoeffs) as? [String: NSValue] {
            
            rotationVector = convertMatToArray(result["rvec"])
            translationVector = convertMatToArray(result["tvec"])
        }
    }
    
    func convertMatToArray(_ value: NSValue?) -> [Double] {
        // Burada cv::Mat içeriğini bir Double dizisine dönüştürün
        return []
    }
}
