//
//  calibrateCamera.swift
//  opencv
//
//  Created by Emrehan Kaya on 29.10.2024.
//

import SwiftUI

struct CalibrateCamera: View {
    @State private var calibrationResults: [String: Any] = [:]

    var body: some View {
        VStack {
            if let cameraMatrix = calibrationResults["cameraMatrix"] as? [Double],
               let distCoeffs = calibrationResults["distCoeffs"] as? [Double] {
                Text("Camera Matrix: \(cameraMatrix)")
                Text("Distortion Coefficients: \(distCoeffs)")
                Text("RMS Error: \(calibrationResults["rms"] ?? 0)")
            } else {
                Text("No calibration data available.")
            }
        }
        .onAppear {
            performCalibration()
        }
    }
    
    func performCalibration() {
        let objectPoints: [[CGPoint]] = [] // Buraya obje noktalarınızı ekleyin
        let imagePoints: [[CGPoint]] = []  // Buraya görüntü noktalarınızı ekleyin
        let imageSize = CGSize(width: 640, height: 480)
        
        let nsObjectPoints = objectPoints.map { $0.map { NSValue(cgPoint: $0) } }
        let nsImagePoints = imagePoints.map { $0.map { NSValue(cgPoint: $0) } }
        
        if let result = Opencv.calibrateCamera(withObjectPoints: nsObjectPoints, imagePoints: nsImagePoints, imageSize: imageSize) as? [String: Any] {
            
            // Swift için cameraMatrix ve distCoeffs'ü uygun formata dönüştürün
            if let cameraMatrixValue = result["cameraMatrix"] as? NSValue,
               let distCoeffsValue = result["distCoeffs"] as? NSValue {
                calibrationResults = [
                    "cameraMatrix": convertMatToSwiftArray(cameraMatrixValue),
                    "distCoeffs": convertMatToSwiftArray(distCoeffsValue),
                    "rms": result["rms"] as? Double ?? 0
                ]
            }
        }
    }
    
    func convertMatToSwiftArray(_ value: NSValue) -> [Double] {
        // NSValue'yi uygun bir Double dizisine dönüştürme işlemi burada yapılır
        // Bu örnekte, sabit bir [0.0] değeri döndürüyoruz; gerçek dönüşüm kodunu OpenCV ile özelleştirmeniz gerekebilir
        return [0.0] // Gerekli dönüşümü buraya ekleyin
    }
}
