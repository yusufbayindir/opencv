//
//  ProjectPoints.swift
//  opencv
//
//  Created by Emrehan Kaya on 29.10.2024.
//

import SwiftUI

import SwiftUI

struct ProjectPoints: View {
    @State private var projectedPoints: [CGPoint] = []

    var body: some View {
        VStack {
            if !projectedPoints.isEmpty {
                ForEach(projectedPoints, id: \.self) { point in
                    Text("Projected Point: \(point)")
                }
            } else {
                Text("Projected points will appear here.")
            }
        }
        .onAppear {
            project3DPoints()
        }
    }
    
    func project3DPoints() {
        let objectPoints: [CGPoint] = [/* 3D noktalar (ör. [(1, 1), (2, 2)]) */]
        
        let rotationVec: [NSNumber] = [0.0, 0.0, 0.0]
        let translationVec: [NSNumber] = [0.0, 0.0, 0.0]
        
        let cameraMatrix: [NSNumber] = [1.0, 0.0, 320.0,
                                        0.0, 1.0, 240.0,
                                        0.0, 0.0, 1.0]
        let distCoeffs: [NSNumber] = [0.1, -0.1, 0.0, 0.0, 0.0]

        let nsObjectPoints = objectPoints.map { NSValue(cgPoint: $0) }

        if let result = Opencv.projectPoints(withObjectPoints: nsObjectPoints,
                                                                    rotationVec: rotationVec,
                                                                    translationVec: translationVec,
                                                                    cameraMatrix: cameraMatrix,
                                                                    distCoeffs: distCoeffs) as? [NSValue] {
            
            projectedPoints = result.map { $0.cgPointValue }
        }
    }
}
