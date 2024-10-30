//
//  FindEssentialMat.swift
//  opencv
//
//  Created by Emrehan Kaya on 29.10.2024.
//

import SwiftUI

struct FindEssentialMat: View {
    @State private var essentialMatrix: [[Double]] = []

    var body: some View {
        VStack {
            if !essentialMatrix.isEmpty {
                Text("Essential Matrix:")
                ForEach(0..<essentialMatrix.count, id: \.self) { i in
                    Text("\(essentialMatrix[i])")
                }
            } else {
                Text("Essential matrix will appear here.")
            }
        }
        .onAppear {
            calculateEssentialMatrix()
        }
    }
    
    func calculateEssentialMatrix() {
        let points1: [CGPoint] = [/* İlk görüntüdeki eşleşen noktalar */]
        let points2: [CGPoint] = [/* İkinci görüntüdeki eşleşen noktalar */]
        let cameraMatrix: [NSNumber] = [/* 3x3 kamera matrisi değerleri */]
        
        let nsPoints1 = points1.map { NSValue(cgPoint: $0) }
        let nsPoints2 = points2.map { NSValue(cgPoint: $0) }

        if let result = Opencv.findEssentialMatrix(withPoints1: nsPoints1,
                                                                     points2: nsPoints2,
                                                                     cameraMatrix: cameraMatrix) as? [NSNumber] {
            
            essentialMatrix = convertTo2DArray(result, rows: 3, cols: 3)
        }
    }
    
    func convertTo2DArray(_ flatArray: [NSNumber], rows: Int, cols: Int) -> [[Double]] {
        var matrix: [[Double]] = []
        for i in 0..<rows {
            let row = Array(flatArray[i*cols..<(i+1)*cols]).map { $0.doubleValue }
            matrix.append(row)
        }
        return matrix
    }
}
