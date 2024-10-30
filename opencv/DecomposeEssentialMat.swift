//
//  DecomposeEssentialMat.swift
//  opencv
//
//  Created by Emrehan Kaya on 29.10.2024.
//

import SwiftUI

struct DecomposeEssentialMat: View {
    @State private var rotation1: [[Double]] = []
    @State private var rotation2: [[Double]] = []
    @State private var translation: [Double] = []

    var body: some View {
        VStack {
            if !rotation1.isEmpty {
                Text("Rotation Matrix 1:")
                ForEach(rotation1, id: \.self) { row in
                    Text("\(row)")
                }
            }
            if !rotation2.isEmpty {
                Text("Rotation Matrix 2:")
                ForEach(rotation2, id: \.self) { row in
                    Text("\(row)")
                }
            }
            if !translation.isEmpty {
                Text("Translation Vector: \(translation)")
            }
        }
        .onAppear {
            decomposeEssentialMatrix()
        }
    }
    
    func decomposeEssentialMatrix() {
        let essentialMatrix: [NSNumber] = [/* Essential matris elemanları */]
        
        if let result = Opencv.decomposeEssentialMatrix(essentialMatrix) as? [String: [NSNumber]] {
            rotation1 = convertTo2DArray(result["rotation1"] ?? [], rows: 3, cols: 3)
            rotation2 = convertTo2DArray(result["rotation2"] ?? [], rows: 3, cols: 3)
            translation = result["translation"]?.map { $0.doubleValue } ?? []
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
