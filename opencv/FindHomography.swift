//
//  FindHomography.swift
//  opencv
//
//  Created by Emrehan Kaya on 29.10.2024.
//

import SwiftUI

struct FindHomography: View {
    @State private var homographyMatrix: [[Double]] = []

    var body: some View {
        VStack {
            if !homographyMatrix.isEmpty {
                Text("Homography Matrix:")
                ForEach(0..<homographyMatrix.count, id: \.self) { i in
                    Text("\(homographyMatrix[i])")
                }
            } else {
                Text("Homography matrix will appear here.")
            }
        }
        .onAppear {
            calculateHomography()
        }
    }
    
    func calculateHomography() {
        let srcPoints: [CGPoint] = [/* Kaynak noktalar */]
        let dstPoints: [CGPoint] = [/* Hedef noktalar */]
        
        let nsSrcPoints = srcPoints.map { NSValue(cgPoint: $0) }
        let nsDstPoints = dstPoints.map { NSValue(cgPoint: $0) }

        if let result = Opencv.findHomography(withSourcePoints: nsSrcPoints,
                                                                     destinationPoints: nsDstPoints) as? [NSNumber] {
            
            homographyMatrix = convertTo2DArray(result, rows: 3, cols: 3)
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
