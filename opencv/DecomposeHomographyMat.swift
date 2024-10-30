//
//  DecomposeHomographyMat.swift
//  opencv
//
//  Created by Emrehan Kaya on 29.10.2024.
//

import SwiftUI

struct DecomposeHomographyMat: View {
    @State private var decompositions: [[String: [Double]]] = []
    
    var body: some View {
        VStack {
            Text("Decomposed Homography Components")
                .font(.title)
                .padding()
            
            List(decompositions.indices, id: \.self) { index in
                Section(header: Text("Decomposition \(index + 1)")) {
                    VStack(alignment: .leading) {
                        if let rotation = decompositions[index]["rotation"] {
                            Text("Rotation Matrix:")
                                .font(.headline)
                            Text(matrixToString(matrix: rotation))
                                .padding(.bottom)
                        }
                        if let translation = decompositions[index]["translation"] {
                            Text("Translation Vector:")
                                .font(.headline)
                            Text(vectorToString(vector: translation))
                                .padding(.bottom)
                        }
                        if let normal = decompositions[index]["normal"] {
                            Text("Normal Vector:")
                                .font(.headline)
                            Text(vectorToString(vector: normal))
                        }
                    }
                    .padding(.vertical)
                }
            }
        }
        .onAppear {
            decomposeHomography()
        }
    }
    
     func decomposeHomography() {
        let homographyMatrix: [NSNumber] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        if let result = Opencv.decomposeHomography(homographyMatrix) as? [[String: [Double]]] {
            decompositions = result
        }
    }
    
    private func matrixToString(matrix: [Double]) -> String {
        return stride(from: 0, to: matrix.count, by: 3).map {
            matrix[$0..<$0+3].map { String(format: "%.2f", $0) }.joined(separator: "\t")
        }.joined(separator: "\n")
    }
    
    private func vectorToString(vector: [Double]) -> String {
        vector.map { String(format: "%.2f", $0) }.joined(separator: ", ")
    }
}

