//
//  ChessboardCorners.swift
//  opencv
//
//  Created by Emrehan Kaya on 29.10.2024.
//

import SwiftUI

import SwiftUI

struct ChessboardCorners: View {
    @State private var corners: [CGPoint] = []

    var body: some View {
        VStack {
            if !corners.isEmpty {
                ForEach(corners, id: \.self) { corner in
                    Text("Corner: (\(corner.x), \(corner.y))")
                }
            } else {
                Text("No corners found.")
            }
        }
        .onAppear {
            processImage()
        }
    }
    
    func processImage() {
        guard let inputImage = UIImage(named: "izmir.jpg") else { return }
        
        // Satranç tahtası köşelerini bul
        let boardSize = CGSize(width: 9, height: 6) // 9x6 köşe sayısı
        let nsValueCorners = Opencv.findChessboardCorners(in: inputImage, boardSize: boardSize)
        
        // NSValue dizisini CGPoint dizisine dönüştür
        self.corners = nsValueCorners.compactMap { $0.cgPointValue }
    }
}
