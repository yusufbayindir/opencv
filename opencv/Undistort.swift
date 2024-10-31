//
//  Undistort().swift
//  opencv
//
//  Created by Emrehan Kaya on 31.10.2024.
//

import SwiftUI

struct Undistort: View {
    @State private var undistortedImageData: [NSNumber] = []

    var body: some View {
        VStack {
            if !undistortedImageData.isEmpty {
                // İşlenmiş görüntü verilerini gösterme
                Text("Görüntü Düzeltildi.")
            } else {
                Text("Görüntü Düzeltiliyor...")
            }
        }
        .onAppear(perform: undistortImage)
    }

    private func undistortImage() {
        let image: [NSNumber] = [/* Görüntü verileri buraya gelecek */]
        let imageSize = CGSize(width: 640, height: 480)
        let cameraMatrix: [NSNumber] = [/* Kamera matrisi verileri buraya gelecek */]
        let distCoeffs: [NSNumber] = [/* Distorsiyon katsayıları buraya gelecek */]
        
        undistortedImageData = Opencv.undistort(withImage: image, imageSize: imageSize, cameraMatrix: cameraMatrix, distCoeffs: distCoeffs)
        print("Görüntü Düzeltildi: \(undistortedImageData)")
    }
}
