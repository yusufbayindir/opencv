//
//  VideoCapture().swift
//  opencv
//
//  Created by Emrehan Kaya on 30.10.2024.
//

import SwiftUI

struct VideoCapture: View {
    @State private var image: UIImage? = nil
    @State var timer: Timer?

    var body: some View {
        VStack {
            if let uiImage = image {
                Image(uiImage: uiImage)
                    .resizable()
                    .scaledToFit()
                    .frame(width: 300, height: 300) // Görüntü boyutunu ayarlayın
            } else {
                Text("Kamera görüntüsü yüklenemedi.")
                    .foregroundColor(.red)
            }
        }
        .onAppear {
            startCamera()
        }
        .onDisappear {
            stopCamera()
        }
    }

     func startCamera() {
        // 0, varsayılan kamera indeksidir. Farklı bir indeks kullanabilirsiniz.
         timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { _ in
             let frame = Opencv.captureFrame(fromCameraIndex: 0)
                image = frame
        }
    }

    func stopCamera() {
        timer?.invalidate()
        timer = nil
    }
}

