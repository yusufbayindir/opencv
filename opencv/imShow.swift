//
//  imShow.swift
//  opencv
//
//  Created by Emrehan Kaya on 31.10.2024.
//

import SwiftUI

struct imShow: View {
    var body: some View {
        VStack {
            Text("Görüntü Gösteriliyor")
                .padding()
                .onAppear(perform: showImage)
        }
    }

    private func showImage() {
        let width = 300
        let height = 100
        let imageData: [NSNumber] = [255, 0, 0, 0, 255, 0, 0, 0, 255] // Örnek RGB verisi
        let windowName = "izmir.jpg"
        
        Opencv.showImage(imageData, width: Int32(width), height: Int32(height), windowName: windowName)
    }
}
