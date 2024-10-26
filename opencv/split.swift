//
//  split.swift
//  opencv
//
//  Created by Emrehan Kaya on 24.10.2024.
//

import SwiftUI

import SwiftUI

struct split: View {
    @State private var splitImages: [UIImage] = []
    private let imageSplitter = Opencv()
    
    var body: some View {
        VStack {
            // Görüntüleri göstermek için bir HStack kullan
            HStack {
                ForEach(splitImages, id: \.self) { image in
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 100, height: 100)
                }
            }
            .onAppear {
                loadImage()
            }
        }
    }
    
    private func loadImage() {
        // Örnek bir UIImage yükleyin
        if let image = UIImage(named: "izmir.jpg") {
            // Görüntüyü bileşenlerine ayırın
            splitImages = Opencv.splitImage(image)
        }
    }
}
