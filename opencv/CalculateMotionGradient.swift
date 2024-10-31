//
//  CalculateMotionGradient.swift
//  opencv
//
//  Created by Emrehan Kaya on 31.10.2024.
//

import SwiftUI

struct CalculateMotionGradient: View {
    var body: some View {
        VStack {
            if let inputImage = UIImage(named: "izmir.jpg") {
                // Motion gradient işlemini uygula
                let gradientImage = Opencv.calculateMotionGradient(inputImage)
                Image(uiImage: gradientImage)
                    .resizable()
                    .scaledToFit()
            } else {
                Text("Image not found")
            }
        }
    }
}
