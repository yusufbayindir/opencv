//
//  StructuringElement.swift
//  opencv
//
//  Created by Emrehan Kaya on 31.10.2024.
//

import SwiftUI

struct StructuringElement: View {
    @State private var structuringElement: [[NSNumber]] = []

    var body: some View {
        VStack {
            if !structuringElement.isEmpty {
                Text("Structuring Element Alındı.")
                    .padding()
            } else {
                Text("Structuring Element Alınıyor...")
                    .padding()
            }
        }
        .onAppear(perform: getStructuringElement)
    }

    private func getStructuringElement() {
        let shape = getCVShapeRectangle()
        let size = CGSize(width: 5, height: 5)
        
        let structuringElement = Opencv.getStructuringElement(withShape: Int32(shape), size: size)
        print("Structuring Element: \(structuringElement)")
    }

    private func getCVShapeRectangle() -> Int {
        return 0  // CV_SHAPE_RECT sabit değeri
    }

    private func getCVShapeEllipse() -> Int {
        return 2  // CV_SHAPE_ELLIPSE sabit değeri
    }

    private func getCVShapeCross() -> Int {
        return 1  // CV_SHAPE_CROSS sabit değeri
    }
}
