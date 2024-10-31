//
//  resizeWindow.swift
//  opencv
//
//  Created by Emrehan Kaya on 31.10.2024.
//

import SwiftUI


struct resizeWindow: View {
    var body: some View {
        VStack {
            Text("Pencere Boyutlandırma")
                .padding()
                .onAppear(perform: resizeWindow)
        }
    }

    private func resizeWindow() {
        let windowName = "Image Window"
        let width = 800
        let height = 600
        
        Opencv.resizeWindow(withName: windowName, width: Int32(width), height: Int32(height))
    }
}
