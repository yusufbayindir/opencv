//
//  VideoWriter.swift
//  opencv
//
//  Created by Emrehan Kaya on 30.10.2024.
//

import SwiftUI

import SwiftUI

struct VideoWriter: View {
    @State private var isVideoWritten = false

    var body: some View {
        VStack {
            Text(isVideoWritten ? "Video written successfully!" : "Press to write video")
            
            Button("Write Video") {
                writeVideo()
            }
            .padding()
        }
    }

    func writeVideo() {
        // Örnek görüntüler dizisi (UIImage türünde)
        let images: [UIImage] = [] // Buraya UIImage dizinizi ekleyin.
        
        // Dosya yolu
        let filePath = NSTemporaryDirectory() + "output_video.mov"
        
        // Video yazma işlemi
        isVideoWritten = Opencv.writeVideo(from: images, toFilePath: filePath, fps: 30)
    }
}
