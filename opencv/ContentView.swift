//
//  ContentView.swift
//  opencv
//
//  Created by Yusuf Bayindir on 10/18/24.
//
import SwiftUI
import PhotosUI
import Photos

struct ContentView: View {
    @State private var selectedImage: UIImage? = nil
    @State private var isPickerPresented: Bool = false
    @State private var hasPhotoLibraryAccess: Bool = false
    
    var body: some View {
        VStack {
            // Galeriden resim seçme butonu
            Button(action: {
                checkPhotoLibraryPermission()
            }) {
                Text("Galeriden Resim Seç")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .padding()
            .sheet(isPresented: $isPickerPresented) {
                PhotoPicker(selectedImage: $selectedImage)
            }
            
            // Seçilen resmi göstermek için (Daha büyük olacak şekilde ayarlandı)
            if let selectedImage = selectedImage {
                Image(uiImage: selectedImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: 600) // Resim daha büyük gösterilecek
            } else {
                Text("Henüz bir resim seçilmedi.")
                    .padding()
            }
            
            Spacer()
            
            // Alttaki 3 buton
            HStack {
                Button(action: {
                    // İlk butonun etkisi
                    if let image = selectedImage {
                        let newImage = Opencv.addText(to: image, text: "Yusuf", position: CGPoint(x: 150, y: 150), fontFace: 4, fontScale: 5, color: UIColor.black, thickness: 4, lineType: 2)
                        
                       selectedImage = newImage // Değiştirilmiş resmi güncelle
                       print("Resim üzerine yazı eklendi.")
                   }
                    print("İlk buton tıklandı")
                }) {
                    Text("Buton 1")
                        .padding()
                        .background(Color.red)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                
                Button(action: {
                    // İkinci butonun etkisi
                   
                    if let image = selectedImage {
                        let newImage = Opencv.applySobel(to: image, dx: 1, dy: 0, kernelSize: 3)
                        
                       selectedImage = newImage // Değiştirilmiş resmi güncelle
                       print("Resim üzerine yazı eklendi.")
                   }
                    print("İkinci buton tıklandı.")
                }) {
                    Text("Buton 2")
                        .padding()
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                
                Button(action: {
                    // Üçüncü butonun etkisi
                    
//                    if let image = selectedImage {
//                        let grayAndSize = Opencv.resizeAndGrayColor(image, to: CGSize(width: 200, height: 500))
//                        selectedImage = grayAndSize
//                        print("Resim gri rengine dönüştürüldü ve boyut değiştirildi.")
//                    }
//                    
                    
//                    if let image = selectedImage {
//                        let borderImage = Opencv.makeBorder(with: image, top: 20, bottom: 20, left: 400, right: 400, borderType: 0, color: .red)
//                        selectedImage = borderImage
//                        print("border eklendi")
//                    }
                    
//                    if let image = selectedImage {
//                        let flipImage = Opencv.flip(image, flipCode: 1)
//                        selectedImage = flipImage
//                        print("Resim çevrildi")
//                    }
                    
//                    if let image = selectedImage {
//                        let filePath = NSTemporaryDirectory().appending(UUID().uuidString).appending(".png")
//                        if Opencv.save(image, toFilePath: filePath) {
//                            Opencv.saveImage(toGallery: image)
//                            print("Resim galeriye kaydedildi.")
//                        }
//                    }
                    
//                    if let image = selectedImage {
//                        let ellipse = Opencv.drawEllipse(on: image, center: CGPoint(x: 900, y: 900), axes: CGSize(width: 1000, height: 500), angle: 45, startAngle: 0, endAngle: 360, color: UIColor.red, thickness: 30)
//                        selectedImage = ellipse
//                        print("ellipse çizildi")
//                    }
                    
//                    if let image = selectedImage {
//                        let lineImage = Opencv.drawLine(on: image, start: CGPoint(x: 50, y: 150), end: CGPoint(x: 4000, y: 4000), color: UIColor.red, thickness: 50)
//                        selectedImage = lineImage
//                        print("line çizildi")
//                    }
                    
//                    if let image = selectedImage {
//                        let circleImage = Opencv.drawCircle(on: image, at: CGPoint(x: 2500, y: 1500), withRadius: 1500, andColor: UIColor.red, lineWidth: 10)
//                        selectedImage = circleImage
//                        print("circle çizildi")
//                    }
                    
//                    if let image = selectedImage {
//                        let rectangleImage = Opencv.drawRectangle(on: image, from: CGPoint(x: 1200, y: 600), to: CGPoint(x: 200, y: 200), with: UIColor.red, lineWidth: 30)
//                        selectedImage = rectangleImage
//                        print("rectangle çizildi")
//                    }
                    
//                    if let image = selectedImage {
//                        let points = [
//                               NSValue(cgPoint: CGPoint(x: 800, y: 250)),
//                               NSValue(cgPoint: CGPoint(x: 500, y: 250)),
//                               NSValue(cgPoint: CGPoint(x: 500, y: 700)),
//                               NSValue(cgPoint: CGPoint(x: 250, y: 600))
//                           ]
//                        let fillPolygon = Opencv.fillPolygon(on: image, withPoints: points, andColor: UIColor.red)
//                        selectedImage = fillPolygon
//                        print("polygon çizildi")
//                    }
                    
//                    if let image = selectedImage {
//                        let points = [
//                            NSValue(cgPoint: CGPoint(x: 800, y: 250)),
//                            NSValue(cgPoint: CGPoint(x: 500, y: 250)),
//                            NSValue(cgPoint: CGPoint(x: 500, y: 700)),
//                            NSValue(cgPoint: CGPoint(x: 250, y: 600))
//                        ]
//                        
//                        let polylinesImage = Opencv.drawPolylines(on: image, withPoints: points, andColor: UIColor.red, lineWidth: 30)
//                        selectedImage = polylinesImage
//                        print("polyline çizildi")
//                    }
                    
//                    if let image = selectedImage {
//                        let center = Opencv.rotateImage(image, center: CGPoint(x: Int(image.size.width) / 2, y: Int(image.size.height) / 2), angle: 45, scale: 1)
//                        selectedImage = center
//                        print("Resim döndürüldü")
//                    }
                    
//                    if let image = selectedImage {
//                        let warpMat : [NSNumber] = [1.0, 0.0, 100.0,
//                                                    0.0, 1.0, 50.0]
//                        let warpImage = Opencv.applyWarpAffine(to: image, matrix: warpMat)
//                        selectedImage = warpImage
//                        print("Resime warp işlemi uygulandı.")
//                    }
                    
//                    if let image = selectedImage {
//                        let srcPoints: [CGPoint] = [
//                                    CGPoint(x: 0, y: 0),
//                                    CGPoint(x: image.size.width, y: 0),
//                                    CGPoint(x: image.size.width, y: image.size.height),
//                                    CGPoint(x: 0, y: image.size.height)
//                                ]
//
//                                let dstPoints: [CGPoint] = [
//                                    CGPoint(x: 50, y: 50),
//                                    CGPoint(x: image.size.width - 50, y: 10),
//                                    CGPoint(x: image.size.width - 30, y: image.size.height - 30),
//                                    CGPoint(x: 10, y: image.size.height - 50)
//                                ]
//                        
//                        let warpedImage = Opencv.warpPerspectiveImage(image, srcPoints: srcPoints.map { NSValue(cgPoint: $0) }, dstPoints: dstPoints.map { NSValue(cgPoint: $0)})
//                        selectedImage = warpedImage
//                        print("Resime warpPerspective uygulandı")
//
//                    }
                    
//                    if let image = selectedImage {
//                        let srcPoints: [NSValue] = [
//                            NSValue(cgPoint: CGPoint(x: 200, y: 200)),
//                            NSValue(cgPoint: CGPoint(x: image.size.width, y: 300)),
//                            NSValue(cgPoint: CGPoint(x: image.size.width, y: image.size.height)),
//                            NSValue(cgPoint: CGPoint(x: 250, y: image.size.height))
//                        ]
//                        let dtsPoints: [NSValue] = [
//                            NSValue(cgPoint: CGPoint(x: 600, y: 600)),
//                            NSValue(cgPoint: CGPoint(x: image.size.width, y: 900)),
//                            NSValue(cgPoint: CGPoint(x: image.size.width, y: image.size.height)),
//                            NSValue(cgPoint: CGPoint(x: 700, y: image.size.height))
//                            ]
//                        
//                        let warpedImage = Opencv.warpPerspectiveImage(image, srcPoints: srcPoints, dstPoints: dtsPoints)
//                        selectedImage = warpedImage
//                        print("Resime warpPerspective uygulandı")    
//                    }
                    
//                    if let image = selectedImage {
//                        let transposedImage = Opencv.transposeImage(image)
//                        selectedImage = transposedImage
//                        print("Transpose işlemi uygulandı")
//                    }
                    
//                    if let image = selectedImage {
//                        let kernelSize = CGSize(width: 15, height: 15)
//                        let sigma = 5.0
//                        let blurredImage = Opencv.gaussianBlur(image, withKernelSize: kernelSize, sigma: sigma)
//                        selectedImage = blurredImage
//                    }
                    
//                    if let image = selectedImage {
//                        let kernelSize = 15
//                        let medianblurImage = Opencv.medianBlur(image, withKernelSize: Int32(kernelSize))
//                        selectedImage = medianblurImage
//                    }
                    
//                    if let image = selectedImage { // UIImage olarak yükleme
//                        let kernelSize = CGSize(width: 15, height: 15)
//                        let blurredImage = Opencv.blur(image, withKernelSize: kernelSize)
//                        selectedImage = blurredImage
//                        print("Resime blur eklendi")
//                    }
                    
//                    if let image = selectedImage {
//                        let bilateralFilter = Opencv.applyBilateralFilter(to: image, diameter: 50, sigmaColor: 250, sigmaSpace: 250)
//                        selectedImage = bilateralFilter
//                    }
//
//                    if let image = selectedImage {
//                        let kernel: [[NSNumber]] = [
//                                                [0, -1, 0],
//                                                [-1, 5, -1],
//                                                [0, -1, 0]
//                                            ]
//                        let filter2D = Opencv.applyFilter2D(to: image, kernel: kernel)
//                        selectedImage = filter2D
//                    }
                    
                    print("Üçüncü buton tıklandı")
                }) {
                    Text("Buton 3")
                        .padding()
                        .background(Color.orange)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
            }
            .padding()
        }
    }
    // Galeri izni kontrolü ve izin isteme
    func checkPhotoLibraryPermission() {
        let status = PHPhotoLibrary.authorizationStatus()
        switch status {
        case .authorized:
            // Eğer izin verilmişse, galeriyi aç
            isPickerPresented = true
        case .denied, .restricted:
            // Eğer izin yoksa, kullanıcıya ayarlardan izin vermesini söyle
            print("Galeri erişim izni reddedildi.")
        case .notDetermined:
            // Eğer izin sorulmadıysa, izin sorulacak
            PHPhotoLibrary.requestAuthorization { newStatus in
                if newStatus == .authorized {
                    DispatchQueue.main.async {
                        isPickerPresented = true
                    }
                } else {
                    print("Kullanıcı izni reddetti.")
                }
            }
        default:
            break
        }
    }
//    // Görüntüyü kaydetme işlemi
//    func saveImage() {
//        if let uiImage = selectedImage {
//            // Belgeler dizinine kaydedilecek dosya yolu
//            if let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
//                let filePath = documentsDirectory.appendingPathComponent("outputImage.png").path
//                
//                // Görüntüyü Opencv ile kaydet
//                let success = Opencv.save(uiImage, toPath: filePath)
//                if success {
//                    saveResult = "Image saved successfully to \(filePath)"
//                    print("Resim kaydedildi.")
//                } else {
//                    saveResult = "Failed to save image"
//                    print("Resim kaydedilmedi.")
//                }
//            }
//        }
//    }
}

struct PhotoPicker: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?

    func makeCoordinator() -> Coordinator {
        return Coordinator(self)
    }

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .images
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        var parent: PhotoPicker

        init(_ parent: PhotoPicker) {
            self.parent = parent
        }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            picker.dismiss(animated: true)
            
            if let result = results.first {
                result.itemProvider.loadObject(ofClass: UIImage.self) { (image, error) in
                    if let image = image as? UIImage {
                        DispatchQueue.main.async {
                            self.parent.selectedImage = image
                            if let uiImage = image as? UIImage {
                                let imagePath = self.saveImageToTemporaryDirectory(image: uiImage)
                                self.parent.selectedImage = Opencv.loadImage(imagePath ?? "")
                            }
                        }
                    }
                }
                
            }
        }
        func saveImageToTemporaryDirectory(image: UIImage) -> String {
           let imageData = image.pngData() // UIImage'ı PNG veri formatına dönüştür.
           let tempDirectory = FileManager.default.temporaryDirectory // Geçici dizini al.
           let imageURL = tempDirectory.appendingPathComponent(UUID().uuidString + ".png") // Geçici dizine benzersiz bir adla dosya oluştur.
           try? imageData?.write(to: imageURL) // PNG verisini dosyaya yaz.
            print(imageURL.path)
           return imageURL.path // Dosya yolunu döner.
       }
        
        
    }
    
  

}
