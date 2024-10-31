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
    
    @State private var globalOrientation: CGFloat = 0.0

    
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
                    
//                                        if let image = selectedImage {
//                                            let borderImage = Opencv.makeBorder(with: image, top: 20, bottom: 20, left: 400, right: 400, borderType: 0, color: .red)
//                                            selectedImage = borderImage
//                                            print("border eklendi")
//                                        }
                    
//                                        if let image = selectedImage {
//                                            let flipImage = Opencv.flip(image, flipCode: 1)
//                                            selectedImage = flipImage
//                                            print("Resim çevrildi")
//                                        }
                    
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
                    
//                    if let image1 = UIImage(named: "izmir.jpg"),
//                       let image2 = UIImage(named: "izmir.jpg") {
//                        let resultImage = Opencv.bitwiseAnd(withImage1: image1, image2: image2)
//                        selectedImage = resultImage
//                        print("bitwiseAnd işlemi uygulandı")
//                    }
                    
//                    if let image1 = UIImage(named: "izmir.jpg"),
//                       let image2 = UIImage(named: "izmir.jpg") {
//                        let resultImage = Opencv.bitwiseOr(withImage1: image1, image2: image2)
//                        selectedImage = resultImage
//                        print("bitwiseOr işlemi uygulandı")
//                    }
                    
//                    if let image = selectedImage {
//                        let resultImage = Opencv.bitwiseNot(with: image)
//                        selectedImage = resultImage
//                        print("bitwiseNot işlemi uygulandı")
//                    }
                    
//                    if let image1 = UIImage(named: "izmir.jpg"),
//                       let image2 = UIImage(named: "izmir.jpg") {
//                        let alpha = 0.7  // image1' in ağırlığı
//                        let beta = 0.3   // image2' nin ağırlığı
//                        let gamma = 0.0  // Sabir bir değeri eklemek isterseniz, örn: 0.0
//                        
//                        let resultImage = Opencv.addWeighted(withImage1: image1, image2: image2, alpha: alpha, beta: beta, gamma: gamma)
//                        selectedImage = resultImage
//                        print("addWighted işlemi uygulandı")
//                    }
                    
//                    if let channel1 = UIImage(named: "izmir.jpg"),
//                       let channel2 = UIImage(named: "izmir.jpg"),
//                       let channel3 = UIImage(named: "izmir.jpg") {
//                        let mergedImage = Opencv.merge(withChannel1: channel1, channel2: channel2, channel3: channel3)
//                        selectedImage = mergedImage
//                        print("merge işlemi uygulandı")
//                    }
                    
                    
//                    // Üç kaynak ve üç hedef noktayı CGPoint olarak tanımlayın
//                        let sourcePoints = [
//                            NSValue(cgPoint: CGPoint(x: 10, y: 20)),
//                            NSValue(cgPoint: CGPoint(x: 20, y: 10)),
//                            NSValue(cgPoint: CGPoint(x: 10, y: 20)),
//                        ]
//                        
//                        let destinationPoints = [
//                            NSValue(cgPoint: CGPoint(x: 30, y: 30)),
//                            NSValue(cgPoint: CGPoint(x: 40, y: 30)),
//                            NSValue(cgPoint: CGPoint(x: 30, y: 40)),
//                        ]
//                    
//                    // Affine dönüşüm matrisini al
//                        if let affineMatrix = Opencv.getAffineTransform(withSourcePoints: sourcePoints, destinationPoints: destinationPoints) {
//                            print("Affine Transformation Matrix: \(affineMatrix)")
//                        }
                    
//                    if let image = selectedImage {
//                        if let enlargedImage = Opencv.pyrUp(with: image) {
//                            selectedImage = enlargedImage
//                            print("pyrUp işlemi uygulandı")
//                        }
//                    }

//                    if let image = selectedImage {
//                        if let reducedImage = Opencv.pyrDown(with: image) {
//                            selectedImage = reducedImage
//                            print("pyrDown işlemi uygulandı")
//                        }
//                    }
                    
//                    Opencv.resizeWindow(withName: "MyWindow", width: 800, height: 600)  // cv2.resizeWindow() fonksiyonu, genellikle OpenCV'nin yerel GUI penceresinde bir pencereyi yeniden boyutlandırmak için kullanılır. Ancak, bu işlev OpenCV'nin Swift'teki görüntüleme yöntemi olan UIImageView gibi Swift ve UIKit tabanlı yapılarda doğrudan geçerli değildir. Bunun yerine, bir UIImageView veya UIView boyutunu ayarlayarak Swift'te aynı etkiyi elde edebiliriz.
                    
//                    if let image = selectedImage {
//                        let processedImage = Opencv.applyThreshold(to: image, threshold: 127, maxValue: 255, thresholdType: 0)
//                        selectedImage = processedImage
//                        print("Threshold işlemi uygulandı.")
//                    }
                    
//                    if let image = selectedImage {
//                        let processedImage = Opencv.applyAdaptiveThreshold(to: image, maxValue: 255, adaptiveMethod: 1, thresholdType: 0, blockSize: 11, c: 2)
//                        selectedImage = processedImage
//                        print("Adaptive Threshold işlemi uygulandı.")
//                    }
                    
//                    if let image = selectedImage {
//                        let processedImage = Opencv.applyCanny(to: image, threshold1: 100, threshold2: 200)
//                        selectedImage = processedImage
//                        print("Canny işlemi uygulandı.")
//                    }
                    
//                    if let image = selectedImage {
//                        let laplacianImage = Opencv.laplacian(with: image, kernelSize: 3)
//                        selectedImage = laplacianImage
//                        print("laplacian işlemi uygulandı.")
//                    }
                    
//                    if let image = selectedImage {
//                        let boxFilter = Opencv.applyBoxFilter(to: image, ddepth: -1, ksize: CGSize(width: 5, height: 5))
//                        selectedImage = boxFilter
//                        print("boxfilter işlemi uygulandı.")
//                    }
//                    
//                    if let image = selectedImage {
//                        let scharrImage = Opencv.applyScharr(on: image)
//                        selectedImage = scharrImage
//                        print("scarr işlemi uygulandı.")
//                    }
                    
//                    if let image = selectedImage {
//                        let addImage = Opencv.add(UIImage(named: "izmir.jpg")!, with: UIImage(named: "izmir.jpg")!)
//                        selectedImage = addImage
//                        print("addImage işlemi uygulandı.")
//                    }
                    
//                    if let image = selectedImage {
//                        let subtrackImage = Opencv.subtract(UIImage(named: "izmir.jpg")!, from: UIImage(named: "izmir.jpg")!)
//                        selectedImage = subtrackImage
//                        print("subtrack işlemi uygulandı.")
//                    }
                    
//                    if let image = selectedImage {
//                        let multiplyImage = Opencv.multiply(UIImage(named: "izmir.jpg")!, with: UIImage(named: "izmir.jpg")!)
//                        selectedImage = multiplyImage
//                        print("multiply işlemi uygulandı.")
//                    }
                    
//                    if let image = selectedImage {
//                        let divideImage = Opencv.divide(UIImage(named: "izmir.jpg")!, by: (UIImage(named: "izmir.jpg")!))
//                        selectedImage = divideImage
//                        print("divide işlemi uygulandı.")
//                    }

//                    if let image = selectedImage {
//                        let erodedImage = Opencv.erodeImage(image, withKernelSize: 5)
//                        selectedImage = erodedImage
//                        print("erode işlemi uygulandı.")
//                    }
                    
//                    if let image = selectedImage {
//                        let dilatedImage = Opencv.dilateImage(image, withKernelSize: 5)
//                        selectedImage = dilatedImage
//                        print("dilate işlemi uygulandı.")
//                    }
                    
//                    if let image = selectedImage {
//                        let morphedImage = Opencv.applyMorphologyEx(image, withOperation: .gradient, kernelSize: 5)
//                        selectedImage = morphedImage
//                        print("morphologyEx işlemi uygulandı.")
//                    }
                    
//                    if let image = selectedImage {
//                        let kernelImage = Opencv.getStructuringElement(with: .rect, kernelSize: 5)
//                        selectedImage = kernelImage
//                        print("structuringElement işlemi uygulandı.")
//                    }
                    
//                    if let image = selectedImage, let contoursJSON = Opencv.findContours(in: image) {
////                        JSON'u Swift'te işlemeye uygun hale getir
//                        if let jsonData = contoursJSON.data(using: .utf8) {
//                            do {
////                                JSON verisini Swift dizisine dönüştür
//                                if let contoursArray = try JSONSerialization.jsonObject(with: jsonData, options: []) as? [[[String: CGFloat]]] {
////                                    Kontur noktalarını işleme (örneğin, her konturu çizme)
//                                    for contour in contoursArray {
//                                        for point in contour {
//                                            print("Point: \(point)")
//                                        }
//                                    }
//                                }
//                            } catch {
//                                print("JSON parsing error: \(error)")
//                            }
//                        }
//                    }
                        
//                    if let image = selectedImage, let countoursJSON = Opencv.findContours(in: image) {
//                        
////                        Konturları kırmızı renk ve 2 kalınlık ile çiz
//                        let countouredImage = Opencv.drawContours(on: image, withContours: countoursJSON, color: .red, thickness: 2)
//                        
//                        selectedImage = countouredImage
//                    }
                    
//                    if let image = selectedImage, let contoursJSON = Opencv.findContours(in: image) {
//                        
////                        İlk konturu çevresini hesaplamak için seçiyoruz
//                        if let jsonData = contoursJSON.data(using: .utf8),
//                           let contoursArray = try? JSONSerialization.jsonObject(with: jsonData, options: []) as? [[[String: CGFloat]]],
//                           let firstContour = contoursArray.first {
//                            
////                            İlk konturu JSON formatına dönüştür
//                            if let contourData = try? JSONSerialization.data(withJSONObject: firstContour, options: []),
//                               let contourJSONString = String(data: contourData, encoding: .utf8) {
//                                
////                                Konturun çevresini hesapla (kapalı olarak)
//                                let arcLenght = Opencv.arcLength(ofContour: contourJSONString, isClosed: true)
//                                print("Contour Arc Lenght: \(arcLenght)")
//                            }
//                        }
//                    }
                    
                    
//                    if let image = selectedImage,
//                       let contoursJSON = Opencv.findContours(in: image) {
//                                
//                                // İlk konturun alanını hesaplamak için seçiyoruz
//                                if let jsonData = contoursJSON.data(using: .utf8),
//                                   let contoursArray = try? JSONSerialization.jsonObject(with: jsonData, options: []) as? [[[String: CGFloat]]],
//                                   let firstContour = contoursArray.first {
//                                    
//                                    // İlk konturu JSON formatına dönüştür
//                                    if let contourData = try? JSONSerialization.data(withJSONObject: firstContour, options: []),
//                                       let contourJSONString = String(data: contourData, encoding: .utf8) {
//                                        
//                                        // Konturun alanını hesapla
//                                        let area = Opencv.contourArea(ofContour: contourJSONString)
//                                        print("Contour Area: \(area)")
//                                    }
//                                }
//                            }
                    
//                    if let image = selectedImage, let contoursJSON = Opencv.findContours(in: image) {
//                        
////                        İlk konturun basitleştirilmiş poligonunu hesaplamak için seçiyoruz
//                        if let jsonData = contoursJSON.data(using: .utf8),
//                           let contoursArray = try? JSONSerialization.jsonObject(with: jsonData, options: []) as? [[[String: CGFloat]]],
//                           let firstContour = contoursArray.first {
//                            
////                            İlk konturu JSON formatına dönüştür
//                            if let contourData = try? JSONSerialization.data(withJSONObject: firstContour, options: []),
//                               let contourJSONString = String(data: contourData, encoding: .utf8) {
//                                
////                                Yaklaşık poligonu hesapla
//                                let epsilon = 5.0 // Yaklaşım hassasiyeti
//                                if let approxContourJSON = Opencv.approxPolyDP(ofContour: contourJSONString, epsilon: epsilon, isClosed: true) {
//                                    
//                                    // Yaklaşık poligon JSON verisini işleyin veya kullanın
//                                    print("Approximate Polygon: \(approxContourJSON)")
//                                }
//                            }
//                        }
//                    }
                    
//                    if let image = selectedImage,
//                       let contoursJSON = Opencv.findContours(in: image) {
//                                
//                                // İlk konturun konveks hull'unu hesaplamak için seçiyoruz
//                                if let jsonData = contoursJSON.data(using: .utf8),
//                                   let contoursArray = try? JSONSerialization.jsonObject(with: jsonData, options: []) as? [[[String: CGFloat]]],
//                                   let firstContour = contoursArray.first {
//                                    
//                                    // İlk konturu JSON formatına dönüştür
//                                    if let contourData = try? JSONSerialization.data(withJSONObject: firstContour, options: []),
//                                       let contourJSONString = String(data: contourData, encoding: .utf8) {
//                                        
//                                        // Konveks hull'ü hesapla
//                                        if let convexHullJSON = Opencv.convexHull(ofContour: contourJSONString) {
//                                            
//                                            // Konveks hull JSON verisini işleyin veya kullanın
//                                            print("Convex Hull: \(convexHullJSON)")
//                                        }
//                                    }
//                                }
//                            }
                    
//                    if let image = selectedImage,
//                       let contoursJSON = Opencv.findContours(in: image){
//                                
//                                // İlk konturun konveks olup olmadığını kontrol etmek için seçiyoruz
//                                if let jsonData = contoursJSON.data(using: .utf8),
//                                   let contoursArray = try? JSONSerialization.jsonObject(with: jsonData, options: []) as? [[[String: CGFloat]]],
//                                   let firstContour = contoursArray.first {
//                                    
//                                    // İlk konturu JSON formatına dönüştür
//                                    if let contourData = try? JSONSerialization.data(withJSONObject: firstContour, options: []),
//                                       let contourJSONString = String(data: contourData, encoding: .utf8) {
//                                        
//                                        // Konturun konveks olup olmadığını kontrol et
//                                        let isConvex = Opencv.isContourConvex(contourJSONString)
//                                        print("Is the contour convex? \(isConvex)")
//                                    }
//                                }
//                            }
                    
//                                if let image = selectedImage,
//                                   let contoursJSON = Opencv.findContours(in: image) {
//                                
//                                // İlk konturun etrafındaki en küçük dikdörtgeni hesaplamak için seçiyoruz
//                                if let jsonData = contoursJSON.data(using: .utf8),
//                                   let contoursArray = try? JSONSerialization.jsonObject(with: jsonData, options: []) as? [[[String: CGFloat]]],
//                                   let firstContour = contoursArray.first {
//                                    
//                                    // İlk konturu JSON formatına dönüştür
//                                    if let contourData = try? JSONSerialization.data(withJSONObject: firstContour, options: []),
//                                       let contourJSONString = String(data: contourData, encoding: .utf8) {
//                                        
//                                        // Konturun etrafındaki en küçük dikdörtgeni hesapla
//                                        if let boundingRect = Opencv.boundingRect(ofContour: contourJSONString) as? [String: Any] {
//                                            // Dikdörtgen bilgilerini yazdır
//                                            if let x = boundingRect["x"] as? Int,
//                                               let y = boundingRect["y"] as? Int,
//                                               let width = boundingRect["width"] as? Int,
//                                               let height = boundingRect["height"] as? Int {
//                                                print("Bounding Rect: x=\(x), y=\(y), width=\(width), height=\(height)")
//                                            }
//                                        }
//                                    }
//                                }
//                            }
                    
//                    if let image = selectedImage,
//                       let contoursJSON = Opencv.findContours(in: image) {
//                                
//                                // İlk konturun minimum alanlı dikdörtgenini hesaplamak için seçiyoruz
//                                if let jsonData = contoursJSON.data(using: .utf8),
//                                   let contoursArray = try? JSONSerialization.jsonObject(with: jsonData, options: []) as? [[[String: CGFloat]]],
//                                   let firstContour = contoursArray.first {
//                                    
//                                    // İlk konturu JSON formatına dönüştür
//                                    if let contourData = try? JSONSerialization.data(withJSONObject: firstContour, options: []),
//                                       let contourJSONString = String(data: contourData, encoding: .utf8) {
//                                        
//                                        // Minimum alanlı dikdörtgeni hesapla
//                                        if let minRect = Opencv.minAreaRect(ofContour: contourJSONString) as? [String: Any] {
//                                            if let center = minRect["center"] as? [String: CGFloat],
//                                               let size = minRect["size"] as? [String: CGFloat],
//                                               let angle = minRect["angle"] as? CGFloat {
//                                                
//                                                let centerX = center["x"] ?? 0.0
//                                                let centerY = center["y"] ?? 0.0
//                                                let width = size["width"] ?? 0.0
//                                                let height = size["height"] ?? 0.0
//                                                
//                                                print("Minimum Area Rect: center=(\(centerX), \(centerY)), width=\(width), height=\(height), angle=\(angle)")
//                                            }
//                                        }
//                                    }
//                                }
//                            }
                        
//                    if let image = selectedImage,
//                       let contoursJSON = Opencv.findContours(in: image) {
//                                
//                                // İlk kontur üzerinde elips uydurmak için seçiyoruz
//                                if let jsonData = contoursJSON.data(using: .utf8),
//                                   let contoursArray = try? JSONSerialization.jsonObject(with: jsonData, options: []) as? [[[String: CGFloat]]],
//                                   let firstContour = contoursArray.first {
//                                    
//                                    // İlk konturu JSON formatına dönüştür
//                                    if let contourData = try? JSONSerialization.data(withJSONObject: firstContour, options: []),
//                                       let contourJSONString = String(data: contourData, encoding: .utf8) {
//                                        
//                                        // Elipsi uydur
//                                        if let ellipse = Opencv.fitEllipse(ofContour: contourJSONString) as? [String: Any] {
//                                            if let center = ellipse["center"] as? [String: CGFloat],
//                                               let size = ellipse["size"] as? [String: CGFloat],
//                                               let angle = ellipse["angle"] as? CGFloat {
//                                                
//                                                let centerX = center["x"] ?? 0.0
//                                                let centerY = center["y"] ?? 0.0
//                                                let majorAxis = size["majorAxis"] ?? 0.0
//                                                let minorAxis = size["minorAxis"] ?? 0.0
//                                                
//                                                print("Fitted Ellipse: center=(\(centerX), \(centerY)), majorAxis=\(majorAxis), minorAxis=\(minorAxis), angle=\(angle)")
//                                            }
//                                        }
//                                    }
//                                }
//                            }

//                    if let image = selectedImage,
//                       let contoursJSON = Opencv.findContours(in: image){
//                                
//                                // İlk kontur üzerinde doğru uydurmak için seçiyoruz
//                                if let jsonData = contoursJSON.data(using: .utf8),
//                                   let contoursArray = try? JSONSerialization.jsonObject(with: jsonData, options: []) as? [[[String: CGFloat]]],
//                                   let firstContour = contoursArray.first {
//                                    
//                                    // İlk konturu JSON formatına dönüştür
//                                    if let contourData = try? JSONSerialization.data(withJSONObject: firstContour, options: []),
//                                       let contourJSONString = String(data: contourData, encoding: .utf8) {
//                                        
//                                        // Doğruyu uydur
//                                        if let line = Opencv.fitLine(ofContour: contourJSONString) as? [String: Any] {
//                                            if let direction = line["direction"] as? [String: CGFloat],
//                                               let point = line["point"] as? [String: CGFloat] {
//                                                
//                                                let vx = direction["vx"] ?? 0.0
//                                                let vy = direction["vy"] ?? 0.0
//                                                let x = point["x"] ?? 0.0
//                                                let y = point["y"] ?? 0.0
//                                                
//                                                print("Fitted Line: direction=(\(vx), \(vy)), point=(\(x), \(y))")
//                                            }
//                                        }
//                                    }
//                                }
//                            }
                    
//                    if let image = selectedImage {
//                                
//                                // Parametreleri ayarlayın
//                                let maxCorners = 100
//                                let qualityLevel = 0.01
//                                let minDistance = 10.0
//                                
//                                // İyi özelliklere sahip noktaları tespit et
//                        if let features = Opencv.goodFeaturesToTrack(in: image, maxCorners: Int32(maxCorners), qualityLevel: qualityLevel, minDistance: minDistance) {
//                                    for feature in features {
//                                        if let x = feature["x"] as? CGFloat, let y = feature["y"] as? CGFloat {
//                                            print("Feature Point: (\(x), \(y))")
//                                        }
//                                    }
//                                }
//                            }
                    
//                    if let image = selectedImage {
//                                
//                                // Hough Dönüşümü parametreleri
//                                let rho = 1.0
//                                let theta = Double.pi / 180
//                                let threshold = 100
//                                
//                                // Çizgileri tespit et
//                        if let lines = Opencv.houghLines(in: image, rho: rho, theta: theta, threshold: Int32(threshold)) {
//                                    for line in lines {
//                                        if let rho = line["rho"] as? CGFloat, let theta = line["theta"] as? CGFloat {
//                                            print("Detected Line: rho = \(rho), theta = \(theta)")
//                                        }
//                                    }
//                                }
//                            }
                    
//                    if let image = selectedImage {
//                                
//                                // Hough Dönüşümü parametreleri
//                                let dp = 1.0
//                                let minDist = 20.0
//                                let param1 = 50.0
//                                let param2 = 30.0
//                                let mwdius = 0
//                                let maxRadius = 0
//                                
//                                // Çemberleri tespit et
//                        if let circles = Opencv.houghCircles(in: image, dp: dp, minDist: minDist, param1: param1, param2: param2, minRadius: Int32(minRadius), maxRadius: Int32(maxRadius)) {
//                                    for circle in circles {
//                                        if let centerX = circle["centerX"] as? CGFloat,
//                                           let centerY = circle["centerY"] as? CGFloat,
//                                           let radius = circle["radius"] as? CGFloat {
//                                            print("Detected Circle: center = (\(centerX), \(centerY)), radius = \(radius)")
//                                        }
//                                    }
//                                }
//                            }
                    
//                    if let image = selectedImage {
//                                
//                                // cornerHarris parametreleri
//                                let blockSize = 2
//                                let ksize = 3
//                                let k = 0.04
//                                
//                                // Köşe tespit edilmiş görüntüyü elde et
//                        if let corneredImage = Opencv.cornerHarris(in: image, blockSize: Int32(blockSize), ksize: Int32(ksize), k: k) {
//                            selectedImage = corneredImage
//                        }
//                            }
                    
//                    if let image = selectedImage {
//                        
//                        // Örnek anahtar noktaları (x, y) koordinatları
//                        let keypoints: [NSValue] = [ NSValue(cgPoint: CGPoint(x: 50, y: 50)),
//                                                    (NSValue(cgPoint: CGPoint(x: 100, y: 100))),
//                                                    (NSValue(cgPoint: CGPoint(x: 150, y: 150))) ]
//                        
//                        selectedImage = Opencv.drawKeypoints(on: image, keypoints: keypoints)
//                        print("keypoints çizildi : \(keypoints)")
//                    }
                    
//                    if let image = selectedImage, let templateImage = UIImage(named: "izmir.jpg") {
//                        selectedImage = Opencv.matchTemplate(in: image, templateImage: templateImage)
//                        print("template işlemi eklendi")
//                    }
                    
//                    if let image = UIImage(named: "izmir.jpg"), let nextImage = UIImage(named: "izmir.jpg") {
//                        selectedImage = Opencv.calculateOpticalFlow(from: image, to: nextImage)
//                    }
                    
//                    if let image = UIImage(named: "izmir.jpg"), let nextImage = UIImage(named: "izmir.jpg") {
//                        
//                        // Örnek anahtar noktaları (x, y) koordinatları
//                                let keypoints: [NSValue] = [NSValue(cgPoint: CGPoint(x: 50, y: 50)),
//                                                            NSValue(cgPoint: CGPoint(x: 100, y: 100)),
//                                                            NSValue(cgPoint: CGPoint(x: 150, y: 150))]
//                        
//                        selectedImage = Opencv.calculateOpticalFlowPyrLK(from: image, to: nextImage, keypoints: keypoints)
//                    }
                    
//                    if let image = selectedImage {
//                        selectedImage = Opencv.calculateMotionGradient(from: image)
//                    }
                    
//                    if let image = selectedImage {
//                        
//                        // Genel yönelimi hesapla
//                        self.globalOrientation = Opencv.calculateGlobalOrientation(from: image)
//                        print("globalOrientation: \(globalOrientation)")
//                    }

                    
//                    if let image = selectedImage {
//                        let nonZeroPoints = Opencv.findNonZero(with: image)
//                        
//                        for pointValue in nonZeroPoints {
//                            let point = pointValue.cgPointValue
//                            print("Non-zero pixel found at: \(point)")
//                        }
//                    }
                    
//                    let points1: [[NSNumber]] = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
//                    let points2: [[NSNumber]] = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
//                    let essentialMat = Opencv.findEssentialMat(withPoints1: points1, points2: points2)
//                        print("Essential Matrix: \(essentialMat)")
                    
//                    let essetialMat : [[NSNumber]] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
//                    let decomposedMat = Opencv.decomposeEssentialMat(withEssentialMat: essetialMat)
//                    print("Decomposed Matrices: \(decomposedMat)")
//
                    
//                    if let image = selectedImage {
//                        
//                        // Kamera matrisini ve distorsiyon katsayılarını tanımlayın
//                        let cameraMatrix: [NSNumber] = [1.0, 0.0, 320.0,
//                                                        0.0, 1.0, 240.0,
//                                                        0.0, 0.0, 1.0]
//                        let distCoeffs: [NSNumber] = [0.1, -0.1, 0.0, 0.0, 0.0]
//
//                        selectedImage = Opencv.undistortImage(image, withCameraMatrix: cameraMatrix, distCoeffs: distCoeffs)
//                    }
                    
//                    if let image = selectedImage {
//                        let showim = Opencv.processAndShow(image)
//                            selectedImage = showim
//                            print("resim gösterildi")
//                    }
                    
                    if let image = selectedImage {
                        // Alt ve üst sınırları tanımlayın (örneğin, mavi renk aralığı)
                        let lowerBound = [NSNumber(value: 100), NSNumber(value: 0), NSNumber(value: 0)]
                        let upperBound = [NSNumber(value: 255), NSNumber(value: 50), NSNumber(value: 50)]
                        
                        let inRangeImage = Opencv.inRange(with: image, lowerBound: lowerBound, upperBound: upperBound)
                        selectedImage = inRangeImage
                        print("inRange işlemi uygulandı")
                    }
                    
                    
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
}
