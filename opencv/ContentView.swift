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
                    print("İkinci buton tıklandı")
                }) {
                    Text("Buton 2")
                        .padding()
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                
                Button(action: {
                    // Üçüncü butonun etkisi
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
                        }
                    }
                }
            }
        }
    }
}
