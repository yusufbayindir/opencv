//
//  BFMarhcher.swift
//  opencv
//
//  Created by Emrehan Kaya on 29.10.2024.
//

import SwiftUI

struct BFMarhcherAndFlannBased: View {
    @State private var bfMatches: [(queryIdx: Int, trainIdx: Int, distance: Float)] = []
    @State private var flannMatches: [(queryIdx: Int, trainIdx: Int, distance: Float)] = []

    var body: some View {
        VStack {
            Text("BFMatcher Results:")
            if !bfMatches.isEmpty {
                ForEach(bfMatches, id: \.queryIdx) { match in
                    Text("Query Index: \(match.queryIdx), Train Index: \(match.trainIdx), Distance: \(match.distance)")
                }
            } else {
                Text("No BFMatcher matches found.")
            }

            Text("FlannBasedMatcher Results:")
            if !flannMatches.isEmpty {
                ForEach(flannMatches, id: \.queryIdx) { match in
                    Text("Query Index: \(match.queryIdx), Train Index: \(match.trainIdx), Distance: \(match.distance)")
                }
            } else {
                Text("No FlannBasedMatcher matches found.")
            }
        }
        .onAppear {
            processImages()
        }
    }
    
    func processImages() {
        let descriptors1: [[NSNumber]] = [[1, 2, 3], [4, 5, 6]] // Örnek descriptor
        let descriptors2: [[NSNumber]] = [[1, 2, 3], [7, 8, 9]] // Örnek descriptor
        
        // BFMatcher ile eşleştir
        let bfMatchesArray = Opencv.matchKeypoints(withBFMatcherDescriptors1: descriptors1, descriptors2: descriptors2)
        self.bfMatches = bfMatchesArray.compactMap {
            guard let queryIdx = $0["queryIdx"] as? Int,
                  let trainIdx = $0["trainIdx"] as? Int,
                  let distance = $0["distance"] as? Float else { return nil }
            return (queryIdx: queryIdx, trainIdx: trainIdx, distance: distance)
        }
        
        // FlannBasedMatcher ile eşleştir
        let flannMatchesArray = Opencv.matchKeypoints(withFlannMatcherDescriptors1: descriptors1, descriptors2: descriptors2)
        self.flannMatches = flannMatchesArray.compactMap {
            guard let queryIdx = $0["queryIdx"] as? Int,
                  let trainIdx = $0["trainIdx"] as? Int,
                  let distance = $0["distance"] as? Float else { return nil }
            return (queryIdx: queryIdx, trainIdx: trainIdx, distance: distance)
        }
    }
}
