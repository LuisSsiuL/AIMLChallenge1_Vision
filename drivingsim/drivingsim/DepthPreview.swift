//
//  DepthPreview.swift
//  drivingsim
//
//  Top-right preview of the colourised depth map + 5-zone overlay + command label.
//

import SwiftUI

struct DepthPreview: View {
    @ObservedObject var driver: DepthDriver

    var body: some View {
        ZStack(alignment: .topLeading) {
            // Depth image
            if let img = driver.depthImage {
                Image(decorative: img, scale: 1, orientation: .up)
                    .resizable()
                    .interpolation(.low)
                    .aspectRatio(contentMode: .fit)
            } else {
                Color.black.overlay(
                    Text("Loading depth…")
                        .foregroundColor(.white.opacity(0.6))
                        .font(.caption2)
                )
            }

            // Zone overlay
            GeometryReader { geo in
                let w = geo.size.width
                let h = geo.size.height
                let row0 = h * 0.30   // navRowStart
                ZStack {
                    ForEach(Array(driver.zones.enumerated()), id: \.offset) { i, z in
                        let x0 = w * CGFloat(i) / 5.0
                        let x1 = w * CGFloat(i + 1) / 5.0
                        Rectangle()
                            .fill(zoneColor(z.state).opacity(0.28))
                            .frame(width: x1 - x0, height: h - row0)
                            .position(x: (x0 + x1) / 2, y: row0 + (h - row0) / 2)
                            .overlay(
                                Rectangle()
                                    .stroke(Color.white.opacity(0.6), lineWidth: 0.5)
                                    .frame(width: x1 - x0, height: h - row0)
                                    .position(x: (x0 + x1) / 2, y: row0 + (h - row0) / 2)
                            )
                        Text("\(Int(z.obstacleScore))")
                            .font(.system(size: 9, weight: .bold, design: .monospaced))
                            .foregroundColor(.white)
                            .shadow(color: .black, radius: 1)
                            .position(x: (x0 + x1) / 2, y: row0 + (h - row0) / 2)
                    }
                }
            }

            // Command label
            VStack {
                HStack {
                    Spacer()
                    Text(driver.command.label)
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundColor(.white)
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Color.black.opacity(0.55))
                        .cornerRadius(4)
                        .padding(4)
                }
                Spacer()
            }
        }
    }

    private func zoneColor(_ s: ZoneState) -> Color {
        switch s {
        case .clear:     return .green
        case .uncertain: return .yellow
        case .blocked:   return .red
        }
    }
}
