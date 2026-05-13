//
//  DepthPreview.swift
//  drivingsim
//
//  Top-right preview of the colourised depth map + 7×7 zone overlay + command label.
//  Zone overlay: 30% top padding, 30% bottom padding, 50/50 FAR/NEAR split.
//  FAR row = F1–F7. NEAR row = N1–N7.
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

            // Zone overlay — matches live_depth_mlmodel_wasd_metric.py draw_nav_overlay:
            // 30% top padding (NAV_ROW_START), 30% bottom padding, 50/50 FAR/NEAR split.
            GeometryReader { geo in
                let w      = geo.size.width
                let h      = geo.size.height
                let row0    = h * 0.30            // 30% top padding (NAV_ROW_START)
                let bottom  = h - h * 0.30        // 30% bottom padding
                let roiH    = bottom - row0
                let farEnd  = row0 + roiH * 0.50  // 50/50 split for display
                let nearEnd = bottom

                let count = CGFloat(7)
                ZStack {
                    // FAR zones F1–F7
                    ForEach(Array(driver.farZones.enumerated()), id: \.offset) { i, z in
                        let x0  = w * CGFloat(i)     / count
                        let x1  = w * CGFloat(i + 1) / count
                        let bh  = farEnd - row0
                        let cy  = row0 + bh / 2
                        Rectangle()
                            .fill(zoneColor(z.state).opacity(0.28))
                            .frame(width: x1 - x0, height: bh)
                            .position(x: (x0 + x1) / 2, y: cy)
                        Rectangle()
                            .stroke(Color.white.opacity(0.5), lineWidth: 0.5)
                            .frame(width: x1 - x0, height: bh)
                            .position(x: (x0 + x1) / 2, y: cy)
                        Text("\(Int(z.obstacleScore))")
                            .font(.system(size: 8, weight: .bold, design: .monospaced))
                            .foregroundColor(.white)
                            .shadow(color: .black, radius: 1)
                            .position(x: (x0 + x1) / 2, y: cy)
                    }
                    // NEAR zones N1–N7
                    ForEach(Array(driver.nearZones.enumerated()), id: \.offset) { i, z in
                        let x0  = w * CGFloat(i)     / count
                        let x1  = w * CGFloat(i + 1) / count
                        let bh  = nearEnd - farEnd
                        let cy  = farEnd + bh / 2
                        Rectangle()
                            .fill(zoneColor(z.state).opacity(0.28))
                            .frame(width: x1 - x0, height: bh)
                            .position(x: (x0 + x1) / 2, y: cy)
                        Rectangle()
                            .stroke(Color.white.opacity(0.5), lineWidth: 0.5)
                            .frame(width: x1 - x0, height: bh)
                            .position(x: (x0 + x1) / 2, y: cy)
                        Text("\(Int(z.obstacleScore))")
                            .font(.system(size: 8, weight: .bold, design: .monospaced))
                            .foregroundColor(.white)
                            .shadow(color: .black, radius: 1)
                            .position(x: (x0 + x1) / 2, y: cy)
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
