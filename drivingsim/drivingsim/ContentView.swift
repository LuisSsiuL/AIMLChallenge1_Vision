//
//  ContentView.swift
//  drivingsim
//
//  Created by Christian Luis Efendy on 06/05/26.
//

import SwiftUI
import RealityKit

private struct KeyCapView: View {
    let label: String
    let pressed: Bool

    var body: some View {
        Text(label)
            .font(.system(size: 14, weight: .bold, design: .monospaced))
            .foregroundColor(pressed ? .black : .white)
            .frame(width: 36, height: 36)
            .background(Color.white.opacity(pressed ? 0.9 : 0.15))
            .cornerRadius(6)
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.white.opacity(0.5), lineWidth: 1))
    }
}

struct ContentView: View {
    @StateObject private var keyboard = KeyboardMonitor()
    // Hold scene outside of RealityView make so timer outlives re-renders
    @State private var scene = SimScene.make()

    var body: some View {
        RealityView { content in
            content.add(scene.root)
            scene.startUpdates(keyboard: keyboard)
        } update: { _ in }
        .frame(minWidth: 900, minHeight: 600)
        .onAppear  { keyboard.start() }
        .onDisappear {
            keyboard.stop()
            scene.stopUpdates()
        }
        .overlay(alignment: .bottomLeading) {
            Text("W/S — accelerate / brake     A/D — steer")
                .font(.caption)
                .foregroundColor(.white)
                .padding(8)
                .background(.black.opacity(0.4))
                .cornerRadius(6)
                .padding(12)
        }
        .overlay(alignment: .bottomTrailing) {
            VStack(spacing: 4) {
                HStack {
                    KeyCapView(label: "W", pressed: keyboard.forward)
                }
                HStack(spacing: 4) {
                    KeyCapView(label: "A", pressed: keyboard.left)
                    KeyCapView(label: "S", pressed: keyboard.backward)
                    KeyCapView(label: "D", pressed: keyboard.right)
                }
            }
            .padding(12)
        }
    }
}
