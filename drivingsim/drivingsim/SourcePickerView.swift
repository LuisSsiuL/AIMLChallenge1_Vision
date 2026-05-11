//
//  SourcePickerView.swift
//  drivingsim
//
//  Launch screen — pick the frame source (Sim or a live AVCapture device,
//  including Continuity Camera from a paired iPhone). Calls `onPick` with
//  the chosen source.
//

import SwiftUI

struct SourcePickerView: View {
    let onPick: (CameraSource, ESP32Profile) -> Void

    @State private var sources: [CameraSource] = []
    @State private var selection: CameraSource?
    @State private var esp32: ESP32Profile = .none

    var body: some View {
        VStack(spacing: 18) {
            Text("drivingsim")
                .font(.system(size: 28, weight: .bold, design: .monospaced))
            Text("Pick a frame source")
                .font(.callout)
                .foregroundColor(.secondary)

            if sources.isEmpty {
                ProgressView("Scanning cameras…")
                    .frame(maxWidth: .infinity, minHeight: 120)
            } else {
                VStack(spacing: 8) {
                    ForEach(sources) { src in
                        Button {
                            selection = src
                        } label: {
                            HStack {
                                Image(systemName: iconName(for: src))
                                    .font(.system(size: 18))
                                    .frame(width: 28)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(src.label)
                                        .font(.system(size: 14, weight: .semibold))
                                    Text(subtitle(for: src))
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                }
                                Spacer()
                                if selection == src {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundColor(.accentColor)
                                }
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 10)
                            .frame(maxWidth: .infinity)
                            .background(selection == src ? Color.accentColor.opacity(0.15) : Color.clear)
                            .overlay(RoundedRectangle(cornerRadius: 8)
                                .stroke(selection == src ? Color.accentColor : Color.secondary.opacity(0.3),
                                        lineWidth: 1))
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                }
            }

            // ESP32-CAM resolution emulation (applies only to live camera sources).
            VStack(alignment: .leading, spacing: 4) {
                Text("Stream profile (live cameras only)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Picker("Profile", selection: $esp32) {
                    ForEach(ESP32Profile.allCases) { p in
                        Text(p.label).tag(p)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .disabled(selection == .sim)
            }

            HStack {
                Button("Rescan") { rescan() }
                    .keyboardShortcut("r")
                Spacer()
                Button("Continue") {
                    if let s = selection { onPick(s, esp32) }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(selection == nil)
            }
        }
        .padding(24)
        .frame(width: 460, height: 540)
        .onAppear { rescan() }
    }

    private func rescan() {
        let list = CameraScanner.discover()
        sources = list
        if selection == nil { selection = list.first }
    }

    private func iconName(for src: CameraSource) -> String {
        switch src {
        case .sim: return "cube.transparent"
        case .live(_, let name):
            let n = name.lowercased()
            if n.contains("iphone")  { return "iphone" }
            if n.contains("display") { return "display" }
            return "camera"
        }
    }

    private func subtitle(for src: CameraSource) -> String {
        switch src {
        case .sim: return "Procedural 3D office, RC-car FPV"
        case .live: return "AVCaptureDevice"
        }
    }
}
