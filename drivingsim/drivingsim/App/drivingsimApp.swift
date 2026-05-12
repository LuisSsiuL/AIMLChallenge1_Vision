//
//  drivingsimApp.swift
//  drivingsim
//

import SwiftUI

@main
struct drivingsimApp: App {
    @State private var chosenSource: CameraSource?
    @State private var chosenProfile: ESP32Profile = .none

    var body: some Scene {
        WindowGroup {
            if let src = chosenSource {
                ContentView(source: src, esp32Profile: chosenProfile)
            } else {
                SourcePickerView { picked, profile in
                    chosenProfile = profile
                    chosenSource  = picked
                }
            }
        }
    }
}
