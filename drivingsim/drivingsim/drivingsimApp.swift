//
//  drivingsimApp.swift
//  drivingsim
//

import SwiftUI

@main
struct drivingsimApp: App {
    @State private var chosenSource: CameraSource?

    var body: some Scene {
        WindowGroup {
            if let src = chosenSource {
                ContentView(source: src)
            } else {
                SourcePickerView { picked in
                    chosenSource = picked
                }
            }
        }
    }
}
