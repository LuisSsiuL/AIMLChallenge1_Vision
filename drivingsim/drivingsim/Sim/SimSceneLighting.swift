//
//  SimSceneLighting.swift
//  drivingsim
//
//  Shared indoor-office lighting rig for SimScene (on-screen) and
//  SimFPVRenderer (offscreen). Both views must light the world identically
//  so the depth model sees what the operator sees.
//
//  Rig:
//   • 6 ceiling point lights distributed 3 × 2 across the 30 × 20 m room
//     (cool-white office luminaires). Casts no shadows (RealityKit point
//     lights don't shadow). Provides position-dependent falloff so stacked
//     boxes get differential illumination instead of flat washing.
//   • 1 weak overhead directional light WITH shadow — gives geometric edge
//     cues to the depth model. Aimed slightly off-axis so silhouettes show.
//   • 1 low fill from the side to reduce crushed blacks.
//

import AppKit
import Foundation
import RealityKit
import simd

enum SimSceneLighting {

    /// Ceiling height for point lights. Room wallH=3m → lights at 2.8m.
    static let ceilingY: Float = 2.8
    /// Distribution: 3 cols × 2 rows of point lights.
    static let xPositions: [Float] = [-10, 0, 10]
    static let zPositions: [Float] = [-6, 6]
    /// Per-lamp brightness (lumens-ish in RealityKit's arbitrary scale).
    static let lampIntensity: Float = 100_000
    /// Falloff radius — beyond this, lamp contributes ~0.
    static let lampAttenuationRadius: Float = 9.0

    /// Attach lighting to a SimScene root entity (on-screen RealityView).
    static func attach(to root: Entity, addShadowKey: Bool) {
        let lamps = makeCeilingLamps()
        for l in lamps { root.addChild(l) }
        if addShadowKey {
            root.addChild(makeShadowKey())
            root.addChild(makeSideFill())
        }
    }

    /// Attach lighting to a RealityRenderer (offscreen FPV).
    @MainActor
    static func attach(toRenderer renderer: RealityRenderer, addShadowKey: Bool) {
        let lamps = makeCeilingLamps()
        for l in lamps { renderer.entities.append(l) }
        if addShadowKey {
            renderer.entities.append(makeShadowKey())
            renderer.entities.append(makeSideFill())
        }
    }

    // MARK: - Builders

    private static func makeCeilingLamps() -> [Entity] {
        var out: [Entity] = []
        for x in xPositions {
            for z in zPositions {
                let lamp = PointLight()
                lamp.light.intensity = lampIntensity
                lamp.light.attenuationRadius = lampAttenuationRadius
                // Slight cool-white office tone.
                lamp.light.color = .init(red: 0.95, green: 0.97, blue: 1.0, alpha: 1.0)
                lamp.position = [x, ceilingY, z]
                out.append(lamp)
            }
        }
        return out
    }

    private static func makeShadowKey() -> Entity {
        let key = DirectionalLight()
        key.light.intensity = 2000
        // Aimed downward + slight angle so vertical surfaces get edge shading.
        key.orientation = simd_quatf(angle: -.pi / 3, axis: [1, 0, 0])
        key.shadow = DirectionalLightComponent.Shadow()
        return key
    }

    private static func makeSideFill() -> Entity {
        let fill = DirectionalLight()
        fill.light.intensity = 800
        // From the side to lift crushed shadows on stacked obstacles.
        fill.orientation = simd_quatf(angle: .pi / 4, axis: [0, 1, 0])
        return fill
    }
}
