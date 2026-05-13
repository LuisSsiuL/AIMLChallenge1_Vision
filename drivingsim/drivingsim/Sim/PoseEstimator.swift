//
//  PoseEstimator.swift
//  drivingsim
//
//  Dead-reckoning pose integrator. Mirrors SimScene.tick physics so estimated
//  pose tracks the real car without using SimScene.carWorldPosition.
//
//  On real RC car: replace tickFromCommand with encoder-based odometry
//  (same math, encoder counts replace commanded velocity estimate).
//

import Combine
import Foundation
import simd

@MainActor
final class PoseEstimator: ObservableObject {
    @Published private(set) var pos: SIMD2<Float>   // (X, Z) world metres
    @Published private(set) var yaw: Float           // radians, 0 = facing -Z (north)

    // Must mirror SimScene exactly — same constants, same update law.
    private let maxSpeed:     Float = 0.6
    private let accel:        Float = 0.4
    private let brakeDecel:   Float = 0.8
    private let coastDrag:    Float = 0.4
    private let maxSteerRate: Float = 1.2
    private let steerSpeedRef:Float = 0.3

    private var speed: Float = 0.0

    init(startPos: SIMD2<Float> = SIMD2<Float>(0, 9.0),
         startYaw: Float = 0.0) {
        self.pos = startPos
        self.yaw = startYaw
    }

    /// Call once per tick (60Hz) with the WASD keys currently active.
    func tickFromCommand(keys: Set<UInt16>, dt: Float) {
        let fwd = keys.contains(KeyboardMonitor.W)
        let bwd = keys.contains(KeyboardMonitor.S)
        let lft = keys.contains(KeyboardMonitor.A)
        let rgt = keys.contains(KeyboardMonitor.D)

        // Throttle — same as SimScene.tick
        if fwd {
            speed = min(speed + accel * dt, maxSpeed)
        } else if bwd {
            speed = max(speed - brakeDecel * dt, -maxSpeed * 0.5)
        } else {
            let drag = coastDrag * dt
            speed = abs(speed) <= drag ? 0 : speed - drag * (speed > 0 ? 1 : -1)
        }

        // Steering — match SimScene.tick: pure rotation when A/D pressed
        // without W/S (skid-steer / in-place spin), otherwise speed-coupled.
        let steerInput: Float = (rgt ? 1 : 0) + (lft ? -1 : 0)
        let inPlaceRotate = (!fwd && !bwd && steerInput != 0)
        let yawRate: Float
        if inPlaceRotate {
            yawRate = steerInput * maxSteerRate
        } else {
            let speedFactor = min(1, abs(speed) / steerSpeedRef)
            yawRate = steerInput * maxSteerRate * speedFactor * (speed >= 0 ? 1 : -1)
        }
        yaw += yawRate * dt

        // Integrate position
        let fwdX =  sin(yaw)
        let fwdZ = -cos(yaw)
        pos.x += fwdX * speed * dt
        pos.y += fwdZ * speed * dt   // SIMD2 y = world Z
    }

    /// Apply an external correction (e.g. from ScanMatcher).
    func applyCorrection(deltaPos: SIMD2<Float>, deltaYaw: Float) {
        pos  += deltaPos
        yaw  += deltaYaw
    }

    func reset(pos newPos: SIMD2<Float>, yaw newYaw: Float) {
        pos   = newPos
        yaw   = newYaw
        speed = 0
    }
}
