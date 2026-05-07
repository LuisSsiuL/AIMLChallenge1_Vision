//
//  SimScene.swift
//  drivingsim
//
//  Indoor office room sim at small-RC-car scale (~15×20cm car, FPV camera ~6cm
//  above floor). Procedural — no external assets.
//

import AppKit
import Combine
import RealityKit
import simd

@MainActor
final class SimScene {
    let root = Entity()
    private(set) var car: ModelEntity!
    private var cam: PerspectiveCamera!

    // Car tunables — RC-car scale, indoor speeds.
    private let maxSpeed:      Float = 1.5    // ~5.4 km/h, slow indoor RC
    private let accel:         Float = 1.0
    private let brakeDecel:    Float = 2.0
    private let coastDrag:     Float = 1.0
    private let maxSteerRate:  Float = 1.6
    private let steerSpeedRef: Float = 0.5    // full steer at lower speed (small car)

    // FPV camera — RC car eye-line ~6cm above floor.
    private let eyeHeight:     Float = 0.06
    private let eyeForward:    Float = 0.0

    // Car AABB (15cm wide, 20cm long).
    private let carHalfX: Float = 0.075
    private let carHalfZ: Float = 0.10
    private let carBodyY: Float = 0.03   // centre of car body above floor

    // State
    private var speed: Float = 0
    private var yaw:   Float = 0

    // Obstacles — single source of truth for render + collision.
    private struct Obstacle { let pos: SIMD3<Float>; let halfX: Float; let halfZ: Float; let height: Float }
    private var obstacles: [Obstacle] = []

    /// Public read-only snapshot for SimFPVRenderer mirror scene.
    var obstacleSnapshot: [(pos: SIMD3<Float>, halfX: Float, halfZ: Float, height: Float)] {
        obstacles.map { ($0.pos, $0.halfX, $0.halfZ, $0.height) }
    }

    var eyeHeightPublic: Float { eyeHeight }
    var carWorldPosition: SIMD3<Float> { car?.position ?? .zero }
    var carYaw: Float { yaw }

    private var timer: Timer?

    static func make() -> SimScene {
        let s = SimScene()
        s.setup()
        return s
    }

    // MARK: - Scene setup

    private func setup() {
        addGround()
        setupOfficeRoom()
        addCar()
        addLights()
        addCamera()
    }

    private func addGround() {
        // Big outdoor ground — visible through the doorway, suggests the office is on a lawn.
        let groundSize: Float = 60
        let groundMesh = MeshResource.generateBox(width: groundSize, height: 0.02, depth: groundSize)
        let groundMat  = UnlitMaterial(color: NSColor(red: 0.30, green: 0.42, blue: 0.30, alpha: 1))
        let ground     = ModelEntity(mesh: groundMesh, materials: [groundMat])
        ground.position = [0, -0.011, 0]
        root.addChild(ground)

        // Wood floor inside the office (visual cue only — not a collider).
        let floorMat = UnlitMaterial(color: NSColor(red: 0.60, green: 0.45, blue: 0.30, alpha: 1))
        let floorMesh = MeshResource.generateBox(width: 6.0, height: 0.005, depth: 4.0)
        let floor = ModelEntity(mesh: floorMesh, materials: [floorMat])
        floor.position = [0, 0.003, 0]
        root.addChild(floor)
    }

    private func setupOfficeRoom() {
        // Room: 6m W × 4m D × 2.5m H, centred on origin.
        // South wall has a 2m doorway centred at x=0.
        let wallH: Float = 2.5
        let wallT: Float = 0.10   // wall thickness
        let roomW: Float = 6.0
        let roomD: Float = 4.0
        let halfW = roomW / 2
        let halfD = roomD / 2

        let wallColor    = NSColor(red: 0.72, green: 0.78, blue: 0.82, alpha: 1)
        let cabinetColor = NSColor(white: 0.78, alpha: 1)
        let deskColor    = NSColor(red: 0.40, green: 0.26, blue: 0.16, alpha: 1)
        let chairColor   = NSColor(white: 0.18, alpha: 1)
        let binColors: [NSColor] = [.systemRed, .systemBlue, .systemOrange, .systemTeal, .systemYellow]

        // ── Walls ─────────────────────────────────────────────────────────
        // North wall (z = -halfD)
        addBox(at: [0, wallH/2, -halfD - wallT/2], w: roomW + wallT*2, h: wallH, d: wallT, color: wallColor)
        // East wall (x = +halfW)
        addBox(at: [halfW + wallT/2, wallH/2, 0], w: wallT, h: wallH, d: roomD, color: wallColor)
        // West wall (x = -halfW)
        addBox(at: [-halfW - wallT/2, wallH/2, 0], w: wallT, h: wallH, d: roomD, color: wallColor)
        // South wall — split for 2m doorway centred at x=0.
        let doorHalf: Float = 1.0
        let southSegW = halfW - doorHalf
        addBox(at: [-halfW + southSegW/2, wallH/2, halfD + wallT/2], w: southSegW, h: wallH, d: wallT, color: wallColor)
        addBox(at: [ halfW - southSegW/2, wallH/2, halfD + wallT/2], w: southSegW, h: wallH, d: wallT, color: wallColor)

        // ── Cabinets along north wall (just inside) ───────────────────────
        let cabH: Float = 1.8
        let cabD: Float = 0.50
        let cabW: Float = 1.0
        let cabZ: Float = -halfD + cabD/2 + 0.05
        for x in stride(from: -2.0 as Float, through: 2.0, by: 2.0) {
            addBox(at: [x, cabH/2, cabZ], w: cabW, h: cabH, d: cabD, color: cabinetColor)
        }

        // ── Desks: 2 rows of 3 ───────────────────────────────────────────
        let deskH: Float = 0.75
        let deskW: Float = 1.5
        let deskD: Float = 0.70
        // North row (closer to cabinets)
        let northRowZ: Float = -0.6
        for x in stride(from: -2.0 as Float, through: 2.0, by: 2.0) {
            addBox(at: [x, deskH/2, northRowZ], w: deskW, h: deskH, d: deskD, color: deskColor)
        }
        // South row
        let southRowZ: Float = 0.6
        for x in stride(from: -2.0 as Float, through: 2.0, by: 2.0) {
            addBox(at: [x, deskH/2, southRowZ], w: deskW, h: deskH, d: deskD, color: deskColor)
        }

        // ── Chairs (5 — skip centre south-row chair for car spawn safety) ─
        let chairH: Float = 0.95
        let chairS: Float = 0.50   // square footprint
        let northChairZ: Float = northRowZ + deskD/2 + 0.30 + chairS/2   // ~ -0.05
        for x in stride(from: -2.0 as Float, through: 2.0, by: 2.0) {
            addBox(at: [x, chairH/2, northChairZ], w: chairS, h: chairH, d: chairS, color: chairColor)
        }
        let southChairZ: Float = southRowZ + deskD/2 + 0.30 + chairS/2   // ~ 1.20
        for x in [-2.0 as Float, 2.0] {  // skip x=0 — leave aisle for car
            addBox(at: [x, chairH/2, southChairZ], w: chairS, h: chairH, d: chairS, color: chairColor)
        }

        // ── Clutter bins / boxes (5 scattered, avoiding everything above) ─
        // Hand-picked clear positions:
        let binPositions: [(SIMD3<Float>, Float, Float, Float)] = [
            // (centre, w, h, d)
            ([-2.6, 0.20, 0.0],  0.30, 0.40, 0.30),   // west aisle, between rows
            ([ 2.6, 0.20, 0.0],  0.30, 0.40, 0.30),   // east aisle
            ([-2.6, 0.20, -1.2], 0.30, 0.40, 0.30),   // west, between cabinet and desk row
            ([ 2.6, 0.20, -1.2], 0.30, 0.40, 0.30),   // east, between cabinet and desk row
            ([ 0.0, 0.15, -1.85], 0.40, 0.30, 0.40),  // small box pushed against north cabinets
        ]
        for (i, p) in binPositions.enumerated() {
            addBox(at: p.0, w: p.1, h: p.2, d: p.3, color: binColors[i % binColors.count])
        }
    }

    /// Add a box-shaped obstacle: registers visual entity AND AABB collision record.
    private func addBox(at pos: SIMD3<Float>, w: Float, h: Float, d: Float, color: NSColor) {
        let mesh = MeshResource.generateBox(width: w, height: h, depth: d)
        let mat  = SimpleMaterial(color: color, isMetallic: false)
        let ent  = ModelEntity(mesh: mesh, materials: [mat])
        ent.position = pos
        root.addChild(ent)
        obstacles.append(Obstacle(pos: pos, halfX: w / 2, halfZ: d / 2, height: h))
    }

    private func addCar() {
        // Tiny invisible proxy — FPV camera sits above, car body never visually rendered.
        let carMesh = MeshResource.generateSphere(radius: 0.001)
        car = ModelEntity(mesh: carMesh, materials: [SimpleMaterial(color: .clear, isMetallic: false)])
        // Spawn near south doorway (z = +1.8), facing -Z (into the room).
        car.position    = [0, carBodyY, 1.8]
        car.orientation = simd_quatf(angle: 0, axis: [0, 1, 0])
        yaw = 0
        root.addChild(car)
    }

    private func addLights() {
        let key = DirectionalLight()
        key.light.intensity = 6000
        key.orientation = simd_quatf(angle: -.pi / 4, axis: [1, 0, 0])
        root.addChild(key)
        let fill = DirectionalLight()
        fill.light.intensity = 2500
        fill.orientation = simd_quatf(angle: .pi / 5, axis: [-1, 0, 0])
        root.addChild(fill)
    }

    private func addCamera() {
        cam = PerspectiveCamera()
        cam.camera.fieldOfViewInDegrees = 75
        cam.position = [0, carBodyY + eyeHeight, 1.8]
        cam.look(at: [0, carBodyY + eyeHeight, -1], from: cam.position, relativeTo: nil)
        root.addChild(cam)
    }

    // MARK: - Per-frame tick

    func startUpdates(keyboard: KeyboardMonitor,
                      hand: HandJoystick,
                      depth: DepthDriver,
                      modeProvider: @escaping @MainActor () -> DrivingMode) {
        guard timer == nil else { return }
        let t = Timer(timeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in
                let m = modeProvider()
                self.tick(kb: keyboard, hand: hand, depth: depth, mode: m, dt: 1.0 / 60.0)
            }
        }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    func stopUpdates() {
        timer?.invalidate()
        timer = nil
    }

    private func tick(kb: KeyboardMonitor,
                      hand: HandJoystick,
                      depth: DepthDriver,
                      mode: DrivingMode,
                      dt: Float) {
        // Input gating per mode:
        //   .off       — kb only
        //   .hand      — kb + hand
        //   .assisted  — depth → W/S; (kb||hand) → A/D
        //   .automated — depth → all four; kb/hand ignored
        let kbActive   = (mode != .automated)
        let handActive = (mode == .hand || mode == .assisted)
        let depthWS    = (mode == .assisted || mode == .automated)
        let depthAD    = (mode == .automated)

        let inForward  = (kbActive && kb.forward)  || (handActive && hand.forward)  || (depthWS && depth.forward)
        let inBackward = (kbActive && kb.backward) || (handActive && hand.backward) || (depthWS && depth.backward)
        let inLeft     = (kbActive && kb.left)     || (handActive && hand.left)     || (depthAD && depth.left)
        let inRight    = (kbActive && kb.right)    || (handActive && hand.right)    || (depthAD && depth.right)

        // ── Throttle / brake ──
        if inForward {
            speed = min(speed + accel * dt, maxSpeed)
        } else if inBackward {
            speed = max(speed - brakeDecel * dt, -maxSpeed * 0.5)
        } else {
            let drag = coastDrag * dt
            speed = abs(speed) <= drag ? 0 : speed - drag * (speed > 0 ? 1 : -1)
        }

        // ── Steering ──
        let speedFactor = min(1, abs(speed) / steerSpeedRef)
        let steerInput: Float = (inRight ? 1 : 0) + (inLeft ? -1 : 0)
        let yawRate     = steerInput * maxSteerRate * speedFactor * (speed >= 0 ? 1 : -1)
        yaw += yawRate * dt

        let fwdX =  sin(yaw)
        let fwdZ = -cos(yaw)
        let fwd  = SIMD3<Float>(fwdX, 0, fwdZ)

        // ── Move car ──
        var nextPos = car.position + fwd * speed * dt
        nextPos.y = carBodyY

        // AABB collision
        for o in obstacles {
            let dx = nextPos.x - o.pos.x
            let dz = nextPos.z - o.pos.z
            let overlapX = (carHalfX + o.halfX) - abs(dx)
            let overlapZ = (carHalfZ + o.halfZ) - abs(dz)
            if overlapX > 0 && overlapZ > 0 {
                if overlapX < overlapZ {
                    nextPos.x += overlapX * (dx >= 0 ? 1 : -1)
                } else {
                    nextPos.z += overlapZ * (dz >= 0 ? 1 : -1)
                }
                speed = 0
            }
        }

        car.position    = nextPos
        car.orientation = simd_quatf(angle: yaw, axis: [0, 1, 0])

        // ── FPV camera ──
        let eyePos = car.position + SIMD3<Float>(fwdX * eyeForward, eyeHeight, fwdZ * eyeForward)
        let lookTarget = eyePos + fwd * 10
        cam.position = eyePos
        cam.look(at: lookTarget, from: eyePos, relativeTo: nil)
    }
}
