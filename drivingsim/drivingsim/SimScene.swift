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
    private let maxSpeed:      Float = 3.0    // ~11 km/h, brisk indoor RC
    private let accel:         Float = 2.0
    private let brakeDecel:    Float = 4.0
    private let coastDrag:     Float = 1.5
    private let maxSteerRate:  Float = 1.8
    private let steerSpeedRef: Float = 0.8    // full steer at low speed

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
        let groundSize: Float = 200
        let groundMesh = MeshResource.generateBox(width: groundSize, height: 0.02, depth: groundSize)
        let groundMat  = UnlitMaterial(color: NSColor(red: 0.30, green: 0.42, blue: 0.30, alpha: 1))
        let ground     = ModelEntity(mesh: groundMesh, materials: [groundMat])
        ground.position = [0, -0.011, 0]
        root.addChild(ground)

        // Office floor (large open-plan, wood-tone, fully unlit so no reflections).
        let floorMat = UnlitMaterial(color: NSColor(red: 0.55, green: 0.42, blue: 0.30, alpha: 1))
        let floorMesh = MeshResource.generateBox(width: 30.0, height: 0.005, depth: 20.0)
        let floor = ModelEntity(mesh: floorMesh, materials: [floorMat])
        floor.position = [0, 0.003, 0]
        root.addChild(floor)
    }

    private func setupOfficeRoom() {
        // Open-plan office: 30m W × 20m D × 3m H, centred on origin.
        // South wall has a 4m doorway centred at x=0.
        let wallH: Float = 3.0
        let wallT: Float = 0.15
        let roomW: Float = 30.0
        let roomD: Float = 20.0
        let halfW = roomW / 2
        let halfD = roomD / 2

        let wallColor    = NSColor(red: 0.78, green: 0.80, blue: 0.82, alpha: 1)
        let cabinetColor = NSColor(white: 0.72, alpha: 1)
        let deskColor    = NSColor(red: 0.42, green: 0.28, blue: 0.18, alpha: 1)
        let chairColor   = NSColor(white: 0.20, alpha: 1)
        let binColors: [NSColor] = [
            NSColor(red: 0.75, green: 0.30, blue: 0.30, alpha: 1),  // muted red
            NSColor(red: 0.30, green: 0.45, blue: 0.70, alpha: 1),  // muted blue
            NSColor(red: 0.80, green: 0.60, blue: 0.25, alpha: 1),  // muted orange
            NSColor(red: 0.30, green: 0.65, blue: 0.55, alpha: 1),  // muted teal
            NSColor(red: 0.55, green: 0.50, blue: 0.30, alpha: 1),  // muted olive
        ]

        // ── Walls ─────────────────────────────────────────────────────────
        addBox(at: [0, wallH/2, -halfD - wallT/2], w: roomW + wallT*2, h: wallH, d: wallT, color: wallColor)
        addBox(at: [halfW + wallT/2, wallH/2, 0],  w: wallT, h: wallH, d: roomD, color: wallColor)
        addBox(at: [-halfW - wallT/2, wallH/2, 0], w: wallT, h: wallH, d: roomD, color: wallColor)
        // South wall — 4m doorway centred at x=0.
        let doorHalf: Float = 2.0
        let southSegW = halfW - doorHalf
        addBox(at: [-halfW + southSegW/2, wallH/2, halfD + wallT/2], w: southSegW, h: wallH, d: wallT, color: wallColor)
        addBox(at: [ halfW - southSegW/2, wallH/2, halfD + wallT/2], w: southSegW, h: wallH, d: wallT, color: wallColor)

        // ── Cabinets along north wall (6, evenly spaced) ──────────────────
        let cabH: Float = 1.8
        let cabD: Float = 0.50
        let cabW: Float = 1.2
        let cabZ: Float = -halfD + cabD/2 + 0.10
        for x in stride(from: -10.0 as Float, through: 10.0, by: 4.0) {
            addBox(at: [x, cabH/2, cabZ], w: cabW, h: cabH, d: cabD, color: cabinetColor)
        }

        // ── Desk pods: 2 rows of 4 desks each, wide aisles ────────────────
        let deskH: Float = 0.75
        let deskW: Float = 1.5
        let deskD: Float = 0.70
        let chairH: Float = 0.95
        let chairS: Float = 0.50

        let deskRowZs: [Float] = [-3.5, 3.5]      // 7m apart, big central aisle
        let deskColX: [Float]  = [-9, -3, 3, 9]   // 4 desks per row, ~6m spacing

        for rowZ in deskRowZs {
            for x in deskColX {
                addBox(at: [x, deskH/2, rowZ], w: deskW, h: deskH, d: deskD, color: deskColor)
            }
        }

        // Chairs in front of each desk (skip nothing — spawn is far from these).
        for rowZ in deskRowZs {
            let chairZ = rowZ + deskD/2 + 0.30 + chairS/2   // 0.40m forward of desk
            for x in deskColX {
                addBox(at: [x, chairH/2, chairZ], w: chairS, h: chairH, d: chairS, color: chairColor)
            }
        }

        // ── Clutter bins / boxes — scattered in big open areas ────────────
        let binPositions: [(SIMD3<Float>, Float, Float, Float)] = [
            ([-13.0, 0.20,  -8.0], 0.40, 0.40, 0.40),
            ([ 13.0, 0.20,  -8.0], 0.40, 0.40, 0.40),
            ([-13.0, 0.20,   8.0], 0.40, 0.40, 0.40),
            ([ 13.0, 0.20,   8.0], 0.40, 0.40, 0.40),
            ([ -6.0, 0.20,   0.0], 0.45, 0.45, 0.45),
            ([  6.0, 0.20,   0.0], 0.45, 0.45, 0.45),
            ([  0.0, 0.20,  -7.5], 0.50, 0.50, 0.50),
            ([ -8.0, 0.20,   7.0], 0.40, 0.40, 0.40),
            ([  8.0, 0.20,   7.0], 0.40, 0.40, 0.40),
            ([  0.0, 0.20,  -1.5], 0.40, 0.40, 0.40),
        ]
        for (i, p) in binPositions.enumerated() {
            addBox(at: p.0, w: p.1, h: p.2, d: p.3, color: binColors[i % binColors.count])
        }

        // ── Extra furniture / structural obstacles ────────────────────────
        let sofaColor       = NSColor(red: 0.30, green: 0.32, blue: 0.40, alpha: 1)  // dark blue-grey
        let pillarColor     = NSColor(white: 0.65, alpha: 1)                          // light grey
        let plantColor      = NSColor(red: 0.18, green: 0.45, blue: 0.22, alpha: 1)  // dark green
        let cardboardColor  = NSColor(red: 0.62, green: 0.45, blue: 0.28, alpha: 1)  // cardboard brown
        let printerColor    = NSColor(white: 0.30, alpha: 1)                          // dark grey
        let coolerColor     = NSColor(red: 0.55, green: 0.75, blue: 0.85, alpha: 1)  // light blue
        let whiteboardColor = NSColor(white: 0.95, alpha: 1)                          // white

        // Long couches / benches against side walls (give the room more "stuff")
        addBox(at: [-14.5, 0.40,  6.0], w: 0.6, h: 0.8, d: 2.5, color: sofaColor)   // west wall, south
        addBox(at: [ 14.5, 0.40, -6.0], w: 0.6, h: 0.8, d: 2.5, color: sofaColor)   // east wall, north

        // Interior structural pillars — flank the central aisle
        addBox(at: [-7.0, 1.50,  0.0], w: 0.4, h: 3.0, d: 0.4, color: pillarColor)
        addBox(at: [ 7.0, 1.50,  0.0], w: 0.4, h: 3.0, d: 0.4, color: pillarColor)

        // Plants near the doorway (flanking the spawn corridor)
        addBox(at: [-2.0, 0.50,  6.0], w: 0.4, h: 1.0, d: 0.4, color: plantColor)
        addBox(at: [ 2.0, 0.50,  6.0], w: 0.4, h: 1.0, d: 0.4, color: plantColor)
        // Plant in the north corridor between desks and cabinets
        addBox(at: [ 0.0, 0.50, -5.5], w: 0.4, h: 1.0, d: 0.4, color: plantColor)

        // Cardboard box stacks scattered in open zones
        addBox(at: [-10.0, 0.50, -7.0], w: 0.5, h: 1.0, d: 0.5, color: cardboardColor)
        addBox(at: [ 10.0, 0.50, -7.0], w: 0.5, h: 1.0, d: 0.5, color: cardboardColor)
        addBox(at: [-11.0, 0.30,  3.5], w: 0.6, h: 0.6, d: 0.6, color: cardboardColor)
        addBox(at: [ 11.0, 0.30,  3.5], w: 0.6, h: 0.6, d: 0.6, color: cardboardColor)

        // Printer / copier (east side of north area)
        addBox(at: [13.0, 0.40, -3.0], w: 0.6, h: 0.8, d: 0.5, color: printerColor)

        // Water cooler (west side)
        addBox(at: [-13.0, 0.55, 4.0], w: 0.4, h: 1.1, d: 0.4, color: coolerColor)

        // Wall-mounted whiteboard — thin slab against west wall, north area
        addBox(at: [-14.85, 1.25, -3.0], w: 0.10, h: 1.5, d: 2.0, color: whiteboardColor)

        // Filing cabinets — extra storage along east wall
        addBox(at: [14.5, 0.60, -2.0], w: 0.5, h: 1.2, d: 0.5, color: cabinetColor)
        addBox(at: [14.5, 0.60,  2.0], w: 0.5, h: 1.2, d: 0.5, color: cabinetColor)

        // Tall plants in the corners (visual richness, fills depth scene)
        addBox(at: [-13.5, 0.75, -5.0], w: 0.5, h: 1.5, d: 0.5, color: plantColor)
        addBox(at: [ 13.5, 0.75,  6.0], w: 0.5, h: 1.5, d: 0.5, color: plantColor)
    }

    /// Add a box-shaped obstacle. UnlitMaterial = no specular, no reflection,
    /// flat-shaded — gives the depth model clean inputs and avoids highlights.
    private func addBox(at pos: SIMD3<Float>, w: Float, h: Float, d: Float, color: NSColor) {
        let mesh = MeshResource.generateBox(width: w, height: h, depth: d)
        let mat  = UnlitMaterial(color: color)
        let ent  = ModelEntity(mesh: mesh, materials: [mat])
        ent.position = pos
        root.addChild(ent)
        obstacles.append(Obstacle(pos: pos, halfX: w / 2, halfZ: d / 2, height: h))
    }

    private func addCar() {
        // Tiny invisible proxy — FPV camera sits above, car body never visually rendered.
        let carMesh = MeshResource.generateSphere(radius: 0.001)
        car = ModelEntity(mesh: carMesh, materials: [UnlitMaterial(color: .clear)])
        // Spawn just inside south doorway, facing -Z (into the office).
        car.position    = [0, carBodyY, 9.0]
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
        cam.position = [0, carBodyY + eyeHeight, 9.0]
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
