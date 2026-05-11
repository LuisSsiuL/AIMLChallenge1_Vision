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

    // Car tunables — autonomous indoor RC robot speeds (TurtleBot-class).
    // ~0.6 m/s ≈ 2 km/h, brisk walking pace ÷ 3. Realistic for vision-based
    // autonomous navigation where reaction time + braking distance matter.
    private let maxSpeed:      Float = 0.6
    private let accel:         Float = 0.4
    private let brakeDecel:    Float = 0.8
    private let coastDrag:     Float = 0.4
    private let maxSteerRate:  Float = 1.2
    private let steerSpeedRef: Float = 0.3   // full steer at low speed

    // FPV camera — RC car eye-line ~6cm above floor.
    private let eyeHeight:     Float = 0.06
    // Camera mounted near front of car but kept INSIDE the AABB so collision
    // protects it from clipping into obstacle geometry (otherwise the obstacle
    // appears see-through when cam slips past its front face).
    private let eyeForward:    Float = 0.085   // 1.5 cm behind front bumper

    // Car AABB (15cm wide, 20cm long).
    private let carHalfX: Float = 0.075
    private let carHalfZ: Float = 0.10
    private let carBodyY: Float = 0.03   // centre of car body above floor
    // Skin epsilon — gap kept between AABB and obstacle after push-out so
    // camera (inside AABB) never reaches obstacle surface.
    private let collisionSkin: Float = 0.01

    // State
    private var speed: Float = 0
    private var yaw:   Float = 0

    // Obstacles — single source of truth for render + collision.
    private struct Obstacle { let pos: SIMD3<Float>; let halfX: Float; let halfZ: Float; let height: Float; var isPerson: Bool = false }
    private var obstacles: [Obstacle] = []

    // Person billboard registry (mirrored separately in FPV renderer with texture).
    private struct PersonBillboard { let pos: SIMD3<Float>; let width: Float; let height: Float; let textureName: String }
    private var personBillboards: [PersonBillboard] = []

    /// Public read-only snapshot for SimFPVRenderer mirror scene.
    var obstacleSnapshot: [(pos: SIMD3<Float>, halfX: Float, halfZ: Float, height: Float)] {
        obstacles.filter { !$0.isPerson }.map { ($0.pos, $0.halfX, $0.halfZ, $0.height) }
    }
    var personBillboardSnapshot: [(pos: SIMD3<Float>, width: Float, height: Float, textureName: String)] {
        personBillboards.map { ($0.pos, $0.width, $0.height, $0.textureName) }
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
        // Outdoor ground (visible through doorway). Matte diffuse — receives shadows.
        let groundSize: Float = 200
        let groundMesh = MeshResource.generateBox(width: groundSize, height: 0.02, depth: groundSize)
        // Bright ground — high-contrast environment so dark objects pop.
        let groundMat  = SimpleMaterial(color: NSColor(white: 0.92, alpha: 1),
                                         roughness: 1.0, isMetallic: false)
        let ground     = ModelEntity(mesh: groundMesh, materials: [groundMat])
        ground.position = [0, -0.011, 0]
        root.addChild(ground)

        // Office floor — bright off-white, receives furniture shadows.
        let floorMat = SimpleMaterial(color: NSColor(white: 0.95, alpha: 1),
                                       roughness: 1.0, isMetallic: false)
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

        // High-contrast scheme: bright walls, dark obstacles.
        let wallColor    = NSColor(white: 0.95, alpha: 1)   // near-white background
        let cabinetColor = NSColor(white: 0.12, alpha: 1)   // dark
        let deskColor    = NSColor(white: 0.10, alpha: 1)   // dark
        let chairColor   = NSColor(white: 0.08, alpha: 1)   // very dark
        let binColors: [NSColor] = [
            NSColor(white: 0.10, alpha: 1),
            NSColor(white: 0.15, alpha: 1),
            NSColor(white: 0.12, alpha: 1),
            NSColor(white: 0.08, alpha: 1),
            NSColor(white: 0.18, alpha: 1),
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
        // All physical obstacles → dark. Whiteboard is a wall fixture, stays bright.
        let sofaColor       = NSColor(white: 0.12, alpha: 1)
        let pillarColor     = NSColor(white: 0.10, alpha: 1)
        let plantColor      = NSColor(white: 0.14, alpha: 1)
        let cardboardColor  = NSColor(white: 0.18, alpha: 1)
        let printerColor    = NSColor(white: 0.08, alpha: 1)
        let coolerColor     = NSColor(white: 0.16, alpha: 1)
        let whiteboardColor = NSColor(white: 0.97, alpha: 1)   // wall fixture, bright

        // Long couches / benches against side walls (give the room more "stuff")
        addBox(at: [-14.5, 0.40,  6.0], w: 0.6, h: 0.8, d: 2.5, color: sofaColor)   // west wall, south
        addBox(at: [ 14.5, 0.40, -6.0], w: 0.6, h: 0.8, d: 2.5, color: sofaColor)   // east wall, north

        // Interior structural pillars — flank the central aisle
        addBox(at: [-7.0, 1.50,  0.0], w: 0.4, h: 3.0, d: 0.4, color: pillarColor)
        addBox(at: [ 7.0, 1.50,  0.0], w: 0.4, h: 3.0, d: 0.4, color: pillarColor)

        // Plants near the doorway (flanking the spawn corridor)
        addBox(at: [-2.0, 0.50,  6.0], w: 0.4, h: 1.0, d: 0.4, color: plantColor)
        addBox(at: [ 2.0, 0.50,  6.0], w: 0.4, h: 1.0, d: 0.4, color: plantColor)
        // Person target (replaces north-corridor plant) — vertical billboard
        // facing the south doorway (-Z direction = away from spawn at +Z).
        addPersonBillboard(at: [0.0, 0.85, -5.5])

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

    /// Add a box-shaped obstacle. Matte diffuse (roughness=1.0) — no specular
    /// highlights, no environment reflection, but DOES respond to lights so
    /// the depth model sees proper shading + cast shadows.
    private func addBox(at pos: SIMD3<Float>, w: Float, h: Float, d: Float, color: NSColor) {
        let mesh = MeshResource.generateBox(width: w, height: h, depth: d)
        let mat  = SimpleMaterial(color: color, roughness: 1.0, isMetallic: false)
        let ent  = ModelEntity(mesh: mesh, materials: [mat])
        ent.position = pos
        root.addChild(ent)
        obstacles.append(Obstacle(pos: pos, halfX: w / 2, halfZ: d / 2, height: h))
    }

    /// Add a vertical person billboard at `pos`. Uses an UnlitMaterial so the
    /// texture renders without shading darken (keeps YOLO detection stable).
    /// Registers a thin collision proxy so the car stops just in front.
    private func addPersonBillboard(at pos: SIMD3<Float>) {
        // Source image is 960×1280 (3:4 portrait). Match plane aspect to avoid
        // stretching — keeps detection looking like a real person.
        let planeH: Float = 1.7
        let planeW: Float = planeH * (960.0 / 1280.0)   // ≈ 1.275m
        let mesh = MeshResource.generatePlane(width: planeW, height: planeH)
        var mat = UnlitMaterial(color: .white)
        if let tex = try? TextureResource.load(named: "person_target") {
            mat.color = .init(tint: .white, texture: .init(tex))
        } else {
            print("[SimScene] person_target texture not found in asset catalog — using fallback color")
            mat = UnlitMaterial(color: NSColor.systemPink)
        }
        let ent = ModelEntity(mesh: mesh, materials: [mat])
        ent.position = pos
        // generatePlane lies in the XY plane (normal = +Z). Rotate 180° around
        // Y so the textured face points toward +Z (the south doorway / spawn).
        ent.orientation = simd_quatf(angle: .pi, axis: [0, 1, 0])
        root.addChild(ent)
        // Back-side clone facing -Z so the plane is visible from either approach.
        let back = ent.clone(recursive: false)
        back.orientation = simd_quatf(angle: 0, axis: [0, 1, 0])
        root.addChild(back)
        // Thin collision proxy — stops car ~0.08m before texture plane.
        obstacles.append(Obstacle(pos: pos,
                                  halfX: planeW / 2,
                                  halfZ: 0.08,
                                  height: planeH,
                                  isPerson: true))
        // Register for FPV mirror scene so renderer can build textured plane too.
        personBillboards.append(PersonBillboard(pos: pos,
                                                width: planeW,
                                                height: planeH,
                                                textureName: "person_target"))
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
        // Key light — enable shadow casting so furniture casts onto floor/walls.
        // Depth model gets stronger geometric cues from cast shadows.
        let key = DirectionalLight()
        key.light.intensity = 6000
        key.orientation = simd_quatf(angle: -.pi / 4, axis: [1, 0, 0])
        key.shadow = DirectionalLightComponent.Shadow()
        root.addChild(key)

        // Fill light — no shadows (avoids double-shadow artefacts, fills ambient).
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
                      yolo: YOLODriver,
                      yoloPy: YOLOPythonDriver,
                      modeProvider: @escaping @MainActor () -> DrivingMode) {
        guard timer == nil else { return }
        let t = Timer(timeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in
                let m = modeProvider()
                self.tick(kb: keyboard, hand: hand, depth: depth,
                          yolo: yolo, yoloPy: yoloPy,
                          mode: m, dt: 1.0 / 60.0)
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
                      yolo: YOLODriver,
                      yoloPy: YOLOPythonDriver,
                      mode: DrivingMode,
                      dt: Float) {
        // Input gating per mode:
        //   .off         — kb only
        //   .hand        — kb + hand
        //   .assisted    — depth → W/S; (kb||hand) → A/D
        //   .automated   — depth → all four; kb/hand ignored
        //   .autoSeek    — CoreML YOLO ROAM↔SEEK state machine
        //   .autoSeekPy  — Python ultralytics ROAM↔SEEK (same logic)
        let kbActive   = (mode != .automated && mode != .autoSeek && mode != .autoSeekPy)
        let handActive = (mode == .hand || mode == .assisted)
        let depthWS    = (mode == .assisted || mode == .automated)
        let depthAD    = (mode == .automated)

        // Pick active YOLO source for this mode.
        let activeYoloSeekState: SeekState
        let activeBest: PersonDetection?
        switch mode {
        case .autoSeek:
            activeYoloSeekState = yolo.seekState
            activeBest = yolo.bestDetection
        case .autoSeekPy:
            activeYoloSeekState = yoloPy.seekState
            activeBest = yoloPy.bestDetection
        default:
            activeYoloSeekState = .roam
            activeBest = nil
        }
        let inRoam = mode.anyYOLO && activeYoloSeekState == .roam
        let inSeek = mode.anyYOLO && activeYoloSeekState == .seek

        // YOLO seek steering law.
        var yoloFwd = false, yoloLeft = false, yoloRight = false
        if inSeek, let d = activeBest {
            if d.cx < 0.45 { yoloLeft = true }
            else if d.cx > 0.55 { yoloRight = true }
            if d.h < 0.55 { yoloFwd = true }
            if yoloFwd && (depth.command == .reverse
                           || depth.command == .reverseLeft
                           || depth.command == .reverseRight
                           || depth.command == .brake) {
                yoloFwd = false
            }
        }

        let inForward  = (kbActive && kb.forward)  || (handActive && hand.forward)
                         || (depthWS && depth.forward)
                         || (inRoam && depth.forward)
                         || yoloFwd
        let inBackward = (kbActive && kb.backward) || (handActive && hand.backward)
                         || (depthWS && depth.backward)
                         || (inRoam && depth.backward)
        let inLeft     = (kbActive && kb.left)     || (handActive && hand.left)
                         || (depthAD && depth.left)
                         || (inRoam && depth.left)
                         || yoloLeft
        let inRight    = (kbActive && kb.right)    || (handActive && hand.right)
                         || (depthAD && depth.right)
                         || (inRoam && depth.right)
                         || yoloRight

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

        // ── Move car (per-axis sweep so we slide along walls) ──
        let delta = fwd * speed * dt
        var nextPos = car.position
        nextPos.y = carBodyY

        var blockedX = false
        var blockedZ = false

        // X axis
        nextPos.x += delta.x
        for o in obstacles {
            let dx = nextPos.x - o.pos.x
            let dz = nextPos.z - o.pos.z
            let overlapX = (carHalfX + o.halfX) - abs(dx)
            let overlapZ = (carHalfZ + o.halfZ) - abs(dz)
            if overlapX > 0 && overlapZ > 0 {
                nextPos.x += (overlapX + collisionSkin) * (dx >= 0 ? 1 : -1)
                blockedX = true
            }
        }

        // Z axis
        nextPos.z += delta.z
        for o in obstacles {
            let dx = nextPos.x - o.pos.x
            let dz = nextPos.z - o.pos.z
            let overlapX = (carHalfX + o.halfX) - abs(dx)
            let overlapZ = (carHalfZ + o.halfZ) - abs(dz)
            if overlapX > 0 && overlapZ > 0 {
                nextPos.z += (overlapZ + collisionSkin) * (dz >= 0 ? 1 : -1)
                blockedZ = true
            }
        }

        // Damp speed only if forward motion was actually blocked.
        // Use travelled-vs-intended ratio so partial blocks (slide along wall) don't kill speed.
        let intended = simd_length(delta)
        let actual   = simd_length(nextPos - car.position - SIMD3<Float>(0, nextPos.y - car.position.y, 0))
        if intended > 1e-5 {
            let ratio = actual / intended
            if ratio < 0.4 { speed = 0 }                 // head-on hit: stop
            else if ratio < 0.85 { speed *= 0.5 }        // grazing: damp, don't stop
        } else if blockedX || blockedZ {
            speed = 0
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
