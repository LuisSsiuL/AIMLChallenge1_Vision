//
//  SimScene.swift
//  drivingsim
//
//  FPV driving sim. Car translates + rotates in world space each frame;
//  camera is a child of car entity — inherits yaw automatically, no per-tick math.
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

    // Car tunables
    private let maxSpeed:      Float = 7
    private let accel:         Float = 4
    private let brakeDecel:    Float = 8
    private let coastDrag:     Float = 3
    private let maxSteerRate:  Float = 1.6
    private let steerSpeedRef: Float = 2

    // FPV camera tunables
    private let eyeHeight:     Float = 1.2   // camera height above car center
    private let eyeForward:    Float = 0.0   // forward offset along car -Z

    // State
    private var speed: Float = 0
    private var yaw:   Float = 0

    // Procedural obstacles — single source of truth for render + collision
    private struct Obstacle { let pos: SIMD3<Float>; let halfX: Float; let halfZ: Float }
    private var obstacles: [Obstacle] = []

    private var timer: Timer?

    static func make() -> SimScene {
        let s = SimScene()
        s.setup()
        return s
    }

    private func setup() {
        // Ground
        let groundSize: Float = 500
        let groundMesh  = MeshResource.generateBox(width: groundSize, height: 0.1, depth: groundSize)
        let groundMat   = UnlitMaterial(color: NSColor(red: 0.25, green: 0.38, blue: 0.25, alpha: 1))
        let ground      = ModelEntity(mesh: groundMesh, materials: [groundMat])
        ground.position = [0, -0.05, 0]
        root.addChild(ground)

        // Grid lines every 10 m, ±100 m around origin
        let gridMat    = UnlitMaterial(color: NSColor(white: 0.9, alpha: 1))
        let gridExtent: Float = 100
        let gridStep:   Float = 10
        let lineThick:  Float = 0.10
        let lineLen     = gridExtent * 2
        let lineMeshX  = MeshResource.generateBox(width: lineLen,   height: 0.02, depth: lineThick)
        let lineMeshZ  = MeshResource.generateBox(width: lineThick, height: 0.02, depth: lineLen)
        var z = -gridExtent
        while z <= gridExtent {
            let l = ModelEntity(mesh: lineMeshX, materials: [gridMat])
            l.position = [0, 0.012, z]; root.addChild(l)
            z += gridStep
        }
        var x = -gridExtent
        while x <= gridExtent {
            let l = ModelEntity(mesh: lineMeshZ, materials: [gridMat])
            l.position = [x, 0.012, 0]; root.addChild(l)
            x += gridStep
        }

        // Procedural rubble obstacles — 120 mixed-size boxes
        let colors: [NSColor] = [.systemBlue, .systemYellow, .systemOrange, .systemPurple, .systemTeal, .systemPink,
                                  .systemBrown, .systemGray, .systemRed, .systemGreen]
        let obstacleCount = 120
        let spawnRadius:  Float = 80.0
        let safeZone:     Float = 5.0    // clear area around car spawn

        var rng = SystemRandomNumberGenerator()
        func frand(_ lo: Float, _ hi: Float) -> Float {
            Float.random(in: lo...hi, using: &rng)
        }

        // Obstacle archetype roll: small rubble, medium boxes, big boulders, walls
        // Weighted: 45% small, 30% medium, 15% big, 10% wall
        for i in 0..<obstacleCount {
            var ox: Float
            var oz: Float
            repeat {
                ox = frand(-spawnRadius, spawnRadius)
                oz = frand(-spawnRadius, spawnRadius)
            } while abs(ox) < safeZone && abs(oz) < safeZone

            let roll = frand(0, 1)
            var w: Float, d: Float, h: Float

            if roll < 0.45 {
                // small rubble
                w = frand(0.4, 1.5)
                d = frand(0.4, 1.5)
                h = frand(0.5, 2.0)
            } else if roll < 0.75 {
                // medium box
                w = frand(1.5, 3.0)
                d = frand(1.5, 3.0)
                h = frand(2.0, 5.0)
            } else if roll < 0.90 {
                // big boulder / debris pile
                w = frand(3.0, 6.0)
                d = frand(3.0, 6.0)
                h = frand(4.0, 9.0)
            } else {
                // wall — long on one axis
                let longAxisX = Bool.random(using: &rng)
                let longLen   = frand(8.0, 18.0)
                let shortLen  = frand(0.4, 1.2)
                w = longAxisX ? longLen  : shortLen
                d = longAxisX ? shortLen : longLen
                h = frand(2.5, 5.5)
            }

            // random yaw rotation for variety (walls and big boulders look better rotated)
            let rot = frand(-Float.pi, Float.pi)

            let pos = SIMD3<Float>(ox, h / 2, oz)
            let mesh = MeshResource.generateBox(width: w, height: h, depth: d)
            let mat  = SimpleMaterial(color: colors[i % colors.count], isMetallic: false)
            let ent  = ModelEntity(mesh: mesh, materials: [mat])
            ent.position    = pos
            ent.orientation = simd_quatf(angle: rot, axis: [0, 1, 0])
            root.addChild(ent)

            // AABB collision uses rotated bounding circle approximation:
            // expand half-extents to enclosing radius so collision still works after yaw
            let r = sqrt((w / 2) * (w / 2) + (d / 2) * (d / 2))
            obstacles.append(Obstacle(pos: pos, halfX: r, halfZ: r))
        }

        // Car — tiny invisible mesh (FPV: not rendered visibly)
        let carMesh = MeshResource.generateSphere(radius: 0.001)
        car = ModelEntity(mesh: carMesh, materials: [SimpleMaterial(color: .clear, isMetallic: false)])
        car.position    = [0, 0.5, 0]
        car.orientation = simd_quatf(angle: 0, axis: [0, 1, 0])
        root.addChild(car)

        // Lights
        let key = DirectionalLight()
        key.light.intensity = 8000
        key.orientation = simd_quatf(angle: -.pi / 4, axis: [1, 0, 0])
        root.addChild(key)
        let fill = DirectionalLight()
        fill.light.intensity = 2500
        fill.orientation = simd_quatf(angle: .pi / 5, axis: [-1, 0, 0])
        root.addChild(fill)

        // FPV camera — at car position, oriented per car yaw via per-tick update
        cam = PerspectiveCamera()
        cam.camera.fieldOfViewInDegrees = 75
        cam.position = [0, 0.5 + eyeHeight, 0]
        cam.look(at: [0, 0.5 + eyeHeight, -10], from: cam.position, relativeTo: nil)
        root.addChild(cam)
    }

    func startUpdates(keyboard: KeyboardMonitor) {
        guard timer == nil else { return }
        let t = Timer(timeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in self.tick(kb: keyboard, dt: 1.0 / 60.0) }
        }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    func stopUpdates() {
        timer?.invalidate()
        timer = nil
    }

    private func tick(kb: KeyboardMonitor, dt: Float) {
        // ── Throttle / brake ──
        if kb.forward {
            speed = min(speed + accel * dt, maxSpeed)
        } else if kb.backward {
            speed = max(speed - brakeDecel * dt, -maxSpeed * 0.5)
        } else {
            let drag = coastDrag * dt
            speed = abs(speed) <= drag ? 0 : speed - drag * (speed > 0 ? 1 : -1)
        }

        // ── Steering ──
        let speedFactor = min(1, abs(speed) / steerSpeedRef)
        let steerInput: Float = (kb.right ? 1 : 0) + (kb.left ? -1 : 0)
        let yawRate     = steerInput * maxSteerRate * speedFactor * (speed >= 0 ? 1 : -1)
        yaw += yawRate * dt

        // Forward vector (car faces -Z at yaw=0)
        let fwdX =  sin(yaw)
        let fwdZ = -cos(yaw)
        let fwd  = SIMD3<Float>(fwdX, 0, fwdZ)

        // ── Move car ──
        var nextPos = car.position + fwd * speed * dt
        nextPos.y = 0.5

        // AABB pillar collision
        let carHalfX: Float = 0.5
        let carHalfZ: Float = 0.5
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

        // ── FPV camera — sit at car position + eye height, look along car forward ──
        let eyePos = car.position + SIMD3<Float>(fwdX * eyeForward, eyeHeight, fwdZ * eyeForward)
        let lookTarget = eyePos + fwd * 10
        cam.position = eyePos
        cam.look(at: lookTarget, from: eyePos, relativeTo: nil)
    }
}
