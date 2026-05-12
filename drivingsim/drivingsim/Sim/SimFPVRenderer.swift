//
//  SimFPVRenderer.swift
//  drivingsim
//
//  Offscreen RealityKit renderer that produces an FPV-camera CVPixelBuffer
//  matching the on-screen sim, for feeding into the depth model.
//
//  Maintains a parallel scene that mirrors SimScene's obstacles + car position.
//  Renders to a Metal texture backed by a CVPixelBuffer (zero-copy via
//  CVMetalTextureCache).
//

@preconcurrency import CoreVideo
import AppKit
import Foundation
import Metal
import RealityKit
import simd

@MainActor
final class SimFPVRenderer {
    private let renderer: RealityRenderer
    private let device: MTLDevice
    private let textureCache: CVMetalTextureCache
    private let pixelBufferPool: CVPixelBufferPool

    private let outputW: Int
    private let outputH: Int
    var outputSize: (width: Int, height: Int) { (outputW, outputH) }
    private let fpvCamera: PerspectiveCamera
    private let carProxy: Entity   // invisible — just for camera attachment math

    /// Latest rendered frame. Replaced atomically each render tick.
    private(set) var latestPixelBuffer: CVPixelBuffer?

    init?(obstacles: [(pos: SIMD3<Float>, halfX: Float, halfZ: Float, height: Float, color: NSColor)],
          personBillboards: [(pos: SIMD3<Float>, width: Float, height: Float, textureName: String)] = [],
          eyeHeight: Float,
          width: Int = 518,
          height: Int = 518) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("[SimFPVRenderer] no Metal device")
            return nil
        }
        self.device = device
        self.outputW = width
        self.outputH = height

        // Texture cache for zero-copy MTLTexture↔CVPixelBuffer.
        var cache: CVMetalTextureCache?
        let cacheStatus = CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        guard cacheStatus == kCVReturnSuccess, let cache else {
            print("[SimFPVRenderer] texture cache create failed: \(cacheStatus)")
            return nil
        }
        self.textureCache = cache

        // Pixel buffer pool — BGRA, IOSurface-backed for Metal compat.
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey:           width,
            kCVPixelBufferHeightKey:          height,
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
        ]
        var pool: CVPixelBufferPool?
        let poolStatus = CVPixelBufferPoolCreate(nil, nil, attrs as CFDictionary, &pool)
        guard poolStatus == kCVReturnSuccess, let pool else {
            print("[SimFPVRenderer] pool create failed: \(poolStatus)")
            return nil
        }
        self.pixelBufferPool = pool

        // RealityRenderer.
        do {
            self.renderer = try RealityRenderer()
        } catch {
            print("[SimFPVRenderer] RealityRenderer init failed: \(error)")
            return nil
        }

        // Build the parallel scene: ground, obstacles, car proxy, camera, lights.
        // Materials + geometry must mirror SimScene so the depth model sees what
        // the user sees. Matte diffuse (roughness=1.0) keeps shading + cast
        // shadows as geometric cues.
        //
        // Outdoor ground (visible through doorway). Bright off-white.
        let groundMesh = MeshResource.generateBox(width: 200, height: 0.02, depth: 200)
        let groundMat  = SimpleMaterial(color: NSColor(white: 0.92, alpha: 1),
                                         roughness: 1.0, isMetallic: false)
        let ground     = ModelEntity(mesh: groundMesh, materials: [groundMat])
        ground.position = [0, -0.011, 0]
        renderer.entities.append(ground)

        // Office floor — sits 1.4cm above outdoor ground so the room reads
        // as a brighter tile when viewed through the doorway.
        let floorMesh = MeshResource.generateBox(width: 30.0, height: 0.005, depth: 20.0)
        let floorMat  = SimpleMaterial(color: NSColor(white: 0.95, alpha: 1),
                                        roughness: 1.0, isMetallic: false)
        let floor     = ModelEntity(mesh: floorMesh, materials: [floorMat])
        floor.position = [0, 0.003, 0]
        renderer.entities.append(floor)

        for o in obstacles {
            let mesh = MeshResource.generateBox(width: o.halfX * 2,
                                                height: o.height,
                                                depth: o.halfZ * 2)
            let mat  = SimpleMaterial(color: o.color, roughness: 1.0, isMetallic: false)
            let ent  = ModelEntity(mesh: mesh, materials: [mat])
            ent.position = o.pos
            renderer.entities.append(ent)
        }

        // Textured person billboards — must look like a person to YOLO.
        for p in personBillboards {
            let mesh = MeshResource.generatePlane(width: p.width, height: p.height)
            var mat = UnlitMaterial(color: .white)
            if let tex = try? TextureResource.load(named: p.textureName) {
                mat.color = .init(tint: .white, texture: .init(tex))
            } else {
                print("[SimFPVRenderer] texture \(p.textureName) not found")
                mat = UnlitMaterial(color: NSColor.systemPink)
            }
            let front = ModelEntity(mesh: mesh, materials: [mat])
            front.position = p.pos
            front.orientation = simd_quatf(angle: .pi, axis: [0, 1, 0])
            renderer.entities.append(front)
            let back = ModelEntity(mesh: mesh, materials: [mat])
            back.position = p.pos
            back.orientation = simd_quatf(angle: 0, axis: [0, 1, 0])
            renderer.entities.append(back)
        }

        // Car proxy — anchor for camera; not rendered (clear material, tiny).
        let proxyMesh = MeshResource.generateSphere(radius: 0.001)
        carProxy = ModelEntity(mesh: proxyMesh,
                               materials: [SimpleMaterial(color: .clear, isMetallic: false)])
        carProxy.position = [0, 0.5, 0]
        renderer.entities.append(carProxy)

        // FPV camera.
        let cam = PerspectiveCamera()
        // Must match SimScene main camera (75°) AND DepthProjector.hfovDeg.
        // For the real ESP32-CAM later, drop to 65° AND update DepthProjector.hfovDeg.
        cam.camera.fieldOfViewInDegrees = 75
        // Tight near plane prevents wall culling on hard collisions.
        cam.camera.near = 0.01
        cam.camera.far  = 1000
        cam.position = [0, 0.5 + eyeHeight, 0]
        cam.look(at: [0, 0.5 + eyeHeight, -10], from: cam.position, relativeTo: nil)
        renderer.entities.append(cam)
        renderer.activeCamera = cam
        self.fpvCamera = cam

        // Lights — shared with SimScene via SimSceneLighting helper.
        // RealityRenderer has no default IBL skybox, so the helper does all
        // the heavy lifting (point lights on the ceiling + one shadow key).
        SimSceneLighting.attach(toRenderer: renderer, addShadowKey: true)

        print("[SimFPVRenderer] initialised \(width)x\(height) with \(obstacles.count) obstacles + \(personBillboards.count) person billboards")
    }

    /// Render one frame. Returns the produced CVPixelBuffer or nil on failure.
    func render(carPosition: SIMD3<Float>, yaw: Float, eyeHeight: Float) -> CVPixelBuffer? {
        // Update camera to match operator's FPV view.
        let fwdX =  sin(yaw)
        let fwdZ = -cos(yaw)
        let eyePos = carPosition + SIMD3<Float>(0, eyeHeight, 0)
        let lookAt = eyePos + SIMD3<Float>(fwdX, 0, fwdZ) * 10
        fpvCamera.position = eyePos
        fpvCamera.look(at: lookAt, from: eyePos, relativeTo: nil)

        // Allocate a pixel buffer from the pool.
        var pb: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(nil, pixelBufferPool, &pb)
        guard status == kCVReturnSuccess, let pixelBuffer = pb else {
            print("[SimFPVRenderer] pool exhausted: \(status)")
            return nil
        }

        // Wrap the pixel buffer as a Metal texture (zero-copy via cache).
        var cvMtl: CVMetalTexture?
        let texStatus = CVMetalTextureCacheCreateTextureFromImage(
            nil, textureCache, pixelBuffer, nil,
            .bgra8Unorm, outputW, outputH, 0, &cvMtl)
        guard texStatus == kCVReturnSuccess,
              let cvMtl,
              let mtlTexture = CVMetalTextureGetTexture(cvMtl) else {
            print("[SimFPVRenderer] texture wrap failed: \(texStatus)")
            return nil
        }

        // Build camera output and render. Wait for completion via DispatchSemaphore so we can
        // hand the finished pixel buffer back synchronously.
        let cameraOutput: RealityRenderer.CameraOutput
        do {
            cameraOutput = try RealityRenderer.CameraOutput(.singleProjection(colorTexture: mtlTexture))
        } catch {
            print("[SimFPVRenderer] cameraOutput init failed: \(error)")
            return nil
        }

        let sem = DispatchSemaphore(value: 0)
        do {
            try renderer.updateAndRender(
                deltaTime: 1.0 / 30.0,
                cameraOutput: cameraOutput,
                onComplete: { _ in sem.signal() }
            )
        } catch {
            print("[SimFPVRenderer] render failed: \(error)")
            return nil
        }
        // Wait briefly for GPU completion so the pixel buffer reflects the final render.
        _ = sem.wait(timeout: .now() + .milliseconds(50))

        latestPixelBuffer = pixelBuffer
        return pixelBuffer
    }
}
