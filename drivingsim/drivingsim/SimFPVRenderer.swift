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
    private let fpvCamera: PerspectiveCamera
    private let carProxy: Entity   // invisible — just for camera attachment math

    /// Latest rendered frame. Replaced atomically each render tick.
    private(set) var latestPixelBuffer: CVPixelBuffer?

    init?(obstacles: [(pos: SIMD3<Float>, halfX: Float, halfZ: Float, height: Float)],
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
        let groundMat   = UnlitMaterial(color: NSColor(red: 0.25, green: 0.38, blue: 0.25, alpha: 1))
        let groundMesh  = MeshResource.generateBox(width: 500, height: 0.1, depth: 500)
        let ground      = ModelEntity(mesh: groundMesh, materials: [groundMat])
        ground.position = [0, -0.05, 0]
        renderer.entities.append(ground)

        for o in obstacles {
            // We don't know the exact w/d (only halfX/halfZ which encode collision AABB
            // post-rotation). Build a box matching the AABB — depth model only needs
            // approximate occupancy, not visual fidelity.
            let mesh = MeshResource.generateBox(width: o.halfX * 2,
                                                height: o.height,
                                                depth: o.halfZ * 2)
            let mat  = SimpleMaterial(color: .systemGray, isMetallic: false)
            let ent  = ModelEntity(mesh: mesh, materials: [mat])
            ent.position = o.pos
            renderer.entities.append(ent)
        }

        // Car proxy — anchor for camera; not rendered (clear material, tiny).
        let proxyMesh = MeshResource.generateSphere(radius: 0.001)
        carProxy = ModelEntity(mesh: proxyMesh,
                               materials: [SimpleMaterial(color: .clear, isMetallic: false)])
        carProxy.position = [0, 0.5, 0]
        renderer.entities.append(carProxy)

        // FPV camera.
        let cam = PerspectiveCamera()
        cam.camera.fieldOfViewInDegrees = 75
        cam.position = [0, 0.5 + eyeHeight, 0]
        cam.look(at: [0, 0.5 + eyeHeight, -10], from: cam.position, relativeTo: nil)
        renderer.entities.append(cam)
        renderer.activeCamera = cam
        self.fpvCamera = cam

        // Lights.
        let key = DirectionalLight()
        key.light.intensity = 8000
        key.orientation = simd_quatf(angle: -.pi / 4, axis: [1, 0, 0])
        renderer.entities.append(key)
        let fill = DirectionalLight()
        fill.light.intensity = 2500
        fill.orientation = simd_quatf(angle: .pi / 5, axis: [-1, 0, 0])
        renderer.entities.append(fill)

        print("[SimFPVRenderer] initialised \(width)x\(height) with \(obstacles.count) obstacles")
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
