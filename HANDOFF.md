# Drivingsim Handoff

Status snapshot for the autonomous-mapping/return-to-start RC car sim.

## What this session shipped

### 1. Metric depth pipeline (root-cause fix for bad mapping)

Old pipeline used `DepthAnythingV2Small` (relative inverse depth) → per-frame min/max normalize → percentile bucket into 4 fake range tiers (0.7/1.8/3.5/5m). Distances were fiction.

Now:
- **Model:** `DepthAnythingV2MetricIndoorSmallF16.mlpackage` (Hypersim indoor, 0–20m metric).
- **Converted via** `tools/convert_depth_metric.ipynb` (Jupyter, runs in `tools/venv_depth/`).
  - Patches `torch.nn.functional.interpolate` bicubic → bilinear (coremltools lacks bicubic).
  - Bakes DPT normalisation `(x − mean)/std` inside the traced model.
  - Outputs `ImageType GRAYSCALE_FLOAT16` so Swift reads it as `CVPixelBuffer` (matches existing `depthBufferToU8` path).
- **Swift conversion:** `DepthDriver.depthBufferToMetricU8(_:)` — fixed scale `U8 = metres × 12.75` (large U8 = far). Sets `DepthDriver.usingMetricDepth = true` on load.
- **DepthProjector rewritten:**
  - Real pinhole back-projection using both col AND row (3D info preserved).
  - 75° FOV matches `SimScene` main camera (was 65°, 10° mismatch).
  - 2.5D height-band filter `[0.02, 0.25]m world Y`. Floor → free ray. Above-car → skip.

### 2. Visual parity between FPV (model input) and player view

`SimFPVRenderer` now:
- Plumbs per-obstacle colors from `SimScene.obstacleSnapshot` (was uniform gray 0.55).
- FOV 65° → **75°** (matches SimScene + DepthProjector).
- Key light 8000 → 6000 (matches main scene).
- Wall colour: white → soft cool blue `(0.78, 0.85, 0.92)` in SimScene.

### 3. Tour mode (autonomous mission)

`MapDriver.TourState = {.exploring, .homing, .completed}`.

- Start cell captured on first frame + 30cm free disc seeded (`OccupancyGrid.markFreeDisc(at:radius:)`) so WFD/A* can boot before depth has carved space.
- `.exploring`: WFD frontier exploration.
- `updateSeekTarget()`: first YOLO sighting (gated by `maxDistFromStart ≥ 2m`) → `markedTarget = cell`, transition `.homing`.
- `.homing`: A* goal = startCell. Arrival after `framesInHoming ≥ 30` AND `dist < 0.5m` → `.completed`.
- `.completed`: `cmd = .brake` forever (idle).

Map visual:
- Frontiers (blue) **hidden** from map render AND from legend (user request).
- `markedTarget` shown as persistent orange dot.

### 4. WFD + heap A* + blacklist (ROS `explore_lite` port)

Replaced naive frontier code with the standard combo:

- **`OccupancyGrid.aStar`:** binary min-heap open set, octile heuristic with 0.001 tie-break, unknown cells passable with +0.4 cost penalty.
- **`FrontierExplorer.findClusters`:** WFD double-BFS (outer through free space, inner grows frontier). Filters clusters with `blacklist` set. Returns clusters sorted by `cost = 3.0·minDist − 1.0·size`.
- **`MapDriver.replan`:** iterates cost-sorted clusters, first reachable wins; unreachable clusters auto-blacklist their centroid.
- **Progress timer:** if pose advances < 20cm for 150 frames (~5s), blacklist current goal.
- **Path-invalidation check:** every frame, scan `currentPath` for occupied cells → force replan.

### 5. Truth pose feed (sim-only, eliminates map drift)

`PoseEstimator` was dead-reckoning commands at 30Hz while SimScene physics ran at 60Hz with collision damping → pose drifted through walls → obstacles stamped at wrong world coords.

Fix: `SimScene.tick` calls `MapDriver.updateTruthPose(_:yaw:)` every frame with the real car pos/yaw. `MapDriver.useTruthPose = true` (default) skips command-integration entirely. For real RC car: flip flag, swap to IMU/encoder odometry. Surface unchanged.

### 6. Steering responsiveness

`SimScene.swift:27-28`:
- `maxSteerRate 1.2 → 1.8`
- `steerSpeedRef 0.3 → 0.08` (full steer kicks in at very low speed, no more "drive forward to gain speed before you can turn")

### 7. Obstacle avoidance (just landed, untested)

`WaypointFollower`:
- Forward safety scan widened: 1..6 cells (10–60cm) instead of single 3-cell check.
- On blocked: check left/right shoulder (±45°, 4 cells). Pick clearer side. Both clear → bias by target heading. Both blocked → `reverseLeft`.
- Clear `path` on safety trigger to force replan.

`MapDriver`:
- `inflationRadius 1 → 2` (paths ≥20cm from walls, matches 15×20cm car body).

### 8. Stuck detection polarity fix

Old: `centralDisparityMean(depthU8) > 100 && std < 4` — with relative model's "large = far" convention, this fired on stable open space (inverted bug).
New: `centralDisparityMean(metricU8) < 20 && std < 3` — small metric U8 = close wall ahead. Now fires correctly.

### 9. YOLO sensitivity bump

`YOLOPythonDriver`:
- `confThreshold 0.35 → 0.15`
- `seekEnterConf 0.50 → 0.30`

Lowered to recover after FPV visual changes (matched colors + brighter scene reduced YOLO confidence vs the old high-contrast gray scene).

## Files touched

| File | Change |
|---|---|
| `tools/convert_depth_metric.ipynb` | New. Jupyter conversion notebook. CPU_ONLY ~3s, CPU_AND_NE 5-15 min. |
| `tools/convert_depth_anything_metric.py` | Legacy CLI version (notebook is canonical). |
| `tools/venv_depth/` | Python venv with torch + coremltools + transformers + ipykernel. |
| `drivingsim/drivingsim/DepthAnythingV2MetricIndoorSmallF16.mlpackage` | Metric model bundled via Xcode synchronized folder group. |
| `drivingsim/drivingsim/DepthDriver.swift` | Metric model load priority, `depthBufferToMetricU8`, `usingMetricDepth` flag. |
| `drivingsim/drivingsim/DepthProjector.swift` | Full rewrite — pinhole back-projection, 2.5D height filter, FOV 75°. |
| `drivingsim/drivingsim/MapDriver.swift` | Tour state machine, WFD integration, blacklist, progress timer, truth-pose hook, path-invalidation, stuck polarity fix. |
| `drivingsim/drivingsim/OccupancyGrid.swift` | Heap A*, octile heuristic, unknown-passable, `markFreeDisc`, `stateCounts`. |
| `drivingsim/drivingsim/FrontierExplorer.swift` | Rewritten: WFD double-BFS + explore_lite cost ranking + `closestCell` goal. |
| `drivingsim/drivingsim/WaypointFollower.swift` | Multi-cell forward scan + shoulder check + steer-around. |
| `drivingsim/drivingsim/SimScene.swift` | Wall colour change, obstacle.color field, steering constants, truth-pose push. |
| `drivingsim/drivingsim/SimFPVRenderer.swift` | Per-obstacle colours, FOV 75°, lighting match. |
| `drivingsim/drivingsim/ContentView.swift` | Frontier legend entry removed. |
| `drivingsim/drivingsim/YOLOPythonDriver.swift` | Lower conf thresholds. |
| `.claude/plans/can-you-check-the-mighty-pike.md` | Plan history (final = metric depth fix). |

## Build + run

1. **One-time:** Open `tools/convert_depth_metric.ipynb` in VS Code / Jupyter, kernel "Depth Conversion (venv)". Run cells 1-5. CPU_ONLY default gives ~3s conversion + ~80ms/frame inference. Switch Cell 4's `COMPUTE_UNITS = ct.ComputeUnit.CPU_AND_NE` and re-run cells 4-5 once you want ANE (~15ms/frame, 5-15 min compile).
2. **Xcode:** Open `drivingsim/drivingsim.xcodeproj`. The mlpackage is auto-included via the synchronized folder group — do NOT manually "Add Files" or you get "Multiple commands produce conflicting outputs". Product → Clean Build Folder → Build → Run.
3. **In app:** pick source (Sim or live cam). Switch mode to **Auto** (autoMap). Expect console:
   ```
   [MapDriver] metric model found
   [DepthDriver] metric model found: DepthAnythingV2MetricIndoorSmallF16
   [MapDriver] start cell locked @ (col,row) — seeded free disc r=3
   ```
4. Car explores via frontiers. When YOLO Python detects person (≥0.30 conf for 3 frames) AND car has roamed ≥2m, it locks `markedTarget` (orange dot on map) and paths back to start.

## Known issues / next up

- **CoreML conversion still on CPU_ONLY by default.** Inference ~80ms (~12fps). Move to ANE for ~15ms once stable. Notebook Cell 4 toggle.
- **No A* heap stress test on real maps.** Should be fine, but if planner stalls, suspect heap or `inflated()` perf (still allocates full grid copy every replan — could cache).
- **Person billboard small in frame at spawn.** Person is 14.5m from spawn. With lowered YOLO conf (0.15), detection should work but may be flaky. Could move person closer in `SimScene.addPersonBillboard` for testing.
- **Stuck-detection threshold (`stuckDispMeanMax = 20`)** = U8 ≈ 1.57m. Tune if too aggressive / too lazy.
- **`updateTruthPose` is sim-only.** Real RC car path: `useTruthPose = false`, wire IMU/encoder readings into `PoseEstimator.applyCorrection` or replace `tickFromCommand`.
- **`inflationRadius = 2`** (20cm clearance). If the room is tight, paths may dead-end; lower to 1.
- **Coverage sweep removed** (`DepthDriver.coverageModeEnabled = false`). Map-based explore handles that responsibility now.
- **WFD outer BFS requires a free 8-neighbor.** Already mitigated by start-cell free disc, but if depth fails to carve free space (e.g., model returns 0 everywhere), exploration stalls. Diagnostic print shows depth range + grid state counts every 30 frames.

## Diagnostics in console

```
[MapDriver] depth center=X(Y.YYm) range=[lo..hi] avg=A(B.BBm) metric=YES|NO | grid free=N occ=M unk=K
[MapDriver] tour=exploring/homing/completed pose=(x,z,y=yaw) cell=(col,row) maxDist=Xm frontiers=N path=N cmd=X markedTarget=(c,r)|nil
[MapDriver] STUCK detected — metricMean=X (~Y.YYm) std=Z, reversing
[MapDriver] no progress for Xf — blacklisting goal (col,row) [bl=N]
[MapDriver] target sighted @ cell=(c,r) — heading home (roamed Xm)
[MapDriver] arrived home — tour complete (X frames homing, dist=Ym)
[YOLOPyDriver] XX fps | dets/frame=Y.YY
[YOLOPyDriver] ROAM → SEEK (conf=X)
```

## Architecture diagram

```
Live cam OR SimFPVRenderer (518×518 BGRA @ 75° FOV)
    │
    ├─► YOLOPythonDriver (subprocess) ──► detections ──► updateSeekTarget (gated by maxDist ≥ 2m) ──► markedTarget + tourState=.homing
    │
    └─► MapDriver / DepthDriver
            ├─► CoreML metric depth (DA2-Metric-Indoor)
            ├─► depthBufferToU8 (per-frame, for stuck + display) + depthBufferToMetricU8 (fixed scale, for map)
            ├─► DepthProjector.update (pinhole back-proj, 2.5D filter) → OccupancyGrid (log-odds, persistent obstacles)
            ├─► FrontierExplorer.findClusters (WFD) → cost-ranked → A* (heap, octile)
            └─► WaypointFollower (multi-cell safety + shoulder check) → WASD keys → SimScene.tick (truth pose pushed back)
```

## Memory / cleanup

`tools/venv_depth/` is ~3GB (torch + coremltools weights cache). Safe to delete if you don't need to re-convert; mlpackage in Xcode bundle is what gets shipped. Re-run `python3 -m venv tools/venv_depth` + `pip install` to rebuild.
