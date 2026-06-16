When running this pipeline on an RTX 4060 (which has 8GB of VRAM), the original 1025x1025 VISO frames cause an Out-Of-Memory (OOM) error during optical flow allocation. To bypass this, the pipeline was instructed to downsample the images to `512x512` for TAPIR tracking using bilinear interpolation.

While this prevents the GPU from crashing, it introduces two catastrophic architectural flaws that destroy the tracking metrics. Here is the breakdown of the exact mistakes in the logic and how to fix them so the pipeline runs successfully on your current GPU.

### 1. The TAPIR Resolution Trap (Point Drift)

**The Flaw:** Satellite tracking targets are extremely small. In your initial frame, the car is represented by a bounding box of just `6.0 by 9.0` pixels. By downsampling the 1025x1025 image to `512x512`, you shrink the car to roughly `3x4` pixels. Bilinear interpolation essentially blends this tiny 3x4 white blob into the gray asphalt, destroying all trackable features.
Consequently, TAPIR loses the car in the very first frame. It locks onto the macro-textures of the asphalt instead, which is why the visual output in "Phase 4 input" shows the yellow TAPIR points trailing in a line behind the car rather than staying on it.

**The Fix (Dynamic Local Cropping):**
Do not downsample the entire image for TAPIR. Instead, crop a small window around the car's expected location from the *original* high-resolution frames, and pass only that crop to TAPIR.
In `sot_pipeline.py`, update the TAPIR loop execution to pass dynamically cropped frames:

1. Use the `compute_crop_window` function with the `current_points_xy` to generate a local window (e.g., 256x256 pixels) in the full frame.
2. Crop all `chunk.frames` using this window.
3. Translate `current_points_xy` to the crop's coordinate space using `map_points_to_crop`.
4. Run `tapir_tracker.track_chunk` on the 256x256 crops with `inference_size_xy=None`. (This will use less than 3GB of VRAM and preserve the 1:1 pixel crispness).
5. Translate the resulting `tracks_xy` back to full-frame coordinates using `map_points_from_crop`.

### 2. The SAM 2 Target Area Feedback Loop (Over-segmentation)

**The Flaw:** In `sam2_pipeline.py`, both `compute_keyframe_consensus` and `predict_mask_from_points` dynamically calculate `target_area` based on the bounding box of the scattered TAPIR points:

```python
prompt_box_crop_xywh = points_xy_to_box_xywh(crop_points_xy, min_box_size=1)
target_area = float(prompt_box_crop_xywh[2] * prompt_box_crop_xywh[3])

```

Because TAPIR is failing and the points are drifting across the road, this dynamic bounding box becomes massive. SAM 2 is then explicitly ordered to find a mask that matches this massive area. As a result, SAM 2 selects the entire road intersection, creating the giant heatmaps and oversized keyframe masks visible in your output.

**The Fix (Constant Area Anchoring):**
The physical area of the car does not change. You must anchor SAM 2's mask selection strictly to the initial size of the car found in Phase 2, completely decoupling it from the point cloud's drift.

1. **In `sot_pipeline.py` (Phase 2):** Save the absolute pixel area of the target from the very first frame:
```python
true_car_area_px = float(initial_mask_result.mask_full.sum())

```


Pass `true_car_area_px` down into `compute_keyframe_consensus` and `predict_mask_from_points` inside your chunk loop.
2. **In `sam2_pipeline.py`:** Delete the flawed `prompt_box_crop_xywh` target area calculation. Instead, mathematically scale the `true_car_area_px` to match the crop frame's current resolution:
```python
# Inside compute_keyframe_consensus and predict_mask_from_points
scale_x = crop_output_size_xy[0] / crop_window.width
scale_y = crop_output_size_xy[1] / crop_window.height
target_area = true_car_area_px * (scale_x * scale_y)

# Pass this constant target_area to select_best_mask

```



By implementing the dynamic spatial crop for TAPIR and the anchored `target_area` for SAM 2, the pipeline will easily fit inside the RTX 4060's VRAM constraints while dramatically improving your Distance Precision Rate (DPR) and Overlap Success Rate (OSR).