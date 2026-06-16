The reason your evaluation metrics are hitting 0.0% while the paper achieves ~63.9% DPR and ~36.5% OSR comes down to how SAM 2 inherently scores its own outputs and how your pipeline extracts bounding boxes from point clouds.

Since VISO targets (like cars) are extremely small and lack distinct internal textures, the standard SAM 2 confidence logic breaks down. Here are the root causes in your `sam2_pipeline.py` implementation and how to patch them to match the paper's expected results.

### 1. The Mask Selection Flaw (Over-segmentation)

**The Problem:** In `sam2_pipeline.py`, your `select_best_mask` function uses `np.argmax(scores)`. When SAM 2 evaluates a tiny, low-resolution object, it typically returns three masks (e.g., the car, the lane, and the entire intersection). Because the car is so blurry, SAM 2 assigns the *highest confidence score* to the largest mask (the intersection). Consequently, TAPIR samples 20 points across the entire road, tracks the background, and your Intersection over Union (IoU) instantly drops to 0.

**The Fix (Area-Aware Selection):** Since you pass a ground-truth bounding box in Phase 2, the pipeline knows exactly how big the car is supposed to be. You must rewrite `select_best_mask` to ignore SAM 2's confidence scores and instead choose the mask whose area closest matches the target's area (or default to the smallest valid mask for point prompts).

Replace your `select_best_mask` in `sam2_pipeline.py` with this:

```python
def select_best_mask(
    masks: np.ndarray,
    scores: Optional[np.ndarray],
    target_area: Optional[float] = None,
) -> tuple[int, np.ndarray]:
  """Selects the SAM 2 mask that best fits the expected object size."""
  masks = np.asarray(masks)
  if masks.ndim == 2:
    return 0, masks.astype(bool)

  # Calculate the pixel area of all generated masks
  areas = masks.sum(axis=(1, 2))
  valid_idx = np.where(areas > 0)[0]

  if len(valid_idx) == 0:
    return 0, masks[0].astype(bool)

  if target_area is not None:
    # Phase 2: Pick the mask closest to the initial bounding box area
    best_idx = valid_idx[np.argmin(np.abs(areas[valid_idx] - target_area))]
  else:
    # Phase 4/5: For point-prompts on VISO, default to the smallest non-empty mask
    # to prevent bleeding into the asphalt.
    best_idx = valid_idx[np.argmin(areas[valid_idx])]

  return int(best_idx), masks[best_idx].astype(bool)

```

You will then need to update `predict_mask_from_box` to pass the area to this new function:

```python
# Inside predict_mask_from_box, right after predictor.predict(...)
target_area = float(crop_box_xywh[2] * crop_box_xywh[3])
selected_index, mask_crop = select_best_mask(masks, scores, target_area=target_area)

```

### 2. The Bounding Box Extraction Flaw (Point Drift Explosion)

**The Problem:** In `sam2_pipeline.py`, your `points_xy_to_box_xywh` function uses absolute `min()` and `max()` coordinates to draw a bounding box around the TAPIR points. TAPIR is highly accurate, but it is standard for 1 or 2 points out of 20 to drift off the car onto the background due to motion blur. If 19 points are on the car, but 1 point drifts 50 pixels away, `min/max` will draw a massive bounding box covering the car and the 50 pixels of empty space, destroying your IoU and Success Rate.

**The Fix (Outlier Rejection):** You must filter out rogue TAPIR points before calculating the bounding box limits. Applying a simple Median Absolute Deviation (MAD) filter will isolate the dense cluster of points actually on the vehicle.

Update `points_xy_to_box_xywh` in `sam2_pipeline.py`:

```python
def points_xy_to_box_xywh(
    points_xy: np.ndarray,
    min_box_size: int = 8,
) -> np.ndarray:
  """Builds a tight xywh box around a point cloud with outlier rejection."""
  points_xy = np.asarray(points_xy, dtype=np.float32)
  if points_xy.ndim != 2 or points_xy.shape[1] != 2:
    raise ValueError('points_xy must have shape [num_points, 2].')

  # --- OUTLIER REJECTION ---
  # Find the median center of the point cloud
  median = np.median(points_xy, axis=0)
  diff = np.abs(points_xy - median)
  mad = np.median(diff, axis=0) + 1e-6 # prevent division by zero

  # Keep points within ~3 MADs of the center
  keep_idx = np.all(diff < 3.0 * mad, axis=1)
  filtered_points = points_xy[keep_idx] if np.any(keep_idx) else points_xy

  # Use the filtered cluster to draw the bounding box limits
  x0 = float(filtered_points[:, 0].min())
  y0 = float(filtered_points[:, 1].min())
  x1 = float(filtered_points[:, 0].max())
  y1 = float(filtered_points[:, 1].max())

  width = max(float(min_box_size), x1 - x0 + 1.0)
  height = max(float(min_box_size), y1 - y0 + 1.0)
  center_x = (x0 + x1) / 2.0
  center_y = (y0 + y1) / 2.0
  return np.array(
      [center_x - width / 2.0, center_y - height / 2.0, width, height],
      dtype=np.float32,
  )

```

### 3. Adding Negative Prompts (Optional but Highly Recommended)

If you still see SAM 2 bleeding into the background during the Keyframe Renewal phase (Phase 5), it is because you are only supplying positive points `np.ones(...)` to SAM 2.

If you look at the `predict_mask_from_points` function in `sam2_pipeline.py`, you can automatically generate "Negative" boundary points by taking the min/max coordinates of your `crop_points_xy`, expanding them slightly (e.g., by 3 pixels), and adding them to the prompt array with a label of `0`. This creates a mathematical "fence" that physically prevents SAM 2's mask from expanding past the edges of the vehicle into the background context.