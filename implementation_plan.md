
---

### Phase 1: Data Pipeline & Pre-Processing

Before touching the models, you need a data harness that can feed frames to both TAPIR and SAM 2 sequentially.

* **Frame Generator:** Create a video data loader that reads images from your `dataset` folder. It needs to serve frames in chunks of length $K$ (e.g., $K=20$ frames).
* **Coordinate Space Matching:** Note the coordinate formats! TAPIR typically handles normalized coordinates $[y, x]$ ranging from $0$ to $1$ relative to the frame size, while SAM 2 expects absolute pixel values $[x, y]$. Your pipeline must include a modular coordinate conversion utility to pass points back and forth seamlessly.
* **Satellite Cropping Up-sampler:** Because satellite targets are tiny, implement a logical crop-and-zoom utility. When given the initial bounding box, the data pipeline should tightly crop around the target area and upscale it before passing it to SAM 2 to help it resolve distinct boundaries.

### Phase 2: First Frame Initialization ($T=0$)

This phase runs exactly once at the beginning of the tracking sequence.

* **Setup the Predictor:** Load your SAM 2 weights using its standalone image predictor mode.
* **Generate the Anchor Mask:** Pass the first video frame and the ground-truth VISO bounding box into SAM 2 as a box prompt. SAM 2 will return a high-resolution binary pixel mask of the car.
* **Point Cloud Sampling:** Find all pixel coordinates where the binary mask value is `True`. Use a random selection algorithm to extract exactly $N$ coordinates (e.g., $N=20$) distributed across the car's mask. Convert these absolute coordinates into TAPIR's expected input format.

### Phase 3: Point Propagation via TAPIR

This handles the frame-by-frame tracking between your keyframes.

* **Chunk Execution:** Pass the current frame chunk (from frame $t$ to frame $t+K$) and your $N$ tracking points into TAPIR's inference function.
* **Trajectory Extraction:** Allow TAPIR to track the points via optical flow across the $K$ frames. Extract the final predicted coordinate positions of those $N$ points at the exact boundary of the keyframe ($t+K$).

---

### Phase 4: SAM 2 Keyframe Consensus Mechanism

At frame $t+K$, you must evaluate which points TAPIR tracked successfully and which ones drifted onto the background asphalt or other cars.

* **Individual Point Querying:** Loop through your $N$ tracked coordinates. For *each* point, prompt SAM 2 on the current keyframe frame using that single coordinate as a "positive/foreground" point prompt. SAM 2 will output an independent mask for that specific point.
* **Heatmap Accumulation:** Initialize an empty matrix matching the frame's dimensions. For every individual mask generated in the loop, perform a point-wise addition into this matrix. This creates an overlap heatmap where pixels with higher values indicate strong consensus among the tracking points.
* **Thresholding and Consensus Filtering:** Isolate the area in the heatmap with the highest overlapping values. Check which of your original $N$ tracking points fell inside this high-consensus zone. Keep those coordinates, and discard any points that fell outside of it (the drifted points).

### Phase 5: Mask Renewal & Re-Sampling

Before kicking off the next tracking chunk, you refresh the points to prevent error accumulation.

* **Multi-Point Re-masking:** Take your filtered, surviving points and pass them *collectively* to SAM 2 as a multi-point foreground prompt. This instructs SAM 2 to generate a single, highly refined, unified mask of the target vehicle at the keyframe.
* **Point Refresh:** Erase your tracking point memory completely. Perform a fresh random sampling routine on this brand-new SAM 2 mask to select a pristine set of $N$ points.
* **Loop Continuum:** Advance your timeline index by $K$, pass the new points and the next frame chunk to TAPIR, and repeat from Phase 3.

---

### Metric Evaluation

To evaluate your system against the VISO benchmark baselines mentioned in the previous papers, your evaluation wrapper needs to take the final unified SAM 2 mask at every keyframe, compute the minimum and maximum coordinate boundaries to wrap a bounding box around it, and compare it to the VISO ground-truth using Center Location Error (for Precision) and Intersection over Union (for Success Rate).

What specific category of vehicle from the VISO dataset (e.g., cars, planes, or trains) are you planning to run your first test execution on?