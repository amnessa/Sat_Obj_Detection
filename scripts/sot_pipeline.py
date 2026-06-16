"""Sequence-level helpers for keyframe-refreshed satellite object tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor

from scripts.sam2_pipeline import (
    compute_keyframe_consensus,
    points_xy_to_box_xywh,
    predict_mask_from_box,
    predict_mask_from_points,
    sample_tapir_points_from_mask,
)
from scripts.tapir_pipeline import (
    CausalTapirChunkTracker,
    ChunkTrackingResult,
    pixel_xy_to_query_points_tyx,
)
from scripts.viso_pipeline import (
    CropWindow,
  crop_and_resize,
    TrackletSpec,
    compute_crop_window,
    get_frame_chunk,
    load_tracklet_boxes,
    map_points_from_crop,
    map_points_to_crop,
    xywh_to_xyxy,
)


@dataclass(frozen=True)
class KeyframeCycleResult:
  """Stores one TAPIR-to-SAM refresh cycle ending at a keyframe."""

  chunk_index: int
  start_offset: int
  end_offset: int
  frame_indices: np.ndarray
  query_points_xy: np.ndarray
  tapir_result: ChunkTrackingResult
  final_visible_points_xy: np.ndarray
  kept_points_xy: np.ndarray
  refreshed_points_xy: np.ndarray
  renewed_mask_full: np.ndarray
  predicted_box_xywh: np.ndarray
  ground_truth_box_xywh: np.ndarray
  consensus_max_overlap: int

  @property
  def keyframe_index(self) -> int:
    return int(self.frame_indices[-1])

  @property
  def renewed_mask_pixels(self) -> int:
    return int(self.renewed_mask_full.sum())


@dataclass(frozen=True)
class SequenceTrackingResult:
  """Stores the sequence-level outputs of the keyframe refresh loop."""

  tracklet: TrackletSpec
  chunk_size: int
  chunk_stride: int
  num_query_points: int
  inference_size_xy: Optional[tuple[int, int]]
  initial_points_xy: np.ndarray
  initial_mask_pixels: int
  keyframes: tuple[KeyframeCycleResult, ...]

  @property
  def keyframe_frame_indices(self) -> np.ndarray:
    return np.asarray([result.keyframe_index for result in self.keyframes], dtype=np.int32)

  @property
  def predicted_boxes_xywh(self) -> np.ndarray:
    return np.stack([result.predicted_box_xywh for result in self.keyframes], axis=0)

  @property
  def ground_truth_boxes_xywh(self) -> np.ndarray:
    return np.stack([result.ground_truth_box_xywh for result in self.keyframes], axis=0)


@dataclass(frozen=True)
class KeyframeEvaluationResult:
  """DPR/OSR metrics computed over keyframe box predictions."""

  frame_indices: np.ndarray
  predicted_boxes_xywh: np.ndarray
  ground_truth_boxes_xywh: np.ndarray
  center_errors_px: np.ndarray
  ious: np.ndarray
  dpr_pass: np.ndarray
  osr_pass: np.ndarray
  center_threshold_px: float
  iou_threshold: float

  @property
  def dpr(self) -> float:
    return float(np.mean(self.dpr_pass, dtype=np.float32) * 100.0)

  @property
  def osr(self) -> float:
    return float(np.mean(self.osr_pass, dtype=np.float32) * 100.0)


def mask_to_box_xywh(mask: np.ndarray) -> np.ndarray:
  """Converts a binary mask into a tight [x, y, w, h] box."""
  ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
  if len(xs) == 0:
    raise ValueError('Cannot convert an empty mask to a bounding box.')

  x0 = float(xs.min())
  y0 = float(ys.min())
  x1 = float(xs.max()) + 1.0
  y1 = float(ys.max()) + 1.0
  return np.array([x0, y0, x1 - x0, y1 - y0], dtype=np.float32)


def compute_chunk_start_offsets(
    num_frames: int,
    chunk_size: int,
    chunk_stride: int,
) -> np.ndarray:
  """Returns the zero-based chunk start offsets for a tracklet."""
  if num_frames <= 0:
    raise ValueError('num_frames must be positive.')
  if chunk_size <= 0:
    raise ValueError('chunk_size must be positive.')
  if chunk_stride <= 0:
    raise ValueError('chunk_stride must be positive.')

  return np.arange(0, num_frames, chunk_stride, dtype=np.int32)


def resample_points_xy(
    points_xy: np.ndarray,
    num_points: int,
    seed: Optional[int] = None,
) -> np.ndarray:
  """Samples a fixed-size point cloud from an existing set of points."""
  points_xy = np.asarray(points_xy, dtype=np.float32)
  if points_xy.ndim != 2 or points_xy.shape[1] != 2:
    raise ValueError('points_xy must have shape [num_points, 2].')
  if len(points_xy) == 0:
    raise ValueError('points_xy must contain at least one point.')
  if num_points <= 0:
    raise ValueError('num_points must be positive.')

  rng = np.random.default_rng(seed)
  replace = len(points_xy) < num_points
  sampled_indices = rng.choice(len(points_xy), size=num_points, replace=replace)
  return points_xy[sampled_indices]


def box_center_xy(boxes_xywh: np.ndarray) -> np.ndarray:
  """Returns [x, y] centers for one or more xywh boxes."""
  boxes_xywh = np.asarray(boxes_xywh, dtype=np.float32)
  return boxes_xywh[..., :2] + boxes_xywh[..., 2:] / 2.0


def compute_center_location_errors(
    predicted_boxes_xywh: np.ndarray,
    ground_truth_boxes_xywh: np.ndarray,
) -> np.ndarray:
  """Computes per-box Euclidean center location error in pixels."""
  predicted_centers_xy = box_center_xy(predicted_boxes_xywh)
  ground_truth_centers_xy = box_center_xy(ground_truth_boxes_xywh)
  return np.linalg.norm(predicted_centers_xy - ground_truth_centers_xy, axis=-1)


def compute_box_ious(
    predicted_boxes_xywh: np.ndarray,
    ground_truth_boxes_xywh: np.ndarray,
) -> np.ndarray:
  """Computes per-box IoU between predicted and ground-truth xywh boxes."""
  predicted_xyxy = np.asarray(
      [xywh_to_xyxy(box) for box in np.asarray(predicted_boxes_xywh, dtype=np.float32)],
      dtype=np.float32,
  )
  ground_truth_xyxy = np.asarray(
      [xywh_to_xyxy(box) for box in np.asarray(ground_truth_boxes_xywh, dtype=np.float32)],
      dtype=np.float32,
  )

  inter_x0 = np.maximum(predicted_xyxy[:, 0], ground_truth_xyxy[:, 0])
  inter_y0 = np.maximum(predicted_xyxy[:, 1], ground_truth_xyxy[:, 1])
  inter_x1 = np.minimum(predicted_xyxy[:, 2], ground_truth_xyxy[:, 2])
  inter_y1 = np.minimum(predicted_xyxy[:, 3], ground_truth_xyxy[:, 3])

  inter_w = np.maximum(0.0, inter_x1 - inter_x0)
  inter_h = np.maximum(0.0, inter_y1 - inter_y0)
  inter_area = inter_w * inter_h

  predicted_area = np.maximum(0.0, predicted_xyxy[:, 2] - predicted_xyxy[:, 0]) * np.maximum(
      0.0, predicted_xyxy[:, 3] - predicted_xyxy[:, 1]
  )
  ground_truth_area = np.maximum(
      0.0, ground_truth_xyxy[:, 2] - ground_truth_xyxy[:, 0]
  ) * np.maximum(0.0, ground_truth_xyxy[:, 3] - ground_truth_xyxy[:, 1])
  union_area = predicted_area + ground_truth_area - inter_area

  ious = np.zeros_like(inter_area, dtype=np.float32)
  valid_union = union_area > 0.0
  ious[valid_union] = inter_area[valid_union] / union_area[valid_union]
  return ious


def crop_frames_to_window(
    frames: np.ndarray,
    crop_window: CropWindow,
  output_size_xy: Sequence[int],
) -> np.ndarray:
  """Crops and resizes a frame chunk to a fixed local TAPIR working grid."""
  return np.stack(
    [
      crop_and_resize(
        frame,
        crop_window=crop_window,
        output_size_xy=output_size_xy,
      )
      for frame in frames
    ],
    axis=0,
  ).astype(np.uint8)


def track_chunk_with_local_tapir_crop(
    tapir_tracker: CausalTapirChunkTracker,
    frames: np.ndarray,
    query_points_xy: np.ndarray,
    crop_size: int = 256,
    crop_context_scale: float = 1.0,
) -> tuple[ChunkTrackingResult, CropWindow]:
  """Tracks a chunk on a local full-resolution crop instead of a downsampled frame."""
  prompt_box_xywh = points_xy_to_box_xywh(query_points_xy, min_box_size=crop_size)
  crop_window = compute_crop_window(
      box_xywh=prompt_box_xywh.tolist(),
      image_shape=frames[0].shape,
      context_scale=crop_context_scale,
      min_crop_size=crop_size,
  )
  crop_output_size_xy = (crop_size, crop_size)
  crop_frames = crop_frames_to_window(
      frames,
      crop_window=crop_window,
      output_size_xy=crop_output_size_xy,
  )
  crop_query_points_xy = map_points_to_crop(
      query_points_xy,
      crop_window=crop_window,
      output_size_xy=crop_output_size_xy,
  )
  crop_tracking_result = tapir_tracker.track_chunk(
      frames=crop_frames,
      query_points_xy=crop_query_points_xy,
      inference_size_xy=None,
  )
  full_tracks_xy = map_points_from_crop(
      crop_tracking_result.tracks_xy.reshape(-1, 2),
      crop_window=crop_window,
      output_size_xy=crop_output_size_xy,
  ).reshape(crop_tracking_result.tracks_xy.shape)

  return (
      ChunkTrackingResult(
          query_points_xy=np.asarray(query_points_xy, dtype=np.float32),
          query_points_tyx=pixel_xy_to_query_points_tyx(query_points_xy),
          tracks_xy=np.asarray(full_tracks_xy, dtype=np.float32),
          tracks_xy_inference=crop_tracking_result.tracks_xy_inference,
          occlusion_logits=crop_tracking_result.occlusion_logits,
          expected_dist_logits=crop_tracking_result.expected_dist_logits,
          visibles=crop_tracking_result.visibles,
          inference_size_xy=crop_tracking_result.inference_size_xy,
      ),
      crop_window,
  )


def evaluate_keyframe_tracking(
    tracking_result: SequenceTrackingResult,
    center_threshold_px: float = 5.0,
    iou_threshold: float = 0.5,
) -> KeyframeEvaluationResult:
  """Evaluates keyframe box predictions with VISO-style DPR and OSR."""
  predicted_boxes_xywh = tracking_result.predicted_boxes_xywh
  ground_truth_boxes_xywh = tracking_result.ground_truth_boxes_xywh
  center_errors_px = compute_center_location_errors(
      predicted_boxes_xywh=predicted_boxes_xywh,
      ground_truth_boxes_xywh=ground_truth_boxes_xywh,
  )
  ious = compute_box_ious(
      predicted_boxes_xywh=predicted_boxes_xywh,
      ground_truth_boxes_xywh=ground_truth_boxes_xywh,
  )
  dpr_pass = center_errors_px <= float(center_threshold_px)
  osr_pass = ious >= float(iou_threshold)

  return KeyframeEvaluationResult(
      frame_indices=tracking_result.keyframe_frame_indices,
      predicted_boxes_xywh=predicted_boxes_xywh,
      ground_truth_boxes_xywh=ground_truth_boxes_xywh,
      center_errors_px=center_errors_px.astype(np.float32),
      ious=ious.astype(np.float32),
      dpr_pass=dpr_pass,
      osr_pass=osr_pass,
      center_threshold_px=float(center_threshold_px),
      iou_threshold=float(iou_threshold),
  )


def track_tracklet_with_keyframe_refresh(
    tracklet: TrackletSpec,
    predictor: SAM2ImagePredictor,
    tapir_tracker: CausalTapirChunkTracker,
    num_query_points: int,
    chunk_size: int,
    inference_size_xy: Optional[Sequence[int]] = (512, 512),
    chunk_stride: Optional[int] = None,
    crop_context_scale: float = 3.0,
    crop_output_size_xy: Sequence[int] = (256, 256),
    min_crop_size: int = 64,
    seed: int = 0,
    tapir_crop_size: int = 256,
    tapir_crop_context_scale: float = 1.0,
) -> SequenceTrackingResult:
  """Runs the SAM 2 <-> TAPIR loop across an entire VISO tracklet.

  When chunk_stride is omitted, the loop reuses each keyframe as the first frame
  of the next chunk by stepping forward chunk_size - 1 frames.
  """
  if num_query_points <= 0:
    raise ValueError('num_query_points must be positive.')
  if chunk_size <= 0:
    raise ValueError('chunk_size must be positive.')

  resolved_chunk_stride = (
      max(1, chunk_size - 1) if chunk_stride is None else int(chunk_stride)
  )
  if resolved_chunk_stride <= 0:
    raise ValueError('chunk_stride must be positive.')

  tracklet_boxes_xywh = load_tracklet_boxes(tracklet.tracklet_file)
  initial_chunk = get_frame_chunk(tracklet, start_offset=0, chunk_size=chunk_size)
  initial_mask_result = predict_mask_from_box(
      predictor=predictor,
      frame=initial_chunk.frames[0],
      box_xywh=initial_chunk.boxes_xywh[0],
      crop_context_scale=crop_context_scale,
      crop_output_size_xy=crop_output_size_xy,
      min_crop_size=min_crop_size,
  )
  current_points_xy, _, _ = sample_tapir_points_from_mask(
      mask=initial_mask_result.mask_full,
      num_points=num_query_points,
      image_size_xy=(initial_chunk.frames.shape[2], initial_chunk.frames.shape[1]),
      seed=seed,
  )
  initial_points_xy = np.asarray(current_points_xy, dtype=np.float32)
  true_target_area_px = float(initial_mask_result.mask_full.sum())

  keyframe_results = []
  chunk_start_offsets = compute_chunk_start_offsets(
      num_frames=len(tracklet_boxes_xywh),
      chunk_size=chunk_size,
      chunk_stride=resolved_chunk_stride,
  )
  if tapir_crop_size <= 0:
    raise ValueError('tapir_crop_size must be positive.')
  if tapir_crop_context_scale < 1.0:
    raise ValueError('tapir_crop_context_scale must be at least 1.0.')

  for chunk_index, start_offset in enumerate(chunk_start_offsets.tolist()):
    chunk = get_frame_chunk(
        tracklet=tracklet,
        start_offset=int(start_offset),
        chunk_size=chunk_size,
    )
    tapir_result, tapir_crop_window = track_chunk_with_local_tapir_crop(
        tapir_tracker=tapir_tracker,
        frames=chunk.frames,
        query_points_xy=current_points_xy,
        crop_size=tapir_crop_size,
        crop_context_scale=tapir_crop_context_scale,
    )
    del tapir_crop_window

    final_visible_mask = tapir_result.visibles[-1].astype(bool)
    final_visible_points_xy = tapir_result.final_points_xy[final_visible_mask]
    if len(final_visible_points_xy) == 0:
      final_visible_points_xy = tapir_result.final_points_xy
    consensus_result = compute_keyframe_consensus(
        predictor=predictor,
        frame=chunk.frames[-1],
        points_xy=final_visible_points_xy,
        crop_context_scale=crop_context_scale,
        crop_output_size_xy=crop_output_size_xy,
        min_crop_size=min_crop_size,
        true_target_area_px=true_target_area_px,
    )
    kept_points_xy = consensus_result.kept_points_xy
    if len(kept_points_xy) == 0:
      kept_points_xy = final_visible_points_xy

    renewed_mask_result = predict_mask_from_points(
        predictor=predictor,
        frame=chunk.frames[-1],
        points_xy=kept_points_xy,
        crop_context_scale=crop_context_scale,
        crop_output_size_xy=crop_output_size_xy,
        min_crop_size=min_crop_size,
        true_target_area_px=true_target_area_px,
    )

    if np.any(renewed_mask_result.mask_full):
      predicted_box_xywh = mask_to_box_xywh(renewed_mask_result.mask_full)
      refreshed_points_xy, _, _ = sample_tapir_points_from_mask(
          mask=renewed_mask_result.mask_full,
          num_points=num_query_points,
          image_size_xy=(chunk.frames.shape[2], chunk.frames.shape[1]),
          seed=seed + chunk_index + 1,
      )
    else:
      predicted_box_xywh = points_xy_to_box_xywh(kept_points_xy, min_box_size=1)
      refreshed_points_xy = resample_points_xy(
          kept_points_xy,
          num_points=num_query_points,
          seed=seed + chunk_index + 1,
      )

    keyframe_results.append(
        KeyframeCycleResult(
            chunk_index=chunk_index,
            start_offset=int(start_offset),
            end_offset=int(start_offset + len(chunk.frame_indices) - 1),
            frame_indices=chunk.frame_indices,
            query_points_xy=np.asarray(current_points_xy, dtype=np.float32),
            tapir_result=tapir_result,
            final_visible_points_xy=np.asarray(final_visible_points_xy, dtype=np.float32),
            kept_points_xy=np.asarray(kept_points_xy, dtype=np.float32),
            refreshed_points_xy=np.asarray(refreshed_points_xy, dtype=np.float32),
            renewed_mask_full=np.asarray(renewed_mask_result.mask_full, dtype=bool),
            predicted_box_xywh=np.asarray(predicted_box_xywh, dtype=np.float32),
            ground_truth_box_xywh=np.asarray(chunk.boxes_xywh[-1], dtype=np.float32),
            consensus_max_overlap=int(consensus_result.max_overlap),
        )
    )
    current_points_xy = refreshed_points_xy

  return SequenceTrackingResult(
      tracklet=tracklet,
      chunk_size=chunk_size,
      chunk_stride=resolved_chunk_stride,
      num_query_points=num_query_points,
        inference_size_xy=(tapir_crop_size, tapir_crop_size),
      initial_points_xy=initial_points_xy,
      initial_mask_pixels=int(initial_mask_result.mask_full.sum()),
      keyframes=tuple(keyframe_results),
  )