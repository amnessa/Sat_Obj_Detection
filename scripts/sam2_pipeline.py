"""Helpers for SAM 2 initialization on VISO satellite tracklets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from PIL import Image
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

from scripts.viso_pipeline import (
    CropWindow,
    compute_crop_window,
    crop_and_resize,
    map_box_to_crop,
  map_points_to_crop,
    pixel_xy_to_normalized_yx,
    pixel_xy_to_tapir_yx,
    xywh_to_xyxy,
)


@dataclass(frozen=True)
class Sam2MaskResult:
  """SAM 2 prediction outputs for one box-prompted frame."""

  mask_full: np.ndarray
  mask_crop: np.ndarray
  crop_frame: np.ndarray
  crop_window: CropWindow
  crop_box_xywh: np.ndarray
  crop_box_xyxy: np.ndarray
  scores: np.ndarray
  logits: np.ndarray
  selected_index: int


@dataclass(frozen=True)
class Sam2PointPromptResult:
  """SAM 2 outputs for a positive point-prompt mask prediction."""

  mask_full: np.ndarray
  mask_crop: np.ndarray
  crop_frame: np.ndarray
  crop_window: CropWindow
  crop_points_xy: np.ndarray
  scores: np.ndarray
  logits: np.ndarray
  selected_index: int


@dataclass(frozen=True)
class Sam2ConsensusResult:
  """Consensus filtering outputs for a keyframe point set."""

  point_masks_full: np.ndarray
  heatmap: np.ndarray
  consensus_mask: np.ndarray
  keep_mask: np.ndarray
  kept_points_xy: np.ndarray
  point_scores: np.ndarray
  crop_frame: np.ndarray
  crop_window: CropWindow
  crop_points_xy: np.ndarray
  max_overlap: int


def infer_torch_device(prefer_cuda: bool = True) -> str:
  """Returns the torch device string for SAM 2."""
  if prefer_cuda and torch.cuda.is_available():
    return 'cuda'
  return 'cpu'


def load_sam2_image_predictor(
    model_id: str = 'facebook/sam2-hiera-tiny',
    device: Optional[str] = None,
    **kwargs,
) -> SAM2ImagePredictor:
  """Loads a pretrained SAM 2 image predictor from Hugging Face."""
  resolved_device = infer_torch_device() if device is None else device
  return SAM2ImagePredictor.from_pretrained(
      model_id,
      device=resolved_device,
      **kwargs,
  )


def resize_binary_mask(
    mask: np.ndarray,
    output_size_xy: Sequence[int],
) -> np.ndarray:
  """Resizes a binary mask with nearest-neighbor resampling."""
  mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
  output_width, output_height = int(output_size_xy[0]), int(output_size_xy[1])
  resized = mask_image.resize(
      (output_width, output_height),
      resample=Image.Resampling.NEAREST,
  )
  return np.array(resized) > 127


def points_xy_to_box_xywh(
    points_xy: np.ndarray,
    min_box_size: int = 8,
) -> np.ndarray:
  """Builds a tight xywh box around a point cloud with outlier rejection."""
  points_xy = np.asarray(points_xy, dtype=np.float32)
  if points_xy.ndim != 2 or points_xy.shape[1] != 2:
    raise ValueError('points_xy must have shape [num_points, 2].')

  median_xy = np.median(points_xy, axis=0)
  absolute_deviation_xy = np.abs(points_xy - median_xy)
  median_absolute_deviation_xy = np.maximum(
      np.median(absolute_deviation_xy, axis=0),
      1e-6,
  )
  keep_mask = np.all(
      absolute_deviation_xy <= 3.0 * median_absolute_deviation_xy,
      axis=1,
  )
  filtered_points_xy = points_xy[keep_mask] if np.any(keep_mask) else points_xy

  x0 = float(filtered_points_xy[:, 0].min())
  y0 = float(filtered_points_xy[:, 1].min())
  x1 = float(filtered_points_xy[:, 0].max())
  y1 = float(filtered_points_xy[:, 1].max())
  width = max(float(min_box_size), x1 - x0 + 1.0)
  height = max(float(min_box_size), y1 - y0 + 1.0)
  center_x = (x0 + x1) / 2.0
  center_y = (y0 + y1) / 2.0
  return np.array(
      [center_x - width / 2.0, center_y - height / 2.0, width, height],
      dtype=np.float32,
  )


def paste_crop_mask_into_full_frame(
    mask_crop: np.ndarray,
    crop_window: CropWindow,
    full_image_shape: Sequence[int],
) -> np.ndarray:
  """Maps a crop-space mask back into the full-frame pixel grid."""
  crop_mask_source = resize_binary_mask(
      mask_crop,
      output_size_xy=(crop_window.width, crop_window.height),
  )
  full_mask = np.zeros(tuple(full_image_shape[:2]), dtype=bool)
  full_mask[crop_window.y0:crop_window.y1, crop_window.x0:crop_window.x1] = (
      crop_mask_source
  )
  return full_mask


def select_best_mask(
    masks: np.ndarray,
    scores: Optional[np.ndarray],
    target_area: Optional[float] = None,
) -> tuple[int, np.ndarray]:
  """Selects the SAM 2 mask that best matches a tiny target."""
  masks = np.asarray(masks)
  if masks.ndim == 2:
    return 0, masks.astype(bool)

  mask_areas = masks.sum(axis=(1, 2)).astype(np.float32)
  valid_indices = np.flatnonzero(mask_areas > 0)
  if len(valid_indices) == 0:
    return 0, masks[0].astype(bool)

  if target_area is not None:
    selected_index = int(
        valid_indices[
            np.argmin(np.abs(mask_areas[valid_indices] - float(target_area)))
        ]
    )
  else:
    del scores
    selected_index = int(valid_indices[np.argmin(mask_areas[valid_indices])])
  return selected_index, masks[selected_index].astype(bool)


def build_negative_prompt_points(
    crop_points_xy: np.ndarray,
    image_size_xy: Sequence[int],
    padding: float = 3.0,
) -> np.ndarray:
  """Places a small negative fence around a positive point cluster."""
  crop_points_xy = np.asarray(crop_points_xy, dtype=np.float32)
  if crop_points_xy.ndim != 2 or crop_points_xy.shape[1] != 2:
    raise ValueError('crop_points_xy must have shape [num_points, 2].')

  width = max(int(image_size_xy[0]), 1)
  height = max(int(image_size_xy[1]), 1)
  x_min = float(crop_points_xy[:, 0].min()) - padding
  x_max = float(crop_points_xy[:, 0].max()) + padding
  y_min = float(crop_points_xy[:, 1].min()) - padding
  y_max = float(crop_points_xy[:, 1].max()) + padding
  x_center = (x_min + x_max) / 2.0
  y_center = (y_min + y_max) / 2.0
  negative_points_xy = np.array(
      [
          [x_min, y_center],
          [x_max, y_center],
          [x_center, y_min],
          [x_center, y_max],
      ],
      dtype=np.float32,
  )
  negative_points_xy[:, 0] = np.clip(negative_points_xy[:, 0], 0.0, width - 1.0)
  negative_points_xy[:, 1] = np.clip(negative_points_xy[:, 1], 0.0, height - 1.0)
  return negative_points_xy


def predict_mask_from_box(
    predictor: SAM2ImagePredictor,
    frame: np.ndarray,
    box_xywh: Sequence[float],
    crop_context_scale: float = 3.0,
    crop_output_size_xy: Sequence[int] = (256, 256),
    min_crop_size: int = 64,
    multimask_output: bool = True,
    return_logits: bool = False,
) -> Sam2MaskResult:
  """Prompts SAM 2 with a box on an upsampled crop around a tiny target."""
  crop_window = compute_crop_window(
      box_xywh=box_xywh,
      image_shape=frame.shape,
      context_scale=crop_context_scale,
      min_crop_size=min_crop_size,
  )
  crop_frame = crop_and_resize(
      frame,
      crop_window=crop_window,
      output_size_xy=crop_output_size_xy,
  )
  crop_box_xywh = map_box_to_crop(
      box_xywh=box_xywh,
      crop_window=crop_window,
      output_size_xy=crop_output_size_xy,
  )
  crop_box_xyxy = xywh_to_xyxy(crop_box_xywh.tolist())

  predictor.set_image(crop_frame)
  masks, scores, logits = predictor.predict(
      box=crop_box_xyxy[None, :],
      multimask_output=multimask_output,
      return_logits=return_logits,
  )
  target_area = float(crop_box_xywh[2] * crop_box_xywh[3])
  selected_index, mask_crop = select_best_mask(
      masks,
      scores,
      target_area=target_area,
  )
  mask_full = paste_crop_mask_into_full_frame(
      mask_crop=mask_crop,
      crop_window=crop_window,
      full_image_shape=frame.shape,
  )

  return Sam2MaskResult(
      mask_full=mask_full,
      mask_crop=mask_crop,
      crop_frame=crop_frame,
      crop_window=crop_window,
      crop_box_xywh=crop_box_xywh,
      crop_box_xyxy=crop_box_xyxy,
      scores=np.asarray(scores),
      logits=np.asarray(logits),
      selected_index=selected_index,
  )


def predict_mask_from_points(
    predictor: SAM2ImagePredictor,
    frame: np.ndarray,
    points_xy: np.ndarray,
    crop_context_scale: float = 3.0,
    crop_output_size_xy: Sequence[int] = (256, 256),
    min_crop_size: int = 64,
    multimask_output: bool = True,
    return_logits: bool = False,
    include_negative_boundary_prompts: bool = True,
    negative_point_padding: float = 3.0,
) -> Sam2PointPromptResult:
  """Prompts SAM 2 with positive points on a crop around a point cloud."""
  points_xy = np.asarray(points_xy, dtype=np.float32)
  if len(points_xy) == 0:
    raise ValueError('points_xy must contain at least one point.')

  prompt_box_xywh = points_xy_to_box_xywh(points_xy, min_box_size=min_crop_size)
  crop_window = compute_crop_window(
      box_xywh=prompt_box_xywh.tolist(),
      image_shape=frame.shape,
      context_scale=crop_context_scale,
      min_crop_size=min_crop_size,
  )
  crop_frame = crop_and_resize(
      frame,
      crop_window=crop_window,
      output_size_xy=crop_output_size_xy,
  )
  crop_points_xy = map_points_to_crop(
      points_xy,
      crop_window=crop_window,
      output_size_xy=crop_output_size_xy,
  )
  prompt_box_crop_xywh = points_xy_to_box_xywh(crop_points_xy, min_box_size=1)
  target_area = float(prompt_box_crop_xywh[2] * prompt_box_crop_xywh[3])

  predictor.set_image(crop_frame)
  prompt_points_xy = crop_points_xy
  point_labels = np.ones((len(crop_points_xy),), dtype=np.int32)
  if include_negative_boundary_prompts:
    negative_points_xy = build_negative_prompt_points(
        crop_points_xy,
        image_size_xy=crop_output_size_xy,
        padding=negative_point_padding,
    )
    prompt_points_xy = np.concatenate([crop_points_xy, negative_points_xy], axis=0)
    point_labels = np.concatenate(
        [
            point_labels,
            np.zeros((len(negative_points_xy),), dtype=np.int32),
        ],
        axis=0,
    )
  masks, scores, logits = predictor.predict(
      point_coords=prompt_points_xy,
      point_labels=point_labels,
      multimask_output=multimask_output,
      return_logits=return_logits,
  )
  selected_index, mask_crop = select_best_mask(
      masks,
      scores,
      target_area=target_area,
  )
  mask_full = paste_crop_mask_into_full_frame(
      mask_crop=mask_crop,
      crop_window=crop_window,
      full_image_shape=frame.shape,
  )

  return Sam2PointPromptResult(
      mask_full=mask_full,
      mask_crop=mask_crop,
      crop_frame=crop_frame,
      crop_window=crop_window,
      crop_points_xy=crop_points_xy,
      scores=np.asarray(scores),
      logits=np.asarray(logits),
      selected_index=selected_index,
  )


def compute_keyframe_consensus(
    predictor: SAM2ImagePredictor,
    frame: np.ndarray,
    points_xy: np.ndarray,
    crop_context_scale: float = 3.0,
    crop_output_size_xy: Sequence[int] = (256, 256),
    min_crop_size: int = 64,
) -> Sam2ConsensusResult:
  """Runs Phase 4 per-point prompting and filters points by consensus."""
  points_xy = np.asarray(points_xy, dtype=np.float32)
  if len(points_xy) == 0:
    raise ValueError('points_xy must contain at least one point.')

  prompt_box_xywh = points_xy_to_box_xywh(points_xy, min_box_size=min_crop_size)
  crop_window = compute_crop_window(
      box_xywh=prompt_box_xywh.tolist(),
      image_shape=frame.shape,
      context_scale=crop_context_scale,
      min_crop_size=min_crop_size,
  )
  crop_frame = crop_and_resize(
      frame,
      crop_window=crop_window,
      output_size_xy=crop_output_size_xy,
  )
  crop_points_xy = map_points_to_crop(
      points_xy,
      crop_window=crop_window,
      output_size_xy=crop_output_size_xy,
  )
  prompt_box_crop_xywh = points_xy_to_box_xywh(crop_points_xy, min_box_size=1)
  target_area = float(prompt_box_crop_xywh[2] * prompt_box_crop_xywh[3])

  predictor.set_image(crop_frame)
  point_masks_full = []
  point_scores = []
  for point_xy in crop_points_xy:
    masks, scores, _ = predictor.predict(
        point_coords=point_xy[None, :],
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
        return_logits=False,
    )
    selected_index, mask_crop = select_best_mask(
      masks,
      scores,
      target_area=target_area,
    )
    point_scores.append(float(np.asarray(scores)[selected_index]))
    point_masks_full.append(
        paste_crop_mask_into_full_frame(
            mask_crop=mask_crop,
            crop_window=crop_window,
            full_image_shape=frame.shape,
        )
    )

  point_masks_full = np.stack(point_masks_full, axis=0)
  heatmap = point_masks_full.sum(axis=0).astype(np.int32)
  max_overlap = int(heatmap.max())
  consensus_mask = heatmap == max_overlap
  keep_mask = np.array(
      [
          bool(np.any(point_mask & consensus_mask))
          for point_mask in point_masks_full
      ],
      dtype=bool,
  )
  kept_points_xy = points_xy[keep_mask]

  return Sam2ConsensusResult(
      point_masks_full=point_masks_full,
      heatmap=heatmap,
      consensus_mask=consensus_mask,
      keep_mask=keep_mask,
      kept_points_xy=kept_points_xy,
      point_scores=np.asarray(point_scores, dtype=np.float32),
      crop_frame=crop_frame,
      crop_window=crop_window,
      crop_points_xy=crop_points_xy,
      max_overlap=max_overlap,
  )


def sample_mask_points(
    mask: np.ndarray,
    num_points: int,
    seed: Optional[int] = None,
) -> np.ndarray:
  """Randomly samples [x, y] points from a binary mask."""
  if num_points <= 0:
    raise ValueError('num_points must be positive.')

  ys, xs = np.nonzero(mask)
  if len(xs) == 0:
    raise ValueError('Cannot sample points from an empty mask.')

  all_points_xy = np.stack([xs, ys], axis=1).astype(np.float32)
  rng = np.random.default_rng(seed)
  replace = len(all_points_xy) < num_points
  sampled_indices = rng.choice(len(all_points_xy), size=num_points, replace=replace)
  return all_points_xy[sampled_indices]


def sample_tapir_points_from_mask(
    mask: np.ndarray,
    num_points: int,
    image_size_xy: Sequence[int],
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Samples mask points and returns pixel xy, TAPIR yx, and normalized yx."""
  points_xy = sample_mask_points(mask=mask, num_points=num_points, seed=seed)
  points_yx = pixel_xy_to_tapir_yx(points_xy)
  normalized_yx = pixel_xy_to_normalized_yx(points_xy, image_size_xy=image_size_xy)
  return points_xy, points_yx, normalized_yx