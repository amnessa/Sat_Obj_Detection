"""Utilities for loading and preparing VISO SOT tracklets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
from PIL import Image


IMAGE_DIRNAME = "img"
TRACKLET_DIRNAME = "gt"
IMAGE_SUFFIX = ".jpg"


@dataclass(frozen=True)
class CropWindow:
  """Describes a crop in source-image coordinates."""

  x0: int
  y0: int
  x1: int
  y1: int
  source_size_xy: tuple[int, int]

  @property
  def width(self) -> int:
    return self.x1 - self.x0

  @property
  def height(self) -> int:
    return self.y1 - self.y0


@dataclass(frozen=True)
class TrackletSpec:
  """Identifies one VISO SOT object tracklet."""

  category: str
  sequence_id: str
  object_id: int
  start_frame: int
  end_frame: int
  frame_dir: Path
  tracklet_file: Path

  @property
  def num_frames(self) -> int:
    return self.end_frame - self.start_frame + 1


@dataclass(frozen=True)
class FrameChunk:
  """Holds a consecutive chunk of frames and their boxes."""

  frame_indices: np.ndarray
  frames: np.ndarray
  boxes_xywh: np.ndarray

  @property
  def image_shape(self) -> tuple[int, int, int]:
    return tuple(self.frames.shape[1:])


def _as_path(path_like: str | Path) -> Path:
  return path_like if isinstance(path_like, Path) else Path(path_like)


def parse_tracklet_filename(tracklet_file: str | Path) -> tuple[int, int, int]:
  """Parses <object_id>_<start_frame>_<end_frame>.txt."""
  stem = _as_path(tracklet_file).stem
  parts = stem.split("_")
  if len(parts) != 3:
    raise ValueError(
        "Tracklet filename must look like '<object_id>_<start>_<end>.txt'."
    )
  object_id, start_frame, end_frame = (int(part) for part in parts)
  return object_id, start_frame, end_frame


def build_tracklet_spec(tracklet_file: str | Path) -> TrackletSpec:
  """Builds a typed tracklet description from a VISO gt file path."""
  tracklet_path = _as_path(tracklet_file).resolve()
  object_id, start_frame, end_frame = parse_tracklet_filename(tracklet_path)

  if tracklet_path.parent.name != TRACKLET_DIRNAME:
    raise ValueError("Expected the tracklet file to live under a 'gt' folder.")

  sequence_dir = tracklet_path.parent.parent
  frame_dir = sequence_dir / IMAGE_DIRNAME
  if not frame_dir.exists():
    raise FileNotFoundError(f"Missing frame directory: {frame_dir}")

  return TrackletSpec(
      category=sequence_dir.parent.name,
      sequence_id=sequence_dir.name,
      object_id=object_id,
      start_frame=start_frame,
      end_frame=end_frame,
      frame_dir=frame_dir,
      tracklet_file=tracklet_path,
  )


def load_tracklet_boxes(tracklet_file: str | Path) -> np.ndarray:
  """Loads one tracklet as [num_frames, 4] float32 xywh boxes."""
  boxes = np.loadtxt(_as_path(tracklet_file), delimiter=",", dtype=np.float32)
  boxes = np.atleast_2d(boxes)
  if boxes.shape[1] != 4:
    raise ValueError(f"Expected 4 columns in tracklet file, got {boxes.shape}.")
  return boxes


def frame_index_to_path(frame_dir: str | Path, frame_index: int) -> Path:
  """Returns the absolute path for a 1-based VISO frame index."""
  return _as_path(frame_dir) / f"{frame_index:06d}{IMAGE_SUFFIX}"


def load_frame(frame_path: str | Path) -> np.ndarray:
  """Loads an RGB frame as uint8 numpy array."""
  with Image.open(_as_path(frame_path)) as image:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def load_tracklet_frames(
    tracklet: TrackletSpec, start_offset: int = 0, length: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
  """Loads a contiguous span of frames and returns frames plus indices.

  Args:
    tracklet: Tracklet descriptor created by build_tracklet_spec.
    start_offset: Zero-based offset inside the tracklet.
    length: Number of frames to load. When omitted, loads to the end.
  """
  boxes = load_tracklet_boxes(tracklet.tracklet_file)
  if start_offset < 0 or start_offset >= len(boxes):
    raise IndexError("start_offset is outside the tracklet range.")

  stop_offset = len(boxes) if length is None else min(len(boxes), start_offset + length)
  frame_indices = np.arange(
      tracklet.start_frame + start_offset,
      tracklet.start_frame + stop_offset,
      dtype=np.int32,
  )
  frames = [load_frame(frame_index_to_path(tracklet.frame_dir, int(index))) for index in frame_indices]
  return np.stack(frames, axis=0), frame_indices


def get_frame_chunk(
    tracklet: TrackletSpec, start_offset: int, chunk_size: int
) -> FrameChunk:
  """Loads one frame chunk together with its aligned boxes."""
  if chunk_size <= 0:
    raise ValueError("chunk_size must be positive.")
  boxes = load_tracklet_boxes(tracklet.tracklet_file)
  stop_offset = min(len(boxes), start_offset + chunk_size)
  frames, frame_indices = load_tracklet_frames(
      tracklet=tracklet, start_offset=start_offset, length=chunk_size
  )
  return FrameChunk(
      frame_indices=frame_indices,
      frames=frames,
      boxes_xywh=boxes[start_offset:stop_offset],
  )


def iter_frame_chunks(
    tracklet: TrackletSpec, chunk_size: int, stride: Optional[int] = None
) -> Iterator[FrameChunk]:
  """Yields consecutive chunks over a tracklet."""
  boxes = load_tracklet_boxes(tracklet.tracklet_file)
  if chunk_size <= 0:
    raise ValueError("chunk_size must be positive.")
  effective_stride = chunk_size if stride is None else stride
  if effective_stride <= 0:
    raise ValueError("stride must be positive.")

  for start_offset in range(0, len(boxes), effective_stride):
    yield get_frame_chunk(tracklet, start_offset=start_offset, chunk_size=chunk_size)


def xywh_to_xyxy(box_xywh: Sequence[float]) -> np.ndarray:
  """Converts [x, y, w, h] to [x0, y0, x1, y1]."""
  x, y, width, height = np.asarray(box_xywh, dtype=np.float32)
  return np.array([x, y, x + width, y + height], dtype=np.float32)


def xyxy_to_xywh(box_xyxy: Sequence[float]) -> np.ndarray:
  """Converts [x0, y0, x1, y1] to [x, y, w, h]."""
  x0, y0, x1, y1 = np.asarray(box_xyxy, dtype=np.float32)
  return np.array([x0, y0, x1 - x0, y1 - y0], dtype=np.float32)


def pixel_xy_to_tapir_yx(points_xy: np.ndarray) -> np.ndarray:
  """Converts absolute pixel [x, y] points to TAPIR-style [y, x]."""
  points_xy = np.asarray(points_xy, dtype=np.float32)
  return points_xy[..., ::-1]


def tapir_yx_to_pixel_xy(points_yx: np.ndarray) -> np.ndarray:
  """Converts TAPIR-style [y, x] points to absolute pixel [x, y]."""
  points_yx = np.asarray(points_yx, dtype=np.float32)
  return points_yx[..., ::-1]


def pixel_xy_to_normalized_yx(
    points_xy: np.ndarray, image_size_xy: Sequence[int]
) -> np.ndarray:
  """Converts absolute pixel [x, y] points to normalized [y, x]."""
  width, height = image_size_xy
  points_xy = np.asarray(points_xy, dtype=np.float32)
  points_yx = pixel_xy_to_tapir_yx(points_xy)
  scale = np.array([height, width], dtype=np.float32)
  return points_yx / scale


def normalized_yx_to_pixel_xy(
    points_yx: np.ndarray, image_size_xy: Sequence[int]
) -> np.ndarray:
  """Converts normalized [y, x] points to absolute pixel [x, y]."""
  width, height = image_size_xy
  points_yx = np.asarray(points_yx, dtype=np.float32)
  scale = np.array([height, width], dtype=np.float32)
  return tapir_yx_to_pixel_xy(points_yx * scale)


def compute_crop_window(
    box_xywh: Sequence[float],
    image_shape: Sequence[int],
    context_scale: float = 2.0,
    min_crop_size: int = 64,
) -> CropWindow:
  """Builds a square crop around a small target with extra context."""
  if context_scale < 1.0:
    raise ValueError("context_scale must be at least 1.0.")

  height, width = image_shape[:2]
  x, y, box_w, box_h = np.asarray(box_xywh, dtype=np.float32)
  center_x = x + box_w / 2.0
  center_y = y + box_h / 2.0
  crop_size = max(box_w, box_h) * context_scale
  crop_size = max(float(min_crop_size), crop_size)
  half = crop_size / 2.0

  x0 = int(np.floor(center_x - half))
  y0 = int(np.floor(center_y - half))
  x1 = int(np.ceil(center_x + half))
  y1 = int(np.ceil(center_y + half))

  x0 = max(0, x0)
  y0 = max(0, y0)
  x1 = min(width, x1)
  y1 = min(height, y1)

  if x1 <= x0:
    x1 = min(width, x0 + 1)
  if y1 <= y0:
    y1 = min(height, y0 + 1)

  return CropWindow(x0=x0, y0=y0, x1=x1, y1=y1, source_size_xy=(width, height))


def crop_and_resize(
    frame: np.ndarray,
    crop_window: CropWindow,
    output_size_xy: Sequence[int],
    resample: int = Image.Resampling.BICUBIC,
) -> np.ndarray:
  """Crops a frame and upsamples it for small-object segmentation."""
  crop = frame[crop_window.y0:crop_window.y1, crop_window.x0:crop_window.x1]
  target_width, target_height = output_size_xy
  resized = Image.fromarray(crop).resize((target_width, target_height), resample=resample)
  return np.array(resized, dtype=np.uint8)


def map_points_to_crop(
    points_xy: np.ndarray,
    crop_window: CropWindow,
    output_size_xy: Sequence[int],
) -> np.ndarray:
  """Maps source-image points into crop pixel coordinates."""
  points_xy = np.asarray(points_xy, dtype=np.float32)
  scale_x = output_size_xy[0] / crop_window.width
  scale_y = output_size_xy[1] / crop_window.height
  translated = points_xy - np.array([crop_window.x0, crop_window.y0], dtype=np.float32)
  return translated * np.array([scale_x, scale_y], dtype=np.float32)


def map_points_from_crop(
    points_xy: np.ndarray,
    crop_window: CropWindow,
    output_size_xy: Sequence[int],
) -> np.ndarray:
  """Maps crop-space points back into source-image pixel coordinates."""
  points_xy = np.asarray(points_xy, dtype=np.float32)
  scale_x = crop_window.width / output_size_xy[0]
  scale_y = crop_window.height / output_size_xy[1]
  return points_xy * np.array([scale_x, scale_y], dtype=np.float32) + np.array(
      [crop_window.x0, crop_window.y0], dtype=np.float32
  )


def map_box_to_crop(
    box_xywh: Sequence[float],
    crop_window: CropWindow,
    output_size_xy: Sequence[int],
) -> np.ndarray:
  """Maps a source-image xywh box into crop pixel coordinates."""
  corners = xywh_to_xyxy(box_xywh)
  crop_corners = map_points_to_crop(
      np.array([[corners[0], corners[1]], [corners[2], corners[3]]], dtype=np.float32),
      crop_window=crop_window,
      output_size_xy=output_size_xy,
  )
  return xyxy_to_xywh(
      np.array(
          [
              crop_corners[0, 0],
              crop_corners[0, 1],
              crop_corners[1, 0],
              crop_corners[1, 1],
          ],
          dtype=np.float32,
      )
  )