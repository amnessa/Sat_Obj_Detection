"""Helpers for causal TAPIR chunk tracking on VISO sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, cast

import jax
import numpy as np
from PIL import Image

from tapnet.models import tapir_model
from tapnet.utils import model_utils


DEFAULT_CAUSAL_TAPIR_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / 'tapnet'
    / 'checkpoints'
    / 'causal_tapir_checkpoint.npy'
)


@dataclass(frozen=True)
class ChunkTrackingResult:
  """Stores TAPIR outputs for one frame chunk."""

  query_points_xy: np.ndarray
  query_points_tyx: np.ndarray
  tracks_xy: np.ndarray
  tracks_xy_inference: np.ndarray
  occlusion_logits: np.ndarray
  expected_dist_logits: np.ndarray
  visibles: np.ndarray
  inference_size_xy: tuple[int, int]

  @property
  def final_points_xy(self) -> np.ndarray:
    return self.tracks_xy[-1]


def load_tapir_checkpoint(checkpoint_path: str | Path) -> tuple[object, object]:
  """Loads the causal TAPIR numpy checkpoint."""
  checkpoint = np.load(Path(checkpoint_path), allow_pickle=True).item()
  return checkpoint['params'], checkpoint['state']


def pixel_xy_to_query_points_tyx(
    points_xy: np.ndarray,
    query_frame_index: int = 0,
) -> np.ndarray:
  """Converts pixel [x, y] points to TAPIR query points [t, y, x]."""
  points_xy = np.asarray(points_xy, dtype=np.float32)
  query_frame = np.full((points_xy.shape[0], 1), query_frame_index, dtype=np.float32)
  points_yx = points_xy[:, ::-1]
  return np.concatenate([query_frame, points_yx], axis=1)


def resize_frames(
    frames: np.ndarray,
    output_size_xy: Sequence[int],
) -> np.ndarray:
  """Resizes frames to a fixed [width, height] for cheaper inference."""
  output_width = int(output_size_xy[0])
  output_height = int(output_size_xy[1])
  resized_frames = []
  for frame in frames:
    resized = Image.fromarray(frame).resize(
        (output_width, output_height),
        resample=Image.Resampling.BILINEAR,
    )
    resized_frames.append(np.array(resized, dtype=np.uint8))
  return np.stack(resized_frames, axis=0)


def scale_points_xy(
    points_xy: np.ndarray,
    source_size_xy: Sequence[int],
    target_size_xy: Sequence[int],
) -> np.ndarray:
  """Scales [x, y] points between image grids."""
  source_width = float(source_size_xy[0])
  source_height = float(source_size_xy[1])
  target_width = float(target_size_xy[0])
  target_height = float(target_size_xy[1])
  scale = np.array(
      [target_width / source_width, target_height / source_height],
      dtype=np.float32,
  )
  return np.asarray(points_xy, dtype=np.float32) * scale


class CausalTapirChunkTracker:
  """Wraps the causal TAPIR live-demo path for chunked notebook inference."""

  def __init__(
      self,
      checkpoint_path: str | Path = DEFAULT_CAUSAL_TAPIR_CHECKPOINT,
  ):
    params, state = load_tapir_checkpoint(checkpoint_path)
    self.tapir = cast(Any, tapir_model.ParameterizedTAPIR(
        params=params,
        state=state,
        tapir_kwargs=dict(
            use_causal_conv=True,
            bilinear_interp_with_depthwise_conv=False,
        ),
    ))

    def online_model_init(frames, points):
      feature_grids = self.tapir.get_feature_grids(frames, is_training=False)
      return self.tapir.get_query_features(
          frames,
          is_training=False,
          query_points=points,
          feature_grids=feature_grids,
      )

    def online_model_predict(frames, features, causal_context):
      feature_grids = self.tapir.get_feature_grids(frames, is_training=False)
      trajectories = self.tapir.estimate_trajectories(
          frames.shape[-3:-1],
          is_training=False,
          feature_grids=feature_grids,
          query_features=features,
          query_points_in_video=None,
          query_chunk_size=64,
          causal_context=causal_context,
          get_causal_context=True,
      )
      next_causal_context = trajectories['causal_context']
      del trajectories['causal_context']
      return (
          {key: value[-1] for key, value in trajectories.items()},
          next_causal_context,
      )

    self._online_init_apply = jax.jit(online_model_init)
    self._online_predict_apply = jax.jit(online_model_predict)

  def track_chunk(
      self,
      frames: np.ndarray,
      query_points_xy: np.ndarray,
      inference_size_xy: Optional[Sequence[int]] = (512, 512),
  ) -> ChunkTrackingResult:
    """Tracks one set of first-frame points through a frame chunk."""
    frames = np.asarray(frames, dtype=np.uint8)
    query_points_xy = np.asarray(query_points_xy, dtype=np.float32)
    source_size_xy = (frames.shape[2], frames.shape[1])

    if inference_size_xy is None:
      resolved_inference_size_xy = (frames.shape[2], frames.shape[1])
      inference_frames = frames
      inference_points_xy = query_points_xy
    else:
      resolved_inference_size_xy = (
          int(inference_size_xy[0]),
          int(inference_size_xy[1]),
      )
      inference_frames = resize_frames(frames, resolved_inference_size_xy)
      inference_points_xy = scale_points_xy(
          query_points_xy,
          source_size_xy=source_size_xy,
          target_size_xy=resolved_inference_size_xy,
      )

    query_points_tyx = pixel_xy_to_query_points_tyx(inference_points_xy)
    init_frames = model_utils.preprocess_frames(inference_frames[0][None, None])
    query_features = self._online_init_apply(
        frames=init_frames,
        points=query_points_tyx[None, :],
    )

    causal_state = self.tapir.construct_initial_causal_state(
        query_points_tyx.shape[0],
        len(query_features.resolutions) - 1,
    )

    chunk_tracks = []
    chunk_occlusions = []
    chunk_expected_dist = []
    chunk_visibles = []

    for frame in inference_frames:
      prediction, causal_state = self._online_predict_apply(
          frames=model_utils.preprocess_frames(frame[None, None]),
          features=query_features,
          causal_context=causal_state,
      )
      tracks_xy = np.asarray(prediction['tracks'][0, :, 0])
      occlusion_logits = np.asarray(prediction['occlusion'][0, :, 0])
      expected_dist_logits = np.asarray(prediction['expected_dist'][0, :, 0])
      visibles = np.asarray(
          model_utils.postprocess_occlusions(
              occlusion_logits,
              expected_dist_logits,
          )
      )

      chunk_tracks.append(tracks_xy)
      chunk_occlusions.append(occlusion_logits)
      chunk_expected_dist.append(expected_dist_logits)
      chunk_visibles.append(visibles)

    tracks_xy_inference = np.stack(chunk_tracks, axis=0)
    if inference_size_xy is None:
      tracks_xy = tracks_xy_inference
    else:
      tracks_xy = scale_points_xy(
          tracks_xy_inference.reshape(-1, 2),
          source_size_xy=resolved_inference_size_xy,
          target_size_xy=source_size_xy,
      ).reshape(tracks_xy_inference.shape)

    return ChunkTrackingResult(
        query_points_xy=query_points_xy,
        query_points_tyx=query_points_tyx,
        tracks_xy=tracks_xy,
        tracks_xy_inference=tracks_xy_inference,
        occlusion_logits=np.stack(chunk_occlusions, axis=0),
        expected_dist_logits=np.stack(chunk_expected_dist, axis=0),
        visibles=np.stack(chunk_visibles, axis=0),
        inference_size_xy=resolved_inference_size_xy,
    )


def track_chunk_with_causal_tapir(
    frames: np.ndarray,
    query_points_xy: np.ndarray,
    checkpoint_path: Optional[str | Path] = None,
    inference_size_xy: Optional[Sequence[int]] = (512, 512),
) -> ChunkTrackingResult:
  """Convenience wrapper for one-off chunk tracking."""
  tracker = CausalTapirChunkTracker(
      checkpoint_path=(
          DEFAULT_CAUSAL_TAPIR_CHECKPOINT if checkpoint_path is None else checkpoint_path
      )
  )
  return tracker.track_chunk(
      frames=frames,
      query_points_xy=query_points_xy,
      inference_size_xy=inference_size_xy,
  )
