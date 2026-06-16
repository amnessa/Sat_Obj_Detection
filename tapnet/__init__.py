# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Legacy API for TAP.  Prefer importing from project subfolders."""

from importlib import import_module
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
  from tapnet.models import tapir_model as tapir_model
  from tapnet.models import tapnet_model as tapnet_model
  from tapnet.robotap import tapir_clustering as tapir_clustering
  from tapnet.tapvid import evaluation_datasets as evaluation_datasets


__all__ = [
    'tapir_model',
    'tapnet_model',
    'tapir_clustering',
    'evaluation_datasets',
]

_LEGACY_MODULES = {
    'tapir_model': 'tapnet.models.tapir_model',
    'tapnet_model': 'tapnet.models.tapnet_model',
    'tapir_clustering': 'tapnet.robotap.tapir_clustering',
    'evaluation_datasets': 'tapnet.tapvid.evaluation_datasets',
}


def __getattr__(name: str) -> Any:
  try:
    module_name = _LEGACY_MODULES[name]
  except KeyError as exc:
    raise AttributeError(
        f'module {__name__!r} has no attribute {name!r}'
    ) from exc

  module = import_module(module_name)
  globals()[name] = module
  return module
