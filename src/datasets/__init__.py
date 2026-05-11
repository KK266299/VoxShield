"""Dataset package initialization and registration."""

from ..registry import register_dataset

from .brats19 import (
    BraTS19VolumeDataset,
)


from .flare21 import (
    FLARE21VolumeDataset,
)

# Import builders so they register themselves
from .brats19 import (
    Brats19SegBuilder,
    Brats19UEBuilder,
)


from .flare21 import (
    Flare21SegBuilder,
    Flare21UEBuilder,
)

# Register dataset implementations with the unified registry
register_dataset('brats19_seg')(BraTS19VolumeDataset)
register_dataset('flare21_seg')(FLARE21VolumeDataset)

__all__ = [
    'BraTS19VolumeDataset',
    'FLARE21VolumeDataset',
]