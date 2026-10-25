"""Contact progress potential shared by scalar and GPU-batched tasks."""
import numpy as np

REWARD_VERSION = 'object_support_progress_v4_10n_5'


def contact_potential(forces):
    """Total measured pad load: 0..10 N maps linearly to 0..5 points."""
    return 5. * np.clip(np.sum(forces, axis=-1) / 10., 0., 1.)
