"""Pure argument and provenance checks for vector tabletop initialization."""


def add_protected_mimic_arguments(parser):
    parser.add_argument('--protected-mimic-resets', action='store_true',
                        help='terminate bounded mimic failures and validate local resets during training')
    parser.add_argument('--allow-protected-mimic-init', action='store_true',
                        help='explicit weights-only migration to validated mimic reset handling')


def validate_protected_mimic_options(*, physics_hz, mimic_frequency,
        protected_mimic_resets=False, allow_protected_mimic_init=False,
        initialize_from=None, resume=None, allow_physics_refinement=False,
        allow_mimic_refinement=False):
    if protected_mimic_resets and (physics_hz != 480 or mimic_frequency != 200):
        raise ValueError('protected mimic resets require 480 Hz physics and 200 s^-1 coupling')
    if allow_protected_mimic_init:
        if not protected_mimic_resets or not initialize_from or resume:
            raise ValueError('protected initialization requires protected resets and initialize-from, without resume')
        if allow_physics_refinement or allow_mimic_refinement:
            raise ValueError('protected initialization cannot combine physics or mimic refinement')
    elif protected_mimic_resets and initialize_from:
        raise ValueError('protected initialization requires explicit allow-protected-mimic-init')


def add_mimic_arguments(parser):
    """Expose diagnosed compliance frequencies and explicit rigid identification."""
    parser.add_argument('--mimic-frequency', type=int, choices=(0, 200, 400), default=200,
                        help='mimic frequency in s^-1; 400 requires 480 Hz; 0 requires 480 Hz and an explicit load profile')
    parser.add_argument('--allow-mimic-refinement', action='store_true',
                        help='allow a validated 200-to-400 s^-1 weights-only transfer at 480 Hz')


def validate_mimic_options(*, physics_hz, mimic_frequency, initialize_from=None,
                           resume=None, allow_mimic_refinement=False,
                           allow_physics_refinement=False, allow_rigid=False):
    """Reject unsupported settings before launching or writing run artifacts.

    Checkpoint provenance is checked separately by the training runner. A
    same-contract 400 s^-1 resume does not need a refinement flag.
    """
    if mimic_frequency == 0 and allow_rigid:
        if physics_hz != 480 or initialize_from or allow_mimic_refinement or allow_physics_refinement:
            raise ValueError('rigid profile requires 480 Hz and fresh training or exact resume')
        return
    if mimic_frequency not in (200, 400):
        raise ValueError('mimic frequency must be 200 or 400 s^-1')
    if mimic_frequency == 400 and physics_hz != 480:
        raise ValueError('400 s^-1 mimic frequency requires 480 Hz physics')
    if allow_mimic_refinement and allow_physics_refinement:
        raise ValueError('mimic and physics refinement are mutually exclusive')
    if allow_mimic_refinement:
        if not initialize_from or resume:
            raise ValueError('mimic refinement requires initialize-from and cannot resume')
        if mimic_frequency != 400:
            raise ValueError('mimic refinement requires target frequency 400 s^-1')


def validate_seed_provenance(source, requested) -> dict:
    """Match seed assets and physics, allowing only a 240 -> 480 Hz refinement.

    ``source`` may be the complete historical training contract; ``requested``
    contains the current seed provenance fields. Extra historical training
    fields do not describe the saved physical state and are not compared.
    Neither input nor the seed's saved state is modified.
    """
    required = {'pose_sha256', 'source_sha256', 'physics_hz', 'control_hz',
                'solver_type', 'position_iterations'}
    if not required.issubset(source) or not required.issubset(requested):
        raise ValueError('seed provenance is missing required assets/pose/physics fields')
    mismatch = {key for key, value in requested.items()
                if key not in source or source[key] != value}
    if not mismatch:
        return {}
    if (mismatch == {'physics_hz'} and source['physics_hz'] == 240
            and requested['physics_hz'] == 480):
        return {'seed_physics_hz': 240}
    raise ValueError(f'seed assets/pose/physics provenance mismatch: {sorted(mismatch)}')
