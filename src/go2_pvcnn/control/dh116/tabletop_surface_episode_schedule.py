"""Pure absolute-minimum episode goals for resumable surface style blocks."""
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class EpisodeBlock:
    index: int
    style: str
    episode_goal: int


def episode_schedule(styles: Sequence[str], *, current_min: int,
                     episodes_per_env: int, episodes_per_style: int = 128) -> list[EpisodeBlock]:
    """Keep the original style phase while skipping every completed goal.

    Goals are cumulative completed-episode minima, not additional episodes or
    iteration limits. A rollout may overshoot a goal; resuming never repeats it.
    """
    for name, value, minimum in (('current_min', current_min, 0),
                                 ('episodes_per_env', episodes_per_env, 1),
                                 ('episodes_per_style', episodes_per_style, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if (not isinstance(styles, Sequence) or isinstance(styles, (str, bytes)) or not styles
            or any(not isinstance(style, str) or not style.strip() for style in styles)
            or len(set(styles)) != len(styles)):
        raise ValueError('styles must be a nonempty sequence of distinct labels')
    if current_min >= episodes_per_env:
        return []
    return [EpisodeBlock(index, styles[index % len(styles)],
                         min((index + 1) * episodes_per_style, episodes_per_env))
            for index in range(current_min // episodes_per_style,
                               (episodes_per_env + episodes_per_style - 1) // episodes_per_style)]
