"""Episode budgets are absolute minima, independent of PPO iteration counts.

Runs with either pytest or ``PYTHONPATH=src python -m unittest discover -s tests
-p test_surface_episode_schedule.py``; the scheduler needs neither Torch nor Isaac.
"""
import unittest

from go2_pvcnn.control.dh116.tabletop_surface_episode_schedule import episode_schedule


STYLES = ('pinch_thumb_index', 'tripod_thumb_index_middle')


class EpisodeScheduleTests(unittest.TestCase):
    def test_saved_minimum_eight_reaches_2048_in_alternating_128_episode_blocks(self):
        blocks = episode_schedule(STYLES, current_min=8, episodes_per_env=2048)
        self.assertEqual(len(blocks), 16)
        self.assertEqual(blocks[0].episode_goal, 128)
        self.assertEqual(blocks[-1].episode_goal, 2048)
        self.assertEqual([block.style for block in blocks], list(STYLES) * 8)
        self.assertEqual([block.index for block in blocks], list(range(16)))
        self.assertTrue(all(b.episode_goal - a.episode_goal == 128
                            for a, b in zip(blocks, blocks[1:])))

    def test_resuming_on_or_past_a_boundary_keeps_the_original_style_phase(self):
        for saved_min in (128, 177, 255):
            with self.subTest(saved_min=saved_min):
                blocks = episode_schedule(STYLES, current_min=saved_min, episodes_per_env=2048)
                self.assertEqual((blocks[0].index, blocks[0].style, blocks[0].episode_goal),
                                 (1, STYLES[1], 256))
        blocks = episode_schedule(STYLES, current_min=300, episodes_per_env=512)
        self.assertEqual([(b.index, b.style, b.episode_goal) for b in blocks],
                         [(2, STYLES[0], 384), (3, STYLES[1], 512)])

    def test_partial_final_block_uses_exact_total_goal(self):
        blocks = episode_schedule(STYLES, current_min=8, episodes_per_env=300)
        self.assertEqual([(b.style, b.episode_goal) for b in blocks],
                         [(STYLES[0], 128), (STYLES[1], 256), (STYLES[0], 300)])
        resumed = episode_schedule(STYLES, current_min=290, episodes_per_env=300)
        self.assertEqual([(b.index, b.episode_goal) for b in resumed], [(2, 300)])

    def test_target_already_complete_starts_no_blocks(self):
        for saved_min in (2048, 2055):
            with self.subTest(saved_min=saved_min):
                self.assertEqual(episode_schedule(STYLES, current_min=saved_min,
                                                  episodes_per_env=2048), [])

    def test_small_goal_and_custom_block_size(self):
        blocks = episode_schedule(STYLES, current_min=0, episodes_per_env=7)
        self.assertEqual([(b.index, b.style, b.episode_goal) for b in blocks],
                         [(0, STYLES[0], 7)])
        blocks = episode_schedule(STYLES, current_min=11, episodes_per_env=35,
                                  episodes_per_style=10)
        self.assertEqual([(b.index, b.style, b.episode_goal) for b in blocks],
                         [(1, STYLES[1], 20), (2, STYLES[0], 30), (3, STYLES[1], 35)])

    def test_all_declared_styles_rotate_in_order(self):
        blocks = episode_schedule(('one', 'two', 'three'), current_min=0,
                                  episodes_per_env=6, episodes_per_style=1)
        self.assertEqual([b.style for b in blocks], ['one', 'two', 'three'] * 2)

    def test_invalid_episode_inputs(self):
        for field, values in (
            ('current_min', (-1, 1.5, True, None, '8')),
            ('episodes_per_env', (0, -1, 2.5, False, None, '2048')),
            ('episodes_per_style', (0, -128, 1.5, True, None, '128')),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    args = dict(current_min=8, episodes_per_env=2048, episodes_per_style=128)
                    args[field] = value
                    with self.assertRaises(ValueError):
                        episode_schedule(STYLES, **args)

    def test_invalid_style_lists(self):
        for styles in (None, 5, {'pinch': 1}, (), [], 'pinch', ('pinch', 'pinch'),
                       ('',), (' ',), ('pinch', None)):
            with self.subTest(styles=styles):
                with self.assertRaises(ValueError):
                    episode_schedule(styles, current_min=8, episodes_per_env=2048)


if __name__ == '__main__':
    unittest.main()
