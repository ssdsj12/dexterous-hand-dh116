import json
from pathlib import Path


def test_flexible_execution_qualifications_are_independent():
    root = Path(__file__).resolve().parents[1]
    packages = json.loads(
        (root / 'docs/contracts/dh116_work_package_registry.json').read_text(encoding='utf-8')
    )['packages']

    def prerequisites(name):
        result = set(packages[name]['dependencies'])
        for dependency in tuple(result):
            result.update(prerequisites(dependency))
        return result

    left = 'flex.f4-left-contact-native'
    right = 'flex.f4-right-contact-native'
    single = 'flex.f4-single-hand-contact-native'
    dual = 'flex.f4-bimanual-contact-native'
    physics = 'flex.f7-physics-state'
    stereo = 'flex.f7-stereo-capture'
    camera = 'flex.f7-task-camera-native'
    legacy = 'flex.f6-render-native'
    visual_training = {'flex.f5-wireseg-pretraining', 'flex.f5-wireseg-subset',
                       'flex.f5-wireseg-full'}
    modes = {f'flex.f8-{mode}' for mode in ('pinch', 'tripod', 'power')}
    assert {'flex.f3-isaac-rigid', 'flex.f4-cooked-pad-sdf'} <= prerequisites(left)
    assert left in prerequisites(right)
    assert {left, right, single} <= prerequisites(dual)
    assert dual in prerequisites(physics)
    assert {camera, physics} <= prerequisites(stereo)
    for name in {left, right, single, dual, physics, stereo, camera} | modes:
        assert prerequisites(name).isdisjoint(visual_training | {legacy})
    for mode in modes:
        assert physics in prerequisites(mode)
        assert prerequisites(mode).isdisjoint((modes - {mode}) | {camera, stereo})
        assert packages[mode]['status'] != 'passed'
    assert modes <= prerequisites('flex.f8-three-teachers')
    assert stereo in prerequisites('flex.f9-three-students')
