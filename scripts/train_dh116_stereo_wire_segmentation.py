"""Diagnostic-only native-instance RGB segmentation fit; never grants promotion."""
import argparse
import json
from pathlib import Path
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
from go2_pvcnn.control.dh116.tabletop_stereo_error_cli import load_capture
from go2_pvcnn.control.dh116.tabletop_stereo_wire_segmentation import (
    StereoWireSegmentation, native_instance_targets, segmentation_loss)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=80)
    parser.add_argument('--seed', type=int, default=116)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--loss-mode', choices=['weighted_bce','balanced_bce_dice'], default='weighted_bce')
    args = parser.parse_args(argv)
    if not 1 <= args.updates <= 200 or not 1 <= args.threads <= 16:
        parser.error('diagnostic fit limited to 1..200 updates and 1..16 CPU threads')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    data, bindings = [], []
    for path in args.proof:
        capture, refs, _, _, proof = load_capture(path)
        bindings.append(dict(path=str(path), sha256=file_sha256(path),
            capture_sha256=proof['capture_artifact']['sha256'], cohort_sha256=proof['cohort_sha256']))
        for frame, ref in zip(capture['frames'], refs):
            target, valid, audit = native_instance_targets(frame, ref)
            data.append((frame, target, valid, audit))
    model = StereoWireSegmentation(width=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    rng = torch.Generator().manual_seed(args.seed)
    log = []
    for step in range(args.updates):
        frame, target, valid, _ = data[step % len(data)]
        env = step % len(frame['left_rgb'])
        # Crops come from native pixel positions, never colour thresholds.
        pixels = (target[env, 0, 0].bool() if step % 3 else valid[env, 0, 0]).nonzero()
        if not len(pixels):
            raise ValueError('no qualified native instance pixels')
        y, x = pixels[int(torch.randint(len(pixels), (), generator=rng))].tolist()
        h, w = target.shape[-2:]
        size = min(96, h, w)
        y, x = max(0, min(h-size, y-size//2)), max(0, min(w-size, x-size//2))
        crop = (slice(env, env+1), slice(None), slice(y, y+size), slice(x, x+size))
        output = model(frame['left_rgb'][crop], frame['right_rgb'][crop])
        tc = target[env:env+1, :, :, y:y+size, x:x+size]
        vc = valid[env:env+1, :, :, y:y+size, x:x+size]
        loss, audit = segmentation_loss(output, tc, vc, mode=args.loss_mode)
        optimizer.zero_grad(); loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.))
        optimizer.step()
        log.append(dict(update=step+1, loss=float(loss.detach()), gradient_norm=norm, **audit))
    report = dict(mode='diagnostic_fit', scope='same_cohort_native_ID_training_fit_not_generalization',
        input_bindings=bindings, seed=args.seed, training=log,
        independent_validation_groups=0, generalization_qualified=False, deployment_qualified=False,
        occlusion_head_supervised=False, per_frame=[])
    model.eval()
    with torch.no_grad():
        for frame, target, valid, audit in data:
            logits = model(frame['left_rgb'], frame['right_rgb'])['logits']
            prediction = logits.sigmoid() >= .5
            for env, identity in enumerate(frame['identity']):
                rows = []
                for view in range(2):
                    known = valid[env, view, 0]
                    truth = target[env, view, 0].bool()
                    hit = prediction[env, view, 0] & known
                    intersection = int((hit & truth).sum()); union = int(((hit | truth) & known).sum())
                    rows.append(dict(supervised_pixels=int(known.sum()), visible_iou=intersection/union if union else None,
                        precision=intersection/int(hit.sum()) if hit.any() else None,
                        recall=intersection/int((truth & known).sum()) if (truth & known).any() else None))
                report['per_frame'].append(dict(identity=identity, frame_id=frame['frame_id'], views=rows, **audit))
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output/'segmentation_diagnostic.pt'
    torch.save(dict(model=model.state_dict(), width=8, optimizer=optimizer.state_dict(),
        contract=dict(version='dh116-wire-segmentation-diagnostic-v1', qualified_views=[False, False],
                      generalization_qualified=False, source='native_instance_id', input_bindings=bindings)), checkpoint)
    report['checkpoint'] = dict(path=str(checkpoint), sha256=file_sha256(checkpoint))
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps(dict(updates=len(log), first_loss=log[0]['loss'], last_loss=log[-1]['loss'],
        checkpoint_sha256=report['checkpoint']['sha256'], generalization_qualified=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
