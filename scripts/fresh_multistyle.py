"""Fresh random initialization adapter with 371D qualified demonstrations."""
import hashlib, pathlib, sys
source_path=pathlib.Path('/root/hand_reward_11_3_validation/scripts/train_dh116_surface_multistyle.py')
expected_sha='c31ea7e1cfc55ade3f020042c41c5b55aac1349d015b51c162effe49215494aa'
if hashlib.sha256(source_path.read_bytes()).hexdigest()!=expected_sha: raise RuntimeError('multistyle source changed; review fresh-start adapter')
if '--resume' in sys.argv or '--student-actor' in sys.argv: raise ValueError('this launcher requires fresh random initialization')
source=source_path.read_text()
source=source.replace('source=p.add_mutually_exclusive_group(required=True)','source=p.add_mutually_exclusive_group(required=False)').replace("    command.extend(['--resume',str(checkpoint)] if checkpoint else ['--student-actor',str(args.student_actor.resolve())])","    if checkpoint:command.extend(['--resume',str(checkpoint)])\n    elif args.student_actor:raise ValueError('fresh launcher cannot load a student actor')")
exec(compile(source,str(source_path),'exec'),{'__name__':'__main__','__file__':str(source_path)})
