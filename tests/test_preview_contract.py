from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

class PreviewContractTests(unittest.TestCase):
    def test_frozen_inventory_rejects_changed_source(self):
        from go2_pvcnn.control.dh116.tabletop_preview_contract import freeze, verify
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ("src","scripts","configs","docs/contracts"):
                (root/name).mkdir(parents=True)
            (root/"src"/"one.py").write_text("x=1")
            (root/"scripts"/"entry.py").write_text("x=2")
            (root/"configs"/"camera.json").write_text("{}")
            kit=root/"render.kit"; kit.write_text("kit")
            contract=freeze(root,kit)
            verify(root,contract)
            (root/"src"/"one.py").write_text("x=3")
            with self.assertRaisesRegex(ValueError,"inventory"):
                verify(root,contract)

    def test_inventory_rejects_added_source(self):
        from go2_pvcnn.control.dh116.tabletop_preview_contract import freeze, verify
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ("src","scripts","configs","docs/contracts"):
                (root/name).mkdir(parents=True)
            kit=root/"render.kit"; kit.write_text("kit")
            contract=freeze(root,kit)
            (root/"scripts"/"new.py").write_text("x=1")
            with self.assertRaisesRegex(ValueError,"inventory"):
                verify(root,contract)

class LauncherTests(unittest.TestCase):
    def test_bad_contract_sha_rejected_before_any_native_launch(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            contract=Path(d)/"contract.json"
            contract.write_text("{}")
            output=Path(d)/"out"
            proc=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/"scripts/preview_dh116_physical_webrtc.py"),
                "--execute-native","--contract",str(contract),"--contract-sha256","0"*64,"--output",str(output)],
                capture_output=True,text=True)
            self.assertNotEqual(proc.returncode,0)
            self.assertIn("contract SHA differs",proc.stderr)
            self.assertFalse(output.exists())

    def test_entry_adds_repository_root_when_run_from_other_directory(self):
        import subprocess
        script=Path(__file__).resolve().parents[1]/"scripts/preview_dh116_physical_webrtc.py"
        with tempfile.TemporaryDirectory() as d:
            code="import runpy; runpy.run_path("+repr(str(script))+"); import importlib.util; assert importlib.util.find_spec("+repr("scripts.run_dh116_work_packages")+")"
            proc=subprocess.run([sys.executable,"-c",code],cwd=d,capture_output=True,text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_render_budget_counts_all_calls_including_failures(self):
        from go2_pvcnn.control.dh116.tabletop_preview_contract import RenderBudget
        calls=[]
        class Sim:
            def render(self):
                calls.append(1)
                if len(calls)==1: raise RuntimeError("render failed")
        sim=RenderBudget(Sim(), 2)
        with self.assertRaisesRegex(RuntimeError,"render failed"): sim.render()
        sim.render()
        with self.assertRaisesRegex(RuntimeError,"render budget"): sim.render()
        self.assertEqual(len(calls),2)
        self.assertEqual(sim.count,2)

    def test_render_wrapper_covers_internal_reset_and_restores_method(self):
        from go2_pvcnn.control.dh116.tabletop_preview_contract import RenderBudget
        class Sim:
            def __init__(self): self.calls=0
            def render(self): self.calls+=1
            def reset(self): self.render(); self.render()
        sim=Sim()
        budget=RenderBudget(sim,2)
        budget.install()
        sim.reset()
        self.assertEqual(budget.count,2)
        with self.assertRaisesRegex(RuntimeError,"render budget"): sim.render()
        budget.close()
        sim.render()
        self.assertEqual(sim.calls,3)

    def test_freeze_requires_native_rgb_resource_module(self):
        import runpy
        script=Path(__file__).resolve().parents[1]/"scripts/preview_dh116_physical_webrtc.py"
        module=runpy.run_path(str(script))
        self.assertIn("src/go2_pvcnn/control/dh116/tabletop_stereo_rgb_resources.py", module["REQUIRED"])
