import sys
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.runtime_patches import _ANALYZER_MODULE, apply_noalign


class _Engine:
    def __init__(self, model):
        self.t3 = model


class _CurrentSampler:
    def inference(self):
        return "cfg logits"

    def inference_turbo(self):
        return "turbo logits"


class _StillCallsAnalyzer:
    def inference(self):
        return AlignmentStreamAnalyzer


class NoAlignTests(unittest.TestCase):
    def test_current_tree_has_no_analyzer_module(self):
        module_path = REPO / "src" / "chatterbox_" / "models" / "t3" / "inference" / "alignment_stream_analyzer.py"
        source = (REPO / "src" / "chatterbox_" / "models" / "t3" / "t3.py").read_text(encoding="utf-8")
        self.assertFalse(module_path.exists())
        self.assertNotIn("AlignmentStreamAnalyzer", source)
        status = apply_noalign(_Engine(_CurrentSampler()))
        self.assertIn("does not apply an alignment stream", status)

    def test_missing_module_still_referenced_by_sampler_raises(self):
        sys.modules.pop(_ANALYZER_MODULE, None)
        with self.assertRaises(RuntimeError) as caught:
            apply_noalign(_Engine(_StillCallsAnalyzer()))
        self.assertIn("still references", str(caught.exception))

    def test_importable_analyzer_step_returns_logits(self):
        class AlignmentStreamAnalyzer:
            def step(self, logits, next_token=None):
                raise AssertionError("alignment edited logits")

        module = types.ModuleType(_ANALYZER_MODULE)
        module.AlignmentStreamAnalyzer = AlignmentStreamAnalyzer
        sys.modules[_ANALYZER_MODULE] = module
        try:
            status = apply_noalign(_Engine(_CurrentSampler()))
            logits = object()
            self.assertIs(AlignmentStreamAnalyzer.step(None, logits), logits)
            self.assertIn("patched AlignmentStreamAnalyzer.step", status)
        finally:
            sys.modules.pop(_ANALYZER_MODULE, None)

    def test_renderers_use_the_helper(self):
        for name in ("render_ft.py", "render_book.py"):
            text = (REPO / name).read_text(encoding="utf-8")
            self.assertIn("apply_noalign", text)
            self.assertNotIn("alignment_stream_analyzer import", text)


if __name__ == "__main__":
    unittest.main()
