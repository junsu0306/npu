"""Host-only regression tests. No SDK, GPU, NPU or network required."""

import importlib.util
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


calibration = load("test_calibration", "compile/calibration/generate_hwc.py")
compiler = load("test_compiler", "compile/mxq/compile_hwc.py")
evaluation = load("test_evaluation", "benchmark/final/eval_final_models.py")
runtime = load("test_runtime", "benchmark/runtime/run_npu.py")
selection = load("test_selection", "compile/onnx/remove_identity_selections.py")
verify = load("test_verify", "scripts/verify_release.py")


class PreprocessingTests(unittest.TestCase):
    def test_calibration_matches_runtime_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            for height, width in ((301, 507), (503, 299), (225, 225)):
                with self.subTest(height=height, width=width):
                    path = Path(tmp) / "test.png"
                    pixels = np.random.default_rng(7).integers(0, 256, (height, width, 3), dtype=np.uint8)
                    self.assertTrue(cv2.imwrite(str(path), pixels))
                    expected = calibration.preprocess(str(path), "timm")
                    np.testing.assert_array_equal(expected, evaluation.preprocess(path, "timm"))
                    np.testing.assert_array_equal(expected, runtime.preprocess(str(path), "timm"))
                    self.assertEqual(expected.shape, (224, 224, 3))
                    self.assertEqual(expected.dtype, np.float32)
                    self.assertTrue(expected.flags.c_contiguous)

    def test_nonfinal_preprocessing_rejected(self):
        self.assertEqual(set(calibration.PROFILES), {"timm"})
        self.assertEqual(set(runtime.PREPROCESS_PROFILES), {"timm"})

    def test_no_sdk_required_for_import(self):
        self.assertTrue(callable(evaluation.discover_models))
        with patch.object(evaluation, "qbruntime", None):
            with self.assertRaisesRegex(RuntimeError, "qbruntime"):
                evaluation.require_qbruntime()

    def test_calibration_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("preserve")
            with patch.object(sys, "argv", ["generate", "--output-dir", str(output)]):
                with self.assertRaises(FileExistsError):
                    calibration.main()
            self.assertEqual(marker.read_text(), "preserve")


class CalibrationValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profile = {
            "profile": "timm", "resize_short": 249, "interpolation_name": "bicubic",
            "mean": [0.5] * 3, "std": [0.5] * 3, "layout": "HWC",
            "shape": [224, 224, 3], "dtype": "float32", "num_tensors": 1,
        }
        self.write_profile()
        np.save(self.root / "0000.npy", np.zeros((224, 224, 3), np.float32))

    def write_profile(self):
        (self.root / "preprocess_profile.json").write_text(json.dumps(self.profile))

    def test_valid(self):
        self.assertEqual(len(compiler.validate_calibration(self.root, expected_count=1)), 1)

    def test_missing_metadata(self):
        (self.root / "preprocess_profile.json").unlink()
        with self.assertRaises(FileNotFoundError):
            compiler.validate_calibration(self.root, expected_count=1)

    def test_wrong_normalization(self):
        self.profile["mean"] = [0.485, 0.456, 0.406]
        self.write_profile()
        with self.assertRaisesRegex(ValueError, "mean"):
            compiler.validate_calibration(self.root, expected_count=1)

    def test_extra_tensor(self):
        np.save(self.root / "0001.npy", np.zeros((224, 224, 3), np.float32))
        with self.assertRaisesRegex(ValueError, "Expected 1"):
            compiler.validate_calibration(self.root, expected_count=1)

    def test_invalid_tensor(self):
        for value in (
            np.zeros((3, 224, 224), np.float32),
            np.zeros((224, 224, 3), np.float64),
            np.full((224, 224, 3), np.nan, np.float32),
        ):
            with self.subTest(shape=value.shape, dtype=value.dtype):
                np.save(self.root / "0000.npy", value)
                with self.assertRaisesRegex(ValueError, "Invalid HWC"):
                    compiler.validate_calibration(self.root, expected_count=1)

    def test_compiler_passes_only_final_preset(self):
        model_dir = self.root / "onnx"
        model_dir.mkdir()
        model = model_dir / "tiny30__reduced__nhwc_b1_npusafe.onnx"
        model.write_bytes(b"test-only")
        output = self.root / "mxq"
        sdk = types.ModuleType("qbcompiler")
        calls = []
        sdk.mxq_compile = lambda **kwargs: calls.append(kwargs)
        env = {
            "NPU_CALIB_DIR": str(self.root), "NPU_ONNX_DIR": str(model_dir),
            "NPU_MXQ_DIR": str(output), "NPU_INFERENCE_SCHEME": "global8",
            "NPU_ONNX_FILTER": "__nhwc_b1_npusafe.onnx",
        }
        with patch.dict("os.environ", env), patch.dict(sys.modules, {"qbcompiler": sdk}):
            with patch.object(compiler, "validate_calibration", return_value=[self.root / "0000.npy"]):
                with patch.object(sys, "argv", ["compile"]):
                    compiler.main()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["config_preset"], "vision_transformer")
        self.assertEqual(calls[0]["inference_scheme"], "global8")
        self.assertTrue(calls[0]["cpu_offload"])
        self.assertFalse(any(k.startswith("quantization_") for k in calls[0]))


class GraphTests(unittest.TestCase):
    def graph(self, matrix):
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["P", "X"], ["Y"], name="/slim.0/select")],
            "selector",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, matrix.shape[0], 2])],
            [numpy_helper.from_array(matrix.astype(np.float32), name="P")],
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    def test_identity_removed(self):
        model = self.graph(np.eye(3))
        stats = selection.remove_identity_selections(model, "/slim.")
        self.assertEqual(len(stats["removed"]), 1)
        onnx.checker.check_model(model)
        self.assertEqual(model.graph.output[0].name, "X")

    def test_actual_selection_preserved(self):
        for matrix in (np.eye(3)[[0, 2]], np.eye(3)[[1, 0, 2]]):
            model = self.graph(matrix)
            stats = selection.remove_identity_selections(model, "/slim.")
            self.assertEqual(stats["removed"], [])
            self.assertEqual(model.graph.node[0].op_type, "MatMul")
            onnx.checker.check_model(model)


class ReleaseTests(unittest.TestCase):
    def test_exact_twelve_model_ids(self):
        self.assertEqual(len(verify.EXPECTED_IDS), 12)
        self.assertNotIn("tiny30__patch_slimming_stage369_relaxed", verify.EXPECTED_IDS)

    def test_unsafe_manifest_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for path in ("/etc/passwd", "../outside", "x/../../outside"):
                with self.subTest(path=path):
                    with self.assertRaises(ValueError):
                        verify.safe_path(Path(tmp), path)

    def test_no_manifest_overwrite(self):
        with self.assertRaisesRegex(ValueError, "already exists"):
            verify.write_manifest(ROOT)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.record = json.loads((ROOT / verify.REPORT / "benchmark.json").read_text())
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "summary.md"

    def render(self):
        evaluation.write_summary_markdown(
            self.path, self.record["run"], self.record["models"], self.record["comparisons"]
        )
        return self.path.read_text()

    def test_no_unmeasured_isolated_success_claim(self):
        text = self.render()
        self.assertIn("**PASS**", text)
        self.assertNotIn("12/12 모델에서 재현", text)
        self.assertNotIn("특히 Tiny50은", text)

    def test_incomplete_telemetry_is_not_pass(self):
        self.record["models"][0]["npu_telemetry"]["status"] = "failed"
        self.assertIn("**CHECK**", self.render())

    def test_isolated_peak_mismatch_reported(self):
        source = json.loads((ROOT / verify.REPORT / "isolated_memory/summary.json").read_text())
        item = copy.deepcopy(source["models"][0])
        item["npu_peak_reported_mb"] = 999
        destination = self.path.parent / "isolated_memory"
        destination.mkdir()
        (destination / "summary.json").write_text(json.dumps({"models": [item]}))
        self.assertIn("독립 측정 1개 중 peak 일치 0개", self.render())


if __name__ == "__main__":
    unittest.main()
