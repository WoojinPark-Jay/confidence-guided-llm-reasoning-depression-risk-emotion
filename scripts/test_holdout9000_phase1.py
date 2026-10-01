import json
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import holdout9000_final_phase1 as runner
import holdout9000_assets as assets


class Phase1Tests(unittest.TestCase):
    def test_received_checkpoint_labels(self):
        self.assertEqual(runner.check_label_order([s + "_Group" for s in runner.LABELS]), runner.LABELS)
        self.assertEqual(runner.check_label_order(runner.LABELS), runner.LABELS)
        with self.assertRaises(ValueError):
            runner.check_label_order(["Happy_Group", "Neutral_Group", "Depression_Group"])
        with self.assertRaises(ValueError):
            runner.check_label_order(["LABEL_0", "LABEL_1", "LABEL_2"])

    def test_wrong_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"wrong model")
            with self.assertRaises(ValueError):
                runner.model_files(root)

    def test_asset_extract_verify_and_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "assets.zip"
            payload = b"test only"
            expected = {"data/test.csv": hashlib.sha256(payload).hexdigest()}
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("data/test.csv", payload)
            digest = assets.file_sha256(archive)
            output = root / "output"
            assets.extract_verified_assets(archive, output, digest, expected)
            self.assertTrue(assets.verify_assets(output, expected))
            assets.extract_verified_assets(archive, output, digest, expected)
            (output / "data/test.csv").write_bytes(b"corrupted")
            self.assertFalse(assets.verify_assets(output, expected))
            with self.assertRaises(ValueError):
                assets.extract_verified_assets(archive, output, "invalid", expected)

    def test_asset_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "assets.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape", b"bad")
            with self.assertRaises(ValueError):
                assets.extract_verified_assets(archive, root / "output", assets.file_sha256(archive),
                                               {"../escape": hashlib.sha256(b"bad").hexdigest()})
            self.assertFalse((root / "escape").exists())

    def test_confidence_routing_and_labels(self):
        frame = pd.DataFrame({"example_id": ["a", "b", "c"]})
        rows = runner.prediction_rows(frame, [[9, 0, 0], [0, 0, 0], [0, 0, 9]])
        self.assertEqual([r["phase1_label"] for r in rows], ["Depression", "Depression", "Happy"])
        self.assertEqual([r["phase1_routed"] for r in rows], [False, True, False])
        self.assertAlmostEqual(rows[1]["phase1_confidence"], 1 / 3)
        self.assertLess(rows[0]["phase1_confidence"], rows[0]["phase1_raw_confidence"])
        with self.assertRaises(ValueError):
            runner.prediction_rows(frame, [[np.nan] * 3] * 3)

    def test_manifest_resume_and_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            runner.lock_manifest(path, {"model": "final"})
            runner.lock_manifest(path, {"model": "final"})
            with self.assertRaises(ValueError):
                runner.lock_manifest(path, {"model": "old"})

    def test_cached_batch_validation(self):
        rows = runner.prediction_rows(pd.DataFrame({"example_id": ["a"]}), [[9, 0, 0]])
        saved = {"run_fingerprint": "one", "predictions": rows}
        runner.verify_batch(saved, ["a"], "one")
        with self.assertRaises(ValueError):
            runner.verify_batch(saved, ["b"], "one")
        with self.assertRaises(ValueError):
            runner.verify_batch(saved, ["a"], "two")

    def test_export_joins_by_id_not_order(self):
        inputs = pd.DataFrame({"example_id": ["a", "b", "c"], "text": ["aa", "bb", "cc"]})
        labels = pd.DataFrame({"example_id": ["c", "a", "b"], "label_str": ["Happy", "Depression", "Neutral"]})
        raw = pd.DataFrame({"example_id": ["b", "c", "a"], "phase2_original_text": ["raw-b", "raw-c", "raw-a"]})
        predictions = pd.DataFrame(runner.prediction_rows(inputs, [[9, 0, 0], [0, 0.1, 0], [0, 0, 9]]))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = runner.export_results(output, inputs, labels, raw, predictions, {})
            final = pd.read_csv(output / "final_phase1_reasoning_input.csv", keep_default_na=False)
            self.assertEqual(summary["phase1_accuracy"], 1)
            self.assertEqual(summary["routed_rows"], 1)
            self.assertEqual(final.phase2_original_text.tolist(), ["", "raw-b", ""])
            self.assertEqual(final.target_label.tolist(), ["Depression", "Neutral", "Happy"])
            self.assertFalse(json.loads((output / "phase2_handoff.json").read_text())["phase2_executed"])

    def test_no_model_means_no_fallback_training(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                runner.resolve_paths(Path(directory))


if __name__ == "__main__":
    unittest.main()
