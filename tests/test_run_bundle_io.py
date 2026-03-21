import json
import math
import tempfile
import unittest
from pathlib import Path

from inferoscope.extraction import (
    MoELayerCaptureInput,
    build_layer_grid_layout,
    build_manifest,
    build_token_complete_event,
    load_run_bundle,
    write_run_bundle,
)


def make_valid_manifest(run_id: str = "run-001") -> dict:
    return build_manifest(
        run_id=run_id,
        created_at="2026-03-20T12:00:00Z",
        model_id="allenai/OLMoE-1B-7B-0125",
        tokenizer_id="allenai/OLMoE-1B-7B-0125",
        prompt_text="hello",
        derivation_version="motifs/v0.1.0-alpha",
        derivation_config_id="motifs/default-alpha",
        seed=7,
        generation_config={"max_new_tokens": 16},
        capture_config={"trace_level": "full-router-probs"},
    )


def make_valid_raw_event(run_id: str = "run-001") -> dict:
    return build_token_complete_event(
        run_id=run_id,
        token_index=0,
        token_id=42,
        token_text="hello",
        context_length=5,
        decode_start_ms=10.0,
        decode_end_ms=25.0,
        layer_inputs=[
            MoELayerCaptureInput(
                layer_index=0,
                router_logits=[2.0, 1.0, 0.0, -1.0],
                num_active_experts=2,
            )
        ],
    )


def make_valid_layout(run_id: str = "run-001") -> dict:
    return build_layer_grid_layout(
        run_id=run_id,
        layout_id="default-grid",
        layer_expert_counts=[(0, 4)],
    )


class WriteRunBundleTests(unittest.TestCase):
    def test_write_and_load_run_bundle_supports_namespaced_run_ids(self) -> None:
        run_id = "group/run-001"

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = write_run_bundle(
                tmpdir,
                make_valid_manifest(run_id=run_id),
                [make_valid_raw_event(run_id=run_id)],
                make_valid_layout(run_id=run_id),
            )

            self.assertEqual(run_dir, Path(tmpdir) / "group" / "run-001")

            bundle = load_run_bundle(run_dir)

            self.assertEqual(bundle["manifest"]["run_id"], run_id)
            self.assertEqual(bundle["raw_events"][0]["run_id"], run_id)
            self.assertEqual(bundle["layout"]["run_id"], run_id)

    def test_write_run_bundle_rejects_non_canonical_run_id_strings(self) -> None:
        for run_id in ["./run-001", "foo/.", "group//run-001", "C:/tmp/evil"]:
            with self.subTest(run_id=run_id):
                with tempfile.TemporaryDirectory() as tmpdir:
                    with self.assertRaisesRegex(ValueError, "run_id"):
                        write_run_bundle(
                            tmpdir,
                            make_valid_manifest(run_id=run_id),
                            [make_valid_raw_event(run_id=run_id)],
                            make_valid_layout(run_id=run_id),
                        )

    def test_write_run_bundle_rejects_schema_invalid_manifest(self) -> None:
        manifest = make_valid_manifest()
        del manifest["schema_version"]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "schema validation"):
                write_run_bundle(
                    tmpdir,
                    manifest,
                    [make_valid_raw_event()],
                    make_valid_layout(),
                )

    def test_write_run_bundle_rejects_non_rfc3339_created_at(self) -> None:
        manifest = make_valid_manifest()
        manifest["created_at"] = "2026-03-20 12:00:00+00:00"

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "schema validation"):
                write_run_bundle(
                    tmpdir,
                    manifest,
                    [make_valid_raw_event()],
                    make_valid_layout(),
                )

    def test_write_run_bundle_rejects_invalid_rfc3339_offset_bounds(self) -> None:
        manifest = make_valid_manifest()
        manifest["created_at"] = "2026-03-20T12:00:00+00:60"

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "schema validation"):
                write_run_bundle(
                    tmpdir,
                    manifest,
                    [make_valid_raw_event()],
                    make_valid_layout(),
                )

    def test_write_run_bundle_rejects_impossible_rfc3339_leap_second(self) -> None:
        manifest = make_valid_manifest()
        manifest["created_at"] = "2026-03-20T12:34:60Z"

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "schema validation"):
                write_run_bundle(
                    tmpdir,
                    manifest,
                    [make_valid_raw_event()],
                    make_valid_layout(),
                )

    def test_write_run_bundle_rejects_non_standard_json_numbers(self) -> None:
        manifest = make_valid_manifest()
        manifest["generation_config"]["temperature"] = float("nan")

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "JSON"):
                write_run_bundle(
                    tmpdir,
                    manifest,
                    [make_valid_raw_event()],
                    make_valid_layout(),
                )

    def test_write_and_load_run_bundle_round_trip_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = write_run_bundle(
                tmpdir,
                make_valid_manifest(),
                [make_valid_raw_event()],
                make_valid_layout(),
            )

            self.assertEqual(run_dir, Path(tmpdir) / "run-001")
            self.assertTrue((run_dir / "manifest.json").is_file())
            self.assertTrue((run_dir / "raw_trace.ndjson").is_file())
            self.assertTrue((run_dir / "layout.json").is_file())
            self.assertFalse((run_dir / "derived.ndjson").exists())

            bundle = load_run_bundle(run_dir)

            self.assertEqual(bundle["manifest"]["run_id"], "run-001")
            self.assertEqual(bundle["raw_events"][0]["token_text"], "hello")
            self.assertEqual(bundle["layout"]["layout_id"], "default-grid")
            self.assertIsNone(bundle["derived_events"])
            self.assertIsNone(bundle["motif_ledger"])
            self.assertIsNone(bundle["contingency"])

    def test_write_run_bundle_rejects_semantically_invalid_bundle(self) -> None:
        manifest = make_valid_manifest()
        raw_event = make_valid_raw_event()
        layout = make_valid_layout()
        layout["run_id"] = "other-run"

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "run_id_mismatch"):
                write_run_bundle(tmpdir, manifest, [raw_event], layout)

    def test_write_run_bundle_rejects_topk_selection_mismatch(self) -> None:
        raw_event = make_valid_raw_event()
        layer = raw_event["layers"][0]
        layer["topk_indices"] = [1, 2]
        layer["topk_probs"] = [0.23688281808991013, 0.08714431874203257]
        layer["top1_prob"] = 0.6439142598879724
        layer["top1_top2_margin"] = 0.14973849934787756

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "topk_selection_mismatch"):
                write_run_bundle(
                    tmpdir,
                    make_valid_manifest(),
                    [raw_event],
                    make_valid_layout(),
                )

    def test_write_run_bundle_rejects_non_positive_num_active_experts(self) -> None:
        raw_event = make_valid_raw_event()
        layer = raw_event["layers"][0]
        layer["num_active_experts"] = 0
        layer["topk_indices"] = []
        layer["topk_probs"] = []
        layer["top1_prob"] = 0.6439142598879724
        layer["top1_top2_margin"] = 0.0

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "num_active_experts_must_be_positive"):
                write_run_bundle(
                    tmpdir,
                    make_valid_manifest(),
                    [raw_event],
                    make_valid_layout(),
                )

    def test_write_run_bundle_accepts_tied_topk_in_any_order(self) -> None:
        raw_event = make_valid_raw_event()
        layer = raw_event["layers"][0]
        layer["router_probs"] = [0.5, 0.5, 0.0, 0.0]
        layer["topk_indices"] = [1, 0]
        layer["topk_probs"] = [0.5, 0.5]
        layer["top1_prob"] = 0.5
        layer["top1_top2_margin"] = 0.0
        layer["entropy"] = math.log(2.0)
        layer["normalized_entropy"] = 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = write_run_bundle(
                tmpdir,
                make_valid_manifest(),
                [raw_event],
                make_valid_layout(),
            )

            bundle = load_run_bundle(run_dir)

            self.assertEqual(bundle["raw_events"][0]["layers"][0]["topk_indices"], [1, 0])

    def test_write_and_load_run_bundle_round_trip_optional_artifacts(self) -> None:
        derived_events = [
            {
                "event_type": "token_derived",
                "schema_version": "derived/v0.1.0-provisional",
                "run_id": "run-001",
                "token_index": 0,
                "derivation_version": "motifs/v0.1.0-alpha",
                "derivation_config_id": "motifs/default-alpha",
                "derived_payload": {},
            }
        ]
        motif_ledger = {
            "schema_version": "motif_ledger/v0.1.0-provisional",
            "run_id": "run-001",
            "derivation_version": "motifs/v0.1.0-alpha",
            "derivation_config_id": "motifs/default-alpha",
            "ledger_payload": {},
        }
        contingency = {
            "schema_version": "contingency/v0.1.0-provisional",
            "run_id": "run-001",
            "derivation_version": "motifs/v0.1.0-alpha",
            "derivation_config_id": "motifs/default-alpha",
            "contingency_payload": {},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = write_run_bundle(
                tmpdir,
                make_valid_manifest(),
                [make_valid_raw_event()],
                make_valid_layout(),
                derived_events=derived_events,
                motif_ledger=motif_ledger,
                contingency=contingency,
            )

            bundle = load_run_bundle(run_dir)

            self.assertEqual(bundle["derived_events"], derived_events)
            self.assertEqual(bundle["motif_ledger"], motif_ledger)
            self.assertEqual(bundle["contingency"], contingency)

    def test_load_run_bundle_rejects_semantically_invalid_payloads_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = write_run_bundle(
                tmpdir,
                make_valid_manifest(),
                [make_valid_raw_event()],
                make_valid_layout(),
            )

            layout_path = run_dir / "layout.json"
            layout = json.loads(layout_path.read_text())
            layout["run_id"] = "other-run"
            layout_path.write_text(json.dumps(layout, indent=2, sort_keys=True) + "\n")

            with self.assertRaisesRegex(ValueError, "run_id_mismatch"):
                load_run_bundle(run_dir)

    def test_load_run_bundle_rejects_unsafe_run_id_values_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = write_run_bundle(
                tmpdir,
                make_valid_manifest(),
                [make_valid_raw_event()],
                make_valid_layout(),
            )

            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["run_id"] = "../evil"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

            raw_trace_path = run_dir / "raw_trace.ndjson"
            raw_event = json.loads(raw_trace_path.read_text().splitlines()[0])
            raw_event["run_id"] = "../evil"
            raw_trace_path.write_text(json.dumps(raw_event, sort_keys=True) + "\n")

            layout_path = run_dir / "layout.json"
            layout = json.loads(layout_path.read_text())
            layout["run_id"] = "../evil"
            layout_path.write_text(json.dumps(layout, indent=2, sort_keys=True) + "\n")

            with self.assertRaisesRegex(ValueError, "run_id"):
                load_run_bundle(run_dir)

    def test_load_run_bundle_rejects_schema_invalid_manifest_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = write_run_bundle(
                tmpdir,
                make_valid_manifest(),
                [make_valid_raw_event()],
                make_valid_layout(),
            )

            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            del manifest["schema_version"]
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

            with self.assertRaisesRegex(ValueError, "schema validation"):
                load_run_bundle(run_dir)

    def test_load_run_bundle_rejects_non_rfc3339_created_at_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = write_run_bundle(
                tmpdir,
                make_valid_manifest(),
                [make_valid_raw_event()],
                make_valid_layout(),
            )

            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["created_at"] = "2026-03-20 12:00:00+00:00"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

            with self.assertRaisesRegex(ValueError, "schema validation"):
                load_run_bundle(run_dir)

    def test_load_run_bundle_rejects_invalid_rfc3339_offset_bounds_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = write_run_bundle(
                tmpdir,
                make_valid_manifest(),
                [make_valid_raw_event()],
                make_valid_layout(),
            )

            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["created_at"] = "2026-03-20T12:00:00+00:60"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

            with self.assertRaisesRegex(ValueError, "schema validation"):
                load_run_bundle(run_dir)

    def test_load_run_bundle_rejects_impossible_rfc3339_leap_second_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = write_run_bundle(
                tmpdir,
                make_valid_manifest(),
                [make_valid_raw_event()],
                make_valid_layout(),
            )

            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["created_at"] = "2026-03-20T12:34:60Z"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

            with self.assertRaisesRegex(ValueError, "schema validation"):
                load_run_bundle(run_dir)

    def test_load_run_bundle_rejects_windows_drive_prefixed_run_ids_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = write_run_bundle(
                tmpdir,
                make_valid_manifest(),
                [make_valid_raw_event()],
                make_valid_layout(),
            )

            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["run_id"] = "C:/tmp/evil"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

            raw_trace_path = run_dir / "raw_trace.ndjson"
            raw_event = json.loads(raw_trace_path.read_text().splitlines()[0])
            raw_event["run_id"] = "C:/tmp/evil"
            raw_trace_path.write_text(json.dumps(raw_event, sort_keys=True) + "\n")

            layout_path = run_dir / "layout.json"
            layout = json.loads(layout_path.read_text())
            layout["run_id"] = "C:/tmp/evil"
            layout_path.write_text(json.dumps(layout, indent=2, sort_keys=True) + "\n")

            with self.assertRaisesRegex(ValueError, "run_id"):
                load_run_bundle(run_dir)

    def test_load_run_bundle_rejects_non_object_ndjson_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = write_run_bundle(
                tmpdir,
                make_valid_manifest(),
                [make_valid_raw_event()],
                make_valid_layout(),
            )

            raw_trace_path = run_dir / "raw_trace.ndjson"
            raw_trace_path.write_text("[]\n")

            with self.assertRaisesRegex(ValueError, "JSON object"):
                load_run_bundle(run_dir)

    def test_load_run_bundle_rejects_non_standard_json_numbers_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = write_run_bundle(
                tmpdir,
                make_valid_manifest(),
                [make_valid_raw_event()],
                make_valid_layout(),
            )

            manifest_path = run_dir / "manifest.json"
            manifest_text = manifest_path.read_text()
            manifest_path.write_text(
                manifest_text.replace('"max_new_tokens": 16', '"max_new_tokens": 16, "temperature": NaN')
            )

            with self.assertRaisesRegex(ValueError, "JSON"):
                load_run_bundle(run_dir)

    def test_write_run_bundle_cleans_up_partial_directory_on_serialization_failure(self) -> None:
        motif_ledger = {
            "schema_version": "motif_ledger/v0.1.0-provisional",
            "run_id": "run-001",
            "derivation_version": "motifs/v0.1.0-alpha",
            "derivation_config_id": "motifs/default-alpha",
            "ledger_payload": {"bad": {1, 2, 3}},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-001"

            with self.assertRaises(TypeError):
                write_run_bundle(
                    tmpdir,
                    make_valid_manifest(),
                    [make_valid_raw_event()],
                    make_valid_layout(),
                    motif_ledger=motif_ledger,
                )

            self.assertFalse(run_dir.exists())

            rewritten_run_dir = write_run_bundle(
                tmpdir,
                make_valid_manifest(),
                [make_valid_raw_event()],
                make_valid_layout(),
            )

            self.assertEqual(rewritten_run_dir, run_dir)


if __name__ == "__main__":
    unittest.main()
