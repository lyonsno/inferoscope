import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from inferoscope.extraction import (
    MoELayerCaptureInput,
    build_layer_grid_layout,
    build_manifest,
    build_token_complete_event,
    write_run_bundle,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
INSPECT_CLI_TIMEOUT_SECONDS = 10


def build_valid_run_bundle(
    root_dir: str | Path,
    *,
    raw_events: list[dict] | None = None,
    layout: dict | None = None,
    include_optional_artifacts: bool = False,
    prompt_text: str = "hello",
) -> Path:
    manifest = build_manifest(
        run_id="run-001",
        created_at="2026-03-22T12:00:00Z",
        model_id="allenai/OLMoE-1B-7B-0125",
        tokenizer_id="allenai/OLMoE-1B-7B-0125",
        prompt_text=prompt_text,
        derivation_version="motifs/v0.1.0-alpha",
        derivation_config_id="motifs/default-alpha",
        generation_config={"max_new_tokens": 4},
    )
    if raw_events is None:
        raw_events = [
            build_token_complete_event(
                run_id="run-001",
                token_index=0,
                token_id=42,
                token_text=" hello",
                context_length=5,
                decode_start_ms=10.0,
                decode_end_ms=20.0,
                layer_inputs=[
                    MoELayerCaptureInput(
                        layer_index=0,
                        router_logits=[2.0, 1.0, 0.0, -1.0],
                        num_active_experts=2,
                    ),
                    MoELayerCaptureInput(
                        layer_index=1,
                        router_logits=[1.0, 0.0],
                        num_active_experts=1,
                    ),
                ],
            )
        ]

    if layout is None:
        layout = build_layer_grid_layout(
            run_id="run-001",
            layout_id="default-grid",
            layer_expert_counts=[(0, 4), (1, 2)],
        )

    derived_events = None
    motif_ledger = None
    contingency = None
    if include_optional_artifacts:
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

    return write_run_bundle(
        root_dir,
        manifest,
        raw_events,
        layout,
        derived_events=derived_events,
        motif_ledger=motif_ledger,
        contingency=contingency,
    )


class InspectCliTests(unittest.TestCase):
    def run_inspect(
        self,
        *args: str,
        cwd: str | Path,
        env_overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)
        if env_overrides is not None:
            env.update(env_overrides)
        return subprocess.run(
            [sys.executable, "-m", "inferoscope.inspect", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=INSPECT_CLI_TIMEOUT_SECONDS,
        )

    def test_inspect_cli_prints_bundle_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = build_valid_run_bundle(tmpdir)
            result = self.run_inspect(str(run_dir), cwd=tmpdir)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("run_id: run-001", result.stdout)
            self.assertIn("created_at: 2026-03-22T12:00:00Z", result.stdout)
            self.assertIn("model_id: allenai/OLMoE-1B-7B-0125", result.stdout)
            self.assertIn("tokenizer_id: allenai/OLMoE-1B-7B-0125", result.stdout)
            self.assertIn("raw_events: 1", result.stdout)
            self.assertIn("layers: 2", result.stdout)
            self.assertIn("layer_expert_counts: 0=4, 1=2", result.stdout)
            self.assertIn("optional_artifacts: derived=no motif_ledger=no contingency=no", result.stdout)

    def test_inspect_cli_requires_run_dir_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_inspect(cwd=tmpdir)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("usage:", result.stderr.lower())

    def test_inspect_cli_reports_bundle_load_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_inspect(str(Path(tmpdir) / "missing-run"), cwd=tmpdir)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("manifest.json", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_inspect_cli_reports_schema_validation_errors_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = build_valid_run_bundle(tmpdir)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["schema_version"]
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

            result = self.run_inspect(str(run_dir), cwd=tmpdir)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("schema validation", result.stderr)
            self.assertIn("manifest", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_inspect_cli_reports_semantic_validation_errors_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = build_valid_run_bundle(tmpdir)
            layout_path = run_dir / "layout.json"
            layout = json.loads(layout_path.read_text(encoding="utf-8"))
            layout["run_id"] = "other-run"
            layout_path.write_text(json.dumps(layout) + "\n", encoding="utf-8")

            result = self.run_inspect(str(run_dir), cwd=tmpdir)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("semantic validation", result.stderr)
            self.assertIn("run_id_mismatch", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_inspect_cli_reports_optional_artifacts_presence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = build_valid_run_bundle(tmpdir, include_optional_artifacts=True)
            result = self.run_inspect(str(run_dir), cwd=tmpdir)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("optional_artifacts: derived=yes motif_ledger=yes contingency=yes", result.stdout)

    def test_inspect_cli_sorts_layer_expert_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = build_valid_run_bundle(tmpdir)
            layout_path = run_dir / "layout.json"
            layout = json.loads(layout_path.read_text(encoding="utf-8"))
            layout["layers"] = list(reversed(layout["layers"]))
            layout_path.write_text(json.dumps(layout) + "\n", encoding="utf-8")

            result = self.run_inspect(str(run_dir), cwd=tmpdir)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("layer_expert_counts: 0=4, 1=2", result.stdout)

    def test_inspect_cli_json_outputs_machine_readable_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = build_valid_run_bundle(tmpdir, include_optional_artifacts=True)
            result = self.run_inspect("--json", str(run_dir), cwd=tmpdir)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["run_id"], "run-001")
            self.assertEqual(summary["created_at"], "2026-03-22T12:00:00Z")
            self.assertEqual(summary["model_id"], "allenai/OLMoE-1B-7B-0125")
            self.assertEqual(summary["tokenizer_id"], "allenai/OLMoE-1B-7B-0125")
            self.assertEqual(summary["prompt_text"], "hello")
            self.assertEqual(summary["raw_events"], 1)
            self.assertEqual(summary["layers"], 2)
            self.assertEqual(
                summary["layer_expert_counts"],
                [
                    {"layer_index": 0, "num_total_experts": 4},
                    {"layer_index": 1, "num_total_experts": 2},
                ],
            )
            self.assertEqual(
                summary["optional_artifacts"],
                {"derived": True, "motif_ledger": True, "contingency": True},
            )
            self.assertEqual(
                summary["decode_duration_ms"],
                {"min": 10.0, "max": 10.0, "avg": 10.0},
            )

    def test_inspect_cli_human_summary_escapes_multiline_prompt_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = build_valid_run_bundle(tmpdir, prompt_text="line1\nline2")
            result = self.run_inspect(str(run_dir), cwd=tmpdir)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn('prompt_text: "line1\\nline2"', result.stdout)
            self.assertNotIn("prompt_text: line1\nline2", result.stdout)

    def test_inspect_cli_human_summary_uses_ascii_safe_prompt_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = build_valid_run_bundle(tmpdir, prompt_text="cafe\u00e9")
            result = self.run_inspect(
                str(run_dir),
                cwd=tmpdir,
                env_overrides={"LC_ALL": "C", "PYTHONUTF8": "0"},
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn('prompt_text: "cafe\\u00e9"', result.stdout)

    def test_inspect_cli_handles_empty_raw_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            layout = build_layer_grid_layout(
                run_id="run-001",
                layout_id="default-grid",
                layer_expert_counts=[(0, 4)],
            )
            run_dir = build_valid_run_bundle(tmpdir, raw_events=[], layout=layout)
            result = self.run_inspect(str(run_dir), cwd=tmpdir)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("raw_events: 0", result.stdout)
            self.assertIn("layers: 1", result.stdout)


if __name__ == "__main__":
    unittest.main()
