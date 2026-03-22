import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_SURFACE_TIMEOUT_SECONDS = 10


class RuntimeSurfaceTests(unittest.TestCase):
    def test_import_inferoscope_extraction_does_not_require_repo_root_schema_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            copied_package = Path(tmpdir) / "inferoscope"
            shutil.copytree(REPO_ROOT / "inferoscope", copied_package)

            env = os.environ.copy()
            env["PYTHONPATH"] = tmpdir

            result = subprocess.run(
                [sys.executable, "-c", "import inferoscope.extraction; print('ok')"],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=RUNTIME_SURFACE_TIMEOUT_SECONDS,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("ok", result.stdout)

    def test_bundle_io_helpers_work_without_repo_root_schema_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            copied_package = Path(tmpdir) / "inferoscope"
            shutil.copytree(REPO_ROOT / "inferoscope", copied_package)

            script = textwrap.dedent(
                """
                import tempfile

                from inferoscope.extraction import (
                    MoELayerCaptureInput,
                    build_layer_grid_layout,
                    build_manifest,
                    build_token_complete_event,
                    load_run_bundle,
                    write_run_bundle,
                )

                with tempfile.TemporaryDirectory() as bundle_root:
                    manifest = build_manifest(
                        run_id="run-001",
                        created_at="2026-03-20T12:00:00Z",
                        model_id="allenai/OLMoE-1B-7B-0125",
                        tokenizer_id="allenai/OLMoE-1B-7B-0125",
                        prompt_text="hello",
                        derivation_version="motifs/v0.1.0-alpha",
                        derivation_config_id="motifs/default-alpha",
                    )
                    raw_event = build_token_complete_event(
                        run_id="run-001",
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
                    layout = build_layer_grid_layout(
                        run_id="run-001",
                        layout_id="default-grid",
                        layer_expert_counts=[(0, 4)],
                    )

                    run_dir = write_run_bundle(bundle_root, manifest, [raw_event], layout)
                    bundle = load_run_bundle(run_dir)

                    assert bundle["manifest"]["run_id"] == "run-001"
                    assert bundle["layout"]["layout_id"] == "default-grid"
                    print("ok")
                """
            )

            env = os.environ.copy()
            env["PYTHONPATH"] = tmpdir

            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=RUNTIME_SURFACE_TIMEOUT_SECONDS,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("ok", result.stdout)

    def test_olmoe_recorder_helpers_work_without_repo_root_schema_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            copied_package = Path(tmpdir) / "inferoscope"
            shutil.copytree(REPO_ROOT / "inferoscope", copied_package)

            script = textwrap.dedent(
                """
                import tempfile

                from inferoscope.extraction import (
                    PyTorchRunBundleRecorder,
                    load_run_bundle,
                    record_olmoe_generated_token,
                )


                class FakeTensor:
                    def __init__(self, values):
                        self._values = values

                    def detach(self):
                        return self

                    def cpu(self):
                        return self

                    def tolist(self):
                        return self._values


                recorder = PyTorchRunBundleRecorder(
                    run_id="run-olmoe-001",
                    created_at="2026-03-22T12:00:00Z",
                    model_id="allenai/OLMoE-1B-7B-0125",
                    tokenizer_id="allenai/OLMoE-1B-7B-0125",
                    prompt_text="hello",
                    derivation_version="motifs/v0.1.0-alpha",
                    derivation_config_id="motifs/default-alpha",
                    generation_config={"max_new_tokens": 4},
                    capture_config={"adapter": "olmoe"},
                )

                record_olmoe_generated_token(
                    recorder,
                    token_id=42,
                    token_text=" hello",
                    context_length=5,
                    decode_start_ms=10.0,
                    decode_end_ms=20.0,
                    router_logits_by_layer={
                        1: FakeTensor([1.0, 0.0]),
                        0: FakeTensor([2.0, 1.0, 0.0, -1.0]),
                    },
                    num_active_experts=1,
                )

                with tempfile.TemporaryDirectory() as bundle_root:
                    run_dir = recorder.write_bundle(bundle_root)
                    bundle = load_run_bundle(run_dir)

                    assert bundle["manifest"]["run_id"] == "run-olmoe-001"
                    assert [layer["layer_index"] for layer in bundle["raw_events"][0]["layers"]] == [0, 1]
                    assert bundle["raw_events"][0]["layers"][0]["topk_indices"] == [0]
                    assert bundle["raw_events"][0]["layers"][1]["topk_indices"] == [0]
                    print("ok")
                """
            )

            env = os.environ.copy()
            env["PYTHONPATH"] = tmpdir

            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=RUNTIME_SURFACE_TIMEOUT_SECONDS,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("ok", result.stdout)

    def test_bundle_io_uses_utf8_under_ascii_locale(self) -> None:
        script = textwrap.dedent(
            """
            import tempfile

            from inferoscope.extraction import (
                MoELayerCaptureInput,
                build_layer_grid_layout,
                build_manifest,
                build_token_complete_event,
                load_run_bundle,
                write_run_bundle,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                manifest = build_manifest(
                    run_id="run-001",
                    created_at="2026-03-20T12:00:00Z",
                    model_id="allenai/OLMoE-1B-7B-0125",
                    tokenizer_id="allenai/OLMoE-1B-7B-0125",
                    prompt_text="cafe\\u00e9",
                    derivation_version="motifs/v0.1.0-alpha",
                    derivation_config_id="motifs/default-alpha",
                    generation_config={"temperature": 0.7},
                )
                raw_event = build_token_complete_event(
                    run_id="run-001",
                    token_index=0,
                    token_id=42,
                    token_text="\\u00e9",
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
                layout = build_layer_grid_layout(
                    run_id="run-001",
                    layout_id="default-grid",
                    layer_expert_counts=[(0, 4)],
                )

                run_dir = write_run_bundle(tmpdir, manifest, [raw_event], layout)
                bundle = load_run_bundle(run_dir)

                assert bundle["manifest"]["prompt"]["text"] == "cafe\\u00e9"
                assert bundle["raw_events"][0]["token_text"] == "\\u00e9"
                print("ok")
            """
        )

        env = os.environ.copy()
        env["LC_ALL"] = "C"
        env["PYTHONUTF8"] = "0"
        env["PYTHONPATH"] = str(REPO_ROOT)

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=RUNTIME_SURFACE_TIMEOUT_SECONDS,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
