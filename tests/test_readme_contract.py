from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from inferoscope.extraction import (
    MoELayerCaptureInput,
    build_layer_grid_layout,
    build_manifest,
    build_token_complete_event,
    write_run_bundle,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
README_EXAMPLE_TIMEOUT_SECONDS = 10


def extract_section(readme_text: str, heading: str) -> str:
    pattern = rf"^## {re.escape(heading)}\n(?P<body>.*?)(?=^## |\Z)"
    match = re.search(pattern, readme_text, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        raise AssertionError(f"missing README section: {heading}")
    return match.group("body")


def extract_first_python_block(section_text: str) -> str:
    match = re.search(r"```python\n(?P<code>.*?)```", section_text, flags=re.DOTALL)
    if match is None:
        raise AssertionError("missing python code block in README section")
    return match.group("code")


def run_quick_example_subprocess() -> subprocess.CompletedProcess[str]:
    quick_example = extract_section((REPO_ROOT / "README.md").read_text(), "Quick Example")
    script = extract_first_python_block(quick_example)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    with tempfile.TemporaryDirectory() as tmpdir:
        return subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmpdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=README_EXAMPLE_TIMEOUT_SECONDS,
        )


def run_documented_inspect_command_subprocess(*extra_args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = build_manifest(
            run_id="run-001",
            created_at="2026-03-23T12:00:00Z",
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
            token_text=" hello",
            context_length=5,
            decode_start_ms=10.0,
            decode_end_ms=20.0,
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

        return subprocess.run(
            [sys.executable, "-m", "inferoscope.inspect", *extra_args, str(run_dir)],
            cwd=tmpdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=README_EXAMPLE_TIMEOUT_SECONDS,
        )


class ReadmeContractTests(unittest.TestCase):
    def test_quick_example_uses_a_rerunnable_temporary_bundle_root(self) -> None:
        quick_example = extract_section((REPO_ROOT / "README.md").read_text(), "Quick Example")

        self.assertIn("TemporaryDirectory", quick_example)
        self.assertNotIn('Path("runs")', quick_example)
        self.assertNotIn("runs/demo-run/", quick_example)

    def test_quick_example_explains_how_to_import_from_a_checkout(self) -> None:
        quick_example = extract_section((REPO_ROOT / "README.md").read_text(), "Quick Example")

        self.assertTrue(
            "PYTHONPATH" in quick_example or "repo root" in quick_example.lower(),
            msg="Quick Example should explain how to import inferoscope from a checkout",
        )

    def test_inspecting_bundles_explains_how_to_import_from_a_checkout(self) -> None:
        inspect_section = extract_section((REPO_ROOT / "README.md").read_text(), "Inspecting Bundles")

        self.assertTrue(
            "PYTHONPATH" in inspect_section or "repo root" in inspect_section.lower(),
            msg="Inspecting Bundles should explain how to run inferoscope.inspect from a checkout",
        )

    def test_readme_documents_olmoe_recorder_bridge_surface(self) -> None:
        readme_text = (REPO_ROOT / "README.md").read_text()

        self.assertIn(
            "PyTorchRunBundleRecorder",
            readme_text,
            msg="README should document the public PyTorchRunBundleRecorder API",
        )
        self.assertIn(
            "record_olmoe_generated_token",
            readme_text,
            msg="README should document the public record_olmoe_generated_token bridge API",
        )

    def test_readme_documents_inspect_cli_surface(self) -> None:
        readme_text = (REPO_ROOT / "README.md").read_text()

        self.assertIn(
            "python -m inferoscope.inspect",
            readme_text,
            msg="README should document the public inspect CLI command",
        )
        self.assertIn(
            "--json",
            readme_text,
            msg="README should document the inspect CLI JSON mode",
        )

    def test_quick_example_python_block_runs_successfully(self) -> None:
        result = run_quick_example_subprocess()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertTrue(result.stdout.strip(), msg="README example should print its output")
        self.assertIn("[0, 1]", result.stdout)

    def test_documented_inspect_command_runs_successfully(self) -> None:
        result = run_documented_inspect_command_subprocess()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("run_id: run-001", result.stdout)

    def test_documented_inspect_json_command_runs_successfully(self) -> None:
        result = run_documented_inspect_command_subprocess("--json")

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('"run_id": "run-001"', result.stdout)

    def test_quick_example_python_block_uses_subprocess_timeout(self) -> None:
        with patch.object(
            subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                args=[sys.executable, "-c", "print('ok')"],
                returncode=0,
                stdout="/tmp/demo-run\n[0, 1]\n",
                stderr="",
            ),
        ) as run_mock:
            run_quick_example_subprocess()

        timeout = run_mock.call_args.kwargs.get("timeout")
        self.assertIsNotNone(timeout, "README quick example subprocess should set a timeout")
        self.assertGreater(timeout, 0)


if __name__ == "__main__":
    unittest.main()
