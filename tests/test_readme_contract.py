import os
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


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

    def test_quick_example_python_block_runs_successfully(self) -> None:
        result = run_quick_example_subprocess()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertTrue(result.stdout.strip(), msg="README example should print its output")
        self.assertIn("[0, 1]", result.stdout)

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
