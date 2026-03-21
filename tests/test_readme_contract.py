import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def extract_section(readme_text: str, heading: str) -> str:
    pattern = rf"^## {re.escape(heading)}\n(?P<body>.*?)(?=^## |\Z)"
    match = re.search(pattern, readme_text, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        raise AssertionError(f"missing README section: {heading}")
    return match.group("body")


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


if __name__ == "__main__":
    unittest.main()
