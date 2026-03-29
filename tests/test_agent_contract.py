from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parent.parent


class AgentContractTests(unittest.TestCase):
    def test_repo_declares_local_python_test_command_for_agents(self) -> None:
        agents_path = REPO_ROOT / "AGENTS.md"
        self.assertTrue(
            agents_path.exists(),
            "Repo should provide a local AGENTS.md that overrides incompatible parent interpreter rules.",
        )

        agents_text = agents_path.read_text(encoding="utf-8")
        self.assertIn(
            "python3 -m unittest discover -s tests -v",
            agents_text,
            msg="Repo-local AGENTS.md should tell agents exactly how to run the test suite here.",
        )
