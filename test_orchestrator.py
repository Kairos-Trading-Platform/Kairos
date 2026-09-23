import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import classes from your orchestrator script
from orchestrator_claude import (
    BranchCache,
    ClaudeOrchestrationSession,
    GitManager,
    TesterSubAgent,
)


class TestOrchestratorDryRun(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory simulating a local Git repository."""
        self.test_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.test_dir)
        
        # Fake a .git folder so BranchCache works without a real git repo
        (self.repo_path / ".git").mkdir()

    def tearDown(self):
        """Clean up temporary files after test run."""
        shutil.rmtree(self.test_dir)

    def test_branch_cache_persistence(self):
        """Verify sub-agent results write to .git/ and persist correctly."""
        branch_name = "orch/test-feature"
        cache = BranchCache(self.repo_path, branch_name)

        # Write fake sub-agent response to cache
        fake_diff = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-# old\n+# new"
        cache.set("round_1_diff", fake_diff)

        # Ensure cache file was created inside .git/
        cache_file = self.repo_path / ".git" / "orchestrator_cache_orch_test-feature.json"
        self.assertTrue(cache_file.exists())

        # Load cache in a new instance and check contents
        cache_new_instance = BranchCache(self.repo_path, branch_name)
        self.assertEqual(cache_new_instance.get("round_1_diff"), fake_diff)

    @patch("orchestrator_claude.anthropic.Anthropic")
    @patch.object(GitManager, "_run")
    @patch.object(TesterSubAgent, "run_tests")
    def test_full_orchestrator_flow_without_api_calls(self, mock_run_tests, mock_git_run, mock_anthropic):
        """Simulate a complete 1-round successful orchestrator run without burning tokens."""
        # 1. Setup Environment
        os.environ["ANTHROPIC_API_KEY"] = "mock_key_for_testing"

        # 2. Mock Anthropic Client Responses for Coder, Security, and Writer Sub-Agents
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Dummy responses for Coder (diff), Security (JSON pass), and Writer (docs)
        dummy_diff = "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1,4 @@\n+@app.route('/health')\n+def health():\n+    return {'status': 'ok'}"
        dummy_security_json = json.dumps({"status": "PASS", "findings": []})
        dummy_writer_docs = "### Health Endpoint Added\nAdded `/health` route returning HTTP 200 OK."

        # Configure response sequence for API calls
        mock_msg_coder = MagicMock()
        mock_msg_coder.content = [MagicMock(type="text", text=dummy_diff)]

        mock_msg_sec = MagicMock()
        mock_msg_sec.content = [MagicMock(type="text", text=dummy_security_json)]

        mock_msg_writer = MagicMock()
        mock_msg_writer.content = [MagicMock(type="text", text=dummy_writer_docs)]

        mock_client.messages.create.side_effect = [
            mock_msg_coder,
            mock_msg_sec,
            mock_msg_writer,
        ]

        # 3. Dynamic Mock for Git Commands
        target_branch = "orch/add-a-get-health-route"
        current_active_branch = ["main"]  # Stateful list to track branch state during test

        def fake_git_run(*args):
            # If checking current branch (git rev-parse --abbrev-ref HEAD)
            if args == ("rev-parse", "--abbrev-ref", "HEAD"):
                return current_active_branch[0]
            # If checking existing branches (git branch --list <branch>)
            elif args[0] == "branch" and "--list" in args:
                return ""
            # If switching or creating branch (git checkout ...)
            elif args[0] == "checkout":
                current_active_branch[0] = target_branch
                return ""
            return ""

        mock_git_run.side_effect = fake_git_run
        mock_run_tests.return_value = (True, "1 passed in 0.05s")

        # 4. Instantiate and run orchestrator
        task = "Add a GET /health route to Flask app"
        session = ClaudeOrchestrationSession(
            task=task,
            repo_context="Fake Flask repo context",
            repo_path=self.repo_path,
            branch_name=target_branch,
            max_rounds=1,
        )

        outcome = session.run()

        # 5. Assertions
        # Verify branch name was deterministically generated
        self.assertEqual(session.branch_name, target_branch)
        
        # Verify final outcome was pushed
        self.assertEqual(outcome, "pushed")

        # Verify documentation file was created in repository
        changelog_file = self.repo_path / "CHANGELOG_ORCHESTRATOR.md"
        self.assertTrue(changelog_file.exists())
        self.assertIn("Health Endpoint Added", changelog_file.read_text())

        # Verify Anthropic API was called exactly 3 times (Coder, Security, Writer)
        self.assertEqual(mock_client.messages.create.call_count, 3)


if __name__ == "__main__":
    unittest.main()