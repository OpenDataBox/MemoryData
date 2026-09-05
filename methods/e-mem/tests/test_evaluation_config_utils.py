"""Tests for evaluation config path helpers."""

from evaluation.config_utils import resolve_eval_config_path


class TestResolveEvalConfigPath:
    def test_resolve_eval_config_path_prefers_script_relative_when_present(self, tmp_path):
        script_path = tmp_path / "evaluation" / "locomo" / "eval_locomo.py"
        script_path.parent.mkdir(parents=True)
        script_path.touch()
        local_config = script_path.parent / "config.yaml"
        local_config.touch()

        resolved = resolve_eval_config_path(str(script_path), "config.yaml")
        assert resolved == str(local_config)

    def test_resolve_eval_config_path_falls_back_to_project_root(self, tmp_path):
        script_path = tmp_path / "evaluation" / "hotpotqa" / "eval_hotpotqa.py"
        script_path.parent.mkdir(parents=True)
        script_path.touch()

        resolved = resolve_eval_config_path(
            str(script_path),
            "evaluation/hotpotqa/config.yaml",
        )
        assert resolved == str(tmp_path / "evaluation" / "hotpotqa" / "config.yaml")
