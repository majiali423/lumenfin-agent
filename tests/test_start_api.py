from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("lumenfin_start_api", ROOT / "start_api.py")
assert SPEC is not None and SPEC.loader is not None
START_API = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(START_API)


class StartApiTestCase(unittest.TestCase):
    def test_default_mode_is_complete_online_stack(self) -> None:
        args = START_API.build_parser().parse_args([])
        self.assertEqual(args.mode, "online")
        self.assertFalse(args.without_observability)
        self.assertFalse(args.no_build)

    def test_online_settings_require_provider_infra_and_auth(self) -> None:
        missing = START_API._missing_online_settings({})
        self.assertEqual(
            missing,
            [*START_API.REQUIRED_ONLINE_SETTINGS, "MAS_API_KEY (or MAS_API_KEY_PRINCIPALS)"],
        )

    def test_api_key_principals_can_replace_single_api_key(self) -> None:
        values = {key: "configured" for key in START_API.REQUIRED_ONLINE_SETTINGS}
        values["MAS_API_KEY_PRINCIPALS"] = '{"demo-key":{"tenant_id":"demo"}}'
        self.assertEqual(START_API._missing_online_settings(values), [])
        self.assertIsNone(START_API._validate_auth_settings(values))

    def test_compose_command_includes_observability_by_default(self) -> None:
        command = START_API._compose_command(["docker", "compose"], observability=True)
        self.assertIn(str(ROOT / "docker-compose.yml"), command)
        self.assertIn(str(ROOT / "docker-compose.observability.yml"), command)

    def test_check_only_does_not_start_services(self) -> None:
        args = START_API.build_parser().parse_args(["--check-only"])
        values = {key: "configured" for key in START_API.REQUIRED_ONLINE_SETTINGS}
        values["MAS_API_KEY"] = "secret"
        completed = type("Completed", (), {"returncode": 0})()
        with (
            patch.object(START_API, "_load_online_environment", return_value=values),
            patch.object(START_API, "_compose_prefix", return_value=["docker", "compose"]),
            patch.object(START_API, "_docker_is_ready", return_value=True),
            patch.object(START_API.subprocess, "run", return_value=completed) as run,
        ):
            self.assertEqual(START_API.start_online(args), 0)
        self.assertEqual(run.call_count, 1)
        self.assertIn("config", run.call_args.args[0])

    def test_check_only_does_not_start_docker_desktop(self) -> None:
        args = START_API.build_parser().parse_args(["--check-only"])
        values = {key: "configured" for key in START_API.REQUIRED_ONLINE_SETTINGS}
        values["MAS_API_KEY"] = "secret"
        completed = type("Completed", (), {"returncode": 0})()
        with (
            patch.object(START_API, "_load_online_environment", return_value=values),
            patch.object(START_API, "_compose_prefix", return_value=["docker", "compose"]),
            patch.object(START_API, "_docker_is_ready", return_value=False),
            patch.object(START_API, "_start_docker_desktop") as start_desktop,
            patch.object(START_API.subprocess, "run", return_value=completed),
        ):
            self.assertEqual(START_API.start_online(args), 2)
        start_desktop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
