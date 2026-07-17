import sys
import types
import unittest
from pathlib import Path


prefect_stub = types.ModuleType("prefect")

def task(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


def get_run_logger():
    return None


prefect_stub.task = task
prefect_stub.get_run_logger = get_run_logger
sys.modules.setdefault("prefect", prefect_stub)


data_validation_stub = types.ModuleType("data_validation")
data_validation_stub.get_run = lambda *args, **kwargs: None
sys.modules.setdefault("data_validation", data_validation_stub)


import linker  # noqa: E402


class MakeRelativePathTests(unittest.TestCase):
    def test_relative_path_is_unchanged(self):
        self.assertEqual(linker.make_relative_path("experiments"), Path("experiments"))

    def test_absolute_path_becomes_relative(self):
        self.assertEqual(linker.make_relative_path("/experiments"), Path("experiments"))

    def test_root_path_becomes_current_directory(self):
        self.assertEqual(linker.make_relative_path("/"), Path("."))

    def test_joined_path_never_overrides_proposal_path(self):
        proposal = Path("/nsls2/data/cms/proposals/2026-1/pass-12345")
        self.assertEqual(
            proposal / linker.make_relative_path("/experiments"),
            proposal / "experiments",
        )


if __name__ == "__main__":
    unittest.main()
