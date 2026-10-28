import types

import pytest

from openpi.shared import lerobot_compat


def test_uses_legacy_dataset_when_available(monkeypatch):
    dataset = types.ModuleType("legacy_dataset")
    paths = []

    def fake_import(name):
        paths.append(name)
        return dataset

    monkeypatch.setattr(lerobot_compat.importlib, "import_module", fake_import)
    assert lerobot_compat.import_lerobot_dataset() is dataset
    assert paths == ["lerobot.common.datasets.lerobot_dataset"]


def test_uses_new_dataset_only_when_legacy_layout_is_missing(monkeypatch):
    dataset = types.ModuleType("new_dataset")
    paths = []

    def fake_import(name):
        paths.append(name)
        if name.startswith("lerobot.common"):
            raise ModuleNotFoundError("missing legacy layout", name="lerobot.common")
        return dataset

    monkeypatch.setattr(lerobot_compat.importlib, "import_module", fake_import)
    assert lerobot_compat.import_lerobot_dataset() is dataset
    assert paths == ["lerobot.common.datasets.lerobot_dataset", "lerobot.datasets.lerobot_dataset"]


def test_does_not_mask_missing_dependencies_in_legacy_dataset(monkeypatch):
    def fake_import(_name):
        raise ModuleNotFoundError("missing dependency", name="pyarrow")

    monkeypatch.setattr(lerobot_compat.importlib, "import_module", fake_import)
    with pytest.raises(ModuleNotFoundError, match="missing dependency"):
        lerobot_compat.import_lerobot_dataset()
