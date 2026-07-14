import pytest
from src import config


@pytest.fixture
def mock_paths(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    monkeypatch.setattr(config, "DATA_RAW", raw)
    monkeypatch.setattr(config, "DATA_PROCESSED", processed)
    monkeypatch.setattr(config, "RESEARCHERS_CSV", raw / "researchers.csv")
    monkeypatch.setattr(config, "PROJECTS_CSV", raw / "projects.csv")
    monkeypatch.setattr(config, "ASSIGNMENTS_CSV", raw / "project_assignments.csv")
    monkeypatch.setattr(config, "PROJECTS_CLEAN_CSV", processed / "projects_clean.csv")
    return tmp_path
