"""
Post-generation integration tests.
Run ONLY after the full pipeline has completed:
  pytest tests/test_dataset.py -v
"""
import pytest
import pandas as pd
from src import config


@pytest.fixture(scope="module")
def researchers():
    if not config.RESEARCHERS_CSV.exists():
        pytest.skip("researchers.csv not found — run pipeline first")
    return pd.read_csv(config.RESEARCHERS_CSV)


@pytest.fixture(scope="module")
def projects():
    if not config.PROJECTS_CSV.exists():
        pytest.skip("projects.csv not found — run pipeline first")
    return pd.read_csv(config.PROJECTS_CSV)


@pytest.fixture(scope="module")
def assignments():
    if not config.ASSIGNMENTS_CSV.exists():
        pytest.skip("project_assignments.csv not found — run pipeline first")
    return pd.read_csv(config.ASSIGNMENTS_CSV)


@pytest.fixture(scope="module")
def projects_clean():
    if not config.PROJECTS_CLEAN_CSV.exists():
        pytest.skip("projects_clean.csv not found — run pipeline first")
    return pd.read_csv(config.PROJECTS_CLEAN_CSV)


def test_researcher_count(researchers):
    assert len(researchers) == 100


def test_researcher_required_fields(researchers):
    for col in ["researcher_id", "name", "academic_rank", "university", "department",
                 "self_declared_specialties", "research_keywords", "paper_titles",
                 "specialty_weights", "num_papers", "topic_diversity", "activity_index"]:
        assert researchers[col].notna().all(), f"Null values in {col}"


def test_researcher_multi_specialty_distribution(researchers):
    counts = researchers["self_declared_specialties"].str.split("|").str.len()
    assert (counts == 1).sum() == 70
    assert (counts == 2).sum() == 10
    assert (counts == 3).sum() == 10
    assert (counts == 4).sum() == 5
    assert (counts == 5).sum() == 5


def test_specialty_weights_sum_to_one(researchers):
    for _, row in researchers.iterrows():
        weights_str = row["specialty_weights"]
        assert weights_str != ""
        weights = dict(item.split(":") for item in str(weights_str).split("|"))
        total = sum(float(v) for v in weights.values())
        assert abs(total - 1.0) < 0.01


def test_project_count(projects):
    assert len(projects) == 500


def test_difficulty_distribution(projects):
    assert (projects["difficulty"] == "easy").sum() == 200
    assert (projects["difficulty"] == "medium").sum() == 200
    assert (projects["difficulty"] == "hard").sum() == 100


def test_each_researcher_has_5_projects(projects, researchers):
    per_researcher = projects.groupby("manager_id").size()
    assert (per_researcher == 5).all()


def test_manager_ids_valid(projects, researchers):
    valid_ids = set(researchers["researcher_id"])
    assert set(projects["manager_id"]).issubset(valid_ids)


def test_project_required_fields(projects):
    for col in ["project_id", "title", "specialty_domain", "abstract",
                 "introduction", "literature_review", "methodology_summary",
                 "results_summary", "keywords", "manager_id", "difficulty", "year"]:
        assert projects[col].notna().all(), f"Null values in {col}"


def test_every_project_has_manager_in_assignments(projects, assignments):
    mgr = assignments[assignments["role"] == "مجری"]
    assert set(mgr["project_id"]) == set(projects["project_id"])


def test_supervisor_rank_constraint(assignments, researchers):
    supervisors = assignments[assignments["role"] == "ناظر"]
    rank_map = researchers.set_index("researcher_id")["academic_rank"]
    for _, row in supervisors.iterrows():
        rank = rank_map[row["researcher_id"]]
        assert rank in config.SUPERVISOR_RANKS


def test_no_همکار_in_assignments(assignments):
    assert "همکار" not in assignments["role"].values


def test_no_researcher_appears_twice_in_same_project(assignments):
    dupes = assignments.groupby(["project_id", "researcher_id"]).size()
    assert (dupes == 1).all()


def test_no_duplicate_titles(projects_clean):
    assert projects_clean["title"].nunique() == len(projects_clean)


def test_no_empty_clean_text(projects_clean):
    assert (projects_clean["clean_text"].str.strip() != "").all()
    assert projects_clean["clean_text"].notna().all()
