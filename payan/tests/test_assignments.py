import pytest
import pandas as pd
from src import config
from src.researchers import generate_researchers
from src.assignments import assign_roles, compute_specialty_weights


@pytest.fixture
def researchers(mock_paths):
    return generate_researchers()


@pytest.fixture
def projects(researchers):
    rows = []
    for _, r in researchers.iterrows():
        specs = r["self_declared_specialties"].split("|")
        for seq, diff in enumerate(["easy"] * 5 + ["medium"] * 5 + ["hard"] * 3):
            rows.append({
                "project_id": f"PRJ_{len(rows)+1:04d}",
                "title": f"طرح {len(rows)+1}",
                "specialty_domain": specs[0],
                "abstract": "چکیده",
                "introduction": "مقدمه",
                "literature_review": "پیشینه",
                "methodology_summary": "روش",
                "results_summary": "نتایج",
                "keywords": "کلید",
                "manager_id": r["researcher_id"],
                "difficulty": diff,
                "year": 1400,
                "seq": seq,
            })
    return pd.DataFrame(rows)


def test_assignments_row_count(researchers, projects):
    asgn = assign_roles(projects, researchers)
    assert len(asgn) >= len(projects)


def test_every_project_has_manager(researchers, projects):
    asgn = assign_roles(projects, researchers)
    mgr = asgn[asgn["role"] == "مجری"]
    assert set(mgr["project_id"]) == set(projects["project_id"])


def test_manager_id_matches_projects(researchers, projects):
    asgn = assign_roles(projects, researchers)
    mgr = asgn[asgn["role"] == "مجری"].set_index("project_id")
    proj = projects.set_index("project_id")
    for pid in proj.index:
        assert mgr.loc[pid, "researcher_id"] == proj.loc[pid, "manager_id"]


def test_supervisor_rank_constraint(researchers, projects):
    asgn = assign_roles(projects, researchers)
    supervisors = asgn[asgn["role"] == "ناظر"]
    if len(supervisors) == 0:
        pytest.skip("No supervisors assigned")
    rank_map = researchers.set_index("researcher_id")["academic_rank"]
    for _, row in supervisors.iterrows():
        rank = rank_map[row["researcher_id"]]
        assert rank in config.SUPERVISOR_RANKS, f"Supervisor {row['researcher_id']} has invalid rank {rank}"


def test_no_duplicate_roles_per_project(researchers, projects):
    asgn = assign_roles(projects, researchers)
    dupes = asgn.groupby(["project_id", "researcher_id"]).size()
    assert (dupes == 1).all(), "Same researcher assigned twice to same project"


def test_role_distribution_approx(researchers, projects):
    asgn = assign_roles(projects, researchers)
    project_roles = asgn.groupby("project_id")["role"].apply(set)
    n_supervisor = sum("ناظر" in roles for roles in project_roles)
    n_collaborator = sum("همکار" in roles for roles in project_roles)
    n_total = len(projects)
    assert 0.75 * n_total <= n_supervisor <= 0.95 * n_total
    assert 0.20 * n_total <= n_collaborator <= 0.45 * n_total


def test_compute_specialty_weights(researchers, projects):
    asgn = assign_roles(projects, researchers)
    updated = compute_specialty_weights(projects, asgn, researchers)
    for _, row in updated.iterrows():
        weights_str = row["specialty_weights"]
        assert weights_str != "", f"Researcher {row['researcher_id']} has empty weights"
        weights = dict(item.split(":") for item in weights_str.split("|"))
        total = sum(float(v) for v in weights.values())
        assert abs(total - 1.0) < 0.01, f"Weights don't sum to 1 for {row['researcher_id']}"
