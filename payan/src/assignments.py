import random
import pandas as pd
from src import config


def assign_roles(
    projects_df: pd.DataFrame,
    researchers_df: pd.DataFrame,
) -> pd.DataFrame:
    rng = random.Random(config.SEED + 2)

    supervisor_pool: dict[str, list[str]] = {}
    collaborator_pool: dict[str, list[str]] = {}
    for _, r in researchers_df.iterrows():
        rid = r["researcher_id"]
        rank = r["academic_rank"]
        specs = r["self_declared_specialties"].split("|")
        primary = specs[0]
        adjacent = config.SPECIALTIES[primary]["adjacent"]
        relevant_specs = set(specs) | set(adjacent)

        for sp in relevant_specs:
            if rank in config.SUPERVISOR_RANKS:
                supervisor_pool.setdefault(sp, []).append(rid)
            collaborator_pool.setdefault(sp, []).append(rid)

    rows: list[dict] = []
    for _, project in projects_df.iterrows():
        pid = project["project_id"]
        mgr = project["manager_id"]
        domain = project["specialty_domain"]

        rows.append({"project_id": pid, "researcher_id": mgr, "role": "مجری"})

        # 70% get a supervisor, 30% are مجری only
        if rng.random() < 0.70:
            pool = [r for r in supervisor_pool.get(domain, []) if r != mgr]
            if not pool:
                pool = [
                    r["researcher_id"] for _, r in researchers_df.iterrows()
                    if r["academic_rank"] in config.SUPERVISOR_RANKS and r["researcher_id"] != mgr
                ]
            if pool:
                supervisor = rng.choice(pool)
                rows.append({"project_id": pid, "researcher_id": supervisor, "role": "ناظر"})

    df = pd.DataFrame(rows)
    df.to_csv(config.ASSIGNMENTS_CSV, index=False, encoding="utf-8-sig")
    return df


def compute_specialty_weights(
    projects_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    researchers_df: pd.DataFrame,
) -> pd.DataFrame:
    managers = assignments_df[assignments_df["role"] == "مجری"]
    proj_domain = projects_df.set_index("project_id")["specialty_domain"]

    manager_projects = managers.copy()
    manager_projects["specialty_domain"] = manager_projects["project_id"].map(proj_domain)
    counts = (
        manager_projects.groupby(["researcher_id", "specialty_domain"])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("researcher_id")["count"].sum()

    weight_strings: dict[str, str] = {}
    for rid, group in counts.groupby("researcher_id"):
        total = totals[rid]
        parts = [f"{row['specialty_domain']}:{row['count'] / total:.3f}" for _, row in group.iterrows()]
        weight_strings[rid] = "|".join(parts)

    df = researchers_df.copy()
    df["specialty_weights"] = df["researcher_id"].map(weight_strings).fillna("")
    df.to_csv(config.RESEARCHERS_CSV, index=False, encoding="utf-8-sig")
    return df
