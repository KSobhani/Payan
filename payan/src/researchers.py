import random
import pandas as pd
from src import config


def _make_rng() -> random.Random:
    return random.Random(config.SEED)


def _sample_keywords(rng: random.Random, specialty: str, n: int = 5) -> list[str]:
    pool = config.SPECIALTIES[specialty]["keywords"]
    return rng.sample(pool, min(n, len(pool)))


def _fill_template(template: str, keywords: list[str], specialty: str) -> str:
    domain = specialty.split("و")[0].strip()
    kw = keywords[:3] + keywords[:3]
    return template.format(
        method=kw[0], task=kw[1], technique=kw[2],
        target=kw[0], domain=domain,
    )


def _generate_paper_titles(rng: random.Random, specialties: list[str], n: int = 15) -> list[str]:
    titles = []
    per_sp = n // len(specialties)
    extra = n % len(specialties)
    for idx, sp in enumerate(specialties):
        count = per_sp + (1 if idx < extra else 0)
        kws = config.SPECIALTIES[sp]["keywords"]
        templates = config.SPECIALTIES[sp]["title_templates"]
        for _ in range(count):
            tmpl = rng.choice(templates)
            sample = rng.sample(kws, 3)
            titles.append(_fill_template(tmpl, sample, sp))
    rng.shuffle(titles)
    return titles[:n]


def _topic_diversity(rng: random.Random, n_specialties: int) -> float:
    base = (n_specialties - 1) / 4.0
    noise = rng.uniform(-0.05, 0.05)
    return round(min(1.0, max(0.0, base + noise)), 3)


def generate_researchers() -> pd.DataFrame:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    rng = _make_rng()

    counts = list(config.SPECIALTY_COUNT_DIST)
    rng.shuffle(counts)

    specialty_list = config.SPECIALTY_LIST
    n_sp = len(specialty_list)
    rows = []

    for i in range(config.N_RESEARCHERS):
        primary = specialty_list[i % n_sp]
        n_specs = counts[i]

        if n_specs == 1:
            specialties = [primary]
        else:
            adjacent = config.SPECIALTIES[primary]["adjacent"]
            pool = [s for s in adjacent if s != primary]
            if len(pool) < n_specs - 1:
                pool = [s for s in specialty_list if s != primary]
            extra = rng.sample(pool, min(n_specs - 1, len(pool)))
            specialties = [primary] + extra

        keywords: list[str] = []
        for sp in specialties:
            keywords.extend(_sample_keywords(rng, sp, 5))

        paper_titles = _generate_paper_titles(rng, specialties, 15)
        rank = rng.choices(config.ACADEMIC_RANKS, weights=config.RANK_WEIGHTS, k=1)[0]

        rows.append({
            "researcher_id": f"Researcher_{i + 1:03d}",
            "name": f"پژوهشگر_{i + 1:03d}",
            "academic_rank": rank,
            "university": rng.choice(config.UNIVERSITIES),
            "department": primary,
            "self_declared_specialties": "|".join(specialties),
            "research_keywords": "|".join(keywords),
            "paper_titles": "|".join(paper_titles),
            "specialty_weights": "",
            "num_papers": 15,
            "topic_diversity": _topic_diversity(rng, len(specialties)),
            "activity_index": round(rng.uniform(0.3, 1.0), 3),
        })

    df = pd.DataFrame(rows)
    df.to_csv(config.RESEARCHERS_CSV, index=False, encoding="utf-8-sig")
    return df
