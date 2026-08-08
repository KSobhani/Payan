import json
import re
import time
import random
import pandas as pd
from src import config

RESEARCHER_SYSTEM_PROMPT = (
    "تو یک دستیار ساخت داده‌های مصنوعی برای پژوهش دانشگاهی هستی. "
    "وظیفه‌ات ساخت پروفایل‌های واقع‌گرایانه از پژوهشگران دانشگاهی ایرانی است. "
    "خروجی را فقط به‌صورت JSON معتبر ارائه بده، بدون هیچ توضیح اضافه."
)


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


def _build_researcher_prompt(
    researcher_id: str,
    academic_rank: str,
    specialties: list[str],
    topic_div: float,
    activity_idx: float,
) -> str:
    primary = specialties[0]
    n_kw = len(specialties) * 5

    if len(specialties) == 1:
        sp_desc = f"تخصص: «{primary}» (تک‌تخصصی)"
    else:
        others = "، ".join(f"«{s}»" for s in specialties[1:])
        sp_desc = f"تخصص اصلی: «{primary}» — تخصص‌های دیگر: {others}"

    return (
        f"{sp_desc}\n"
        f"رتبه علمی: {academic_rank}\n"
        f"کلیدواژه‌های پژوهشی: دقیقاً {n_kw} کلیدواژه (۵ عدد از هر تخصص) — فارسی یا ترکیب فارسی/انگلیسی تخصصی\n"
        f"عناوین مقالات: دقیقاً ۱۵ عنوان مقاله دانشگاهی فارسی — واقعی‌نما، از هر تخصص، متنوع\n\n"
        f"خروجی JSON با این فیلدها دقیقاً:\n"
        f'{{\n'
        f'  "researcher_id": "{researcher_id}",\n'
        f'  "name": "[نام و نام‌خانوادگی فارسی واقعی‌نما]",\n'
        f'  "academic_rank": "{academic_rank}",\n'
        f'  "university": "[نام یک دانشگاه ایرانی]",\n'
        f'  "department": "{primary}",\n'
        f'  "self_declared_specialties": "{"|".join(specialties)}",\n'
        f'  "research_keywords": "[کلمه۱|کلمه۲|...|کلمه{n_kw}]",\n'
        f'  "paper_titles": "[عنوان ۱|عنوان ۲|...|عنوان ۱۵]",\n'
        f'  "specialty_weights": "",\n'
        f'  "num_papers": 15,\n'
        f'  "topic_diversity": {topic_div},\n'
        f'  "activity_index": {activity_idx}\n'
        f'}}'
    )


def _call_gemini_researcher(client, system: str, user: str, max_retries: int = 8) -> dict:
    from google.genai import types

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    temperature=config.GEMINI_TEMPERATURE,
                ),
                contents=user,
            )
            return json.loads(response.text)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                match = re.search(r"retry in ([\d.]+)s", str(e))
                wait = float(match.group(1)) + 5 if match else 65
                print(f"  ⏳ Rate limit — {wait:.0f}s صبر کن... (تلاش {attempt+1}/{max_retries})")
                print(f"  ❌ {e}")
                time.sleep(wait)
            elif attempt == max_retries - 1:
                print(f"  ❌ خطا نهایی: {type(e).__name__}: {e}")
                raise
            else:
                print(f"  ❌ خطا (تلاش {attempt+1}/{max_retries}): {type(e).__name__}: {e}")
                time.sleep(2 ** attempt)
    return {}


def generate_researchers() -> pd.DataFrame:
    """Programmatic researcher generation (no API) — for testing and fallback."""
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


def generate_researchers_with_api(client=None) -> pd.DataFrame:
    """AI-powered researcher generation — realistic names, universities, paper titles."""
    from src.projects import get_gemini_client

    if client is None:
        client = get_gemini_client()

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    rng = _make_rng()

    counts = list(config.SPECIALTY_COUNT_DIST)
    rng.shuffle(counts)

    specialty_list = config.SPECIALTY_LIST
    n_sp = len(specialty_list)

    # Resume: بارگذاری پژوهشگرهای قبلاً ساخته‌شده
    done_ids: set[str] = set()
    if config.RESEARCHERS_CSV.exists():
        existing = pd.read_csv(config.RESEARCHERS_CSV)
        done_ids = set(existing["researcher_id"].tolist())
        print(f"  ♻️  {len(done_ids)} پژوهشگر از قبل موجود — ادامه از همون‌جا")
    else:
        # ساخت فایل خالی با هدر
        pd.DataFrame(columns=[
            "researcher_id", "name", "academic_rank", "university", "department",
            "self_declared_specialties", "research_keywords", "paper_titles",
            "specialty_weights", "num_papers", "topic_diversity", "activity_index",
        ]).to_csv(config.RESEARCHERS_CSV, index=False, encoding="utf-8-sig")

    for i in range(config.N_RESEARCHERS):
        researcher_id = f"Researcher_{i + 1:03d}"

        # مصرف rng برای حفظ seed یکسان حتی اگه skip بشه
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

        rank = rng.choices(config.ACADEMIC_RANKS, weights=config.RANK_WEIGHTS, k=1)[0]
        topic_div = _topic_diversity(rng, len(specialties))
        activity_idx = round(rng.uniform(0.3, 1.0), 3)

        if researcher_id in done_ids:
            continue

        prompt = _build_researcher_prompt(researcher_id, rank, specialties, topic_div, activity_idx)
        result = _call_gemini_researcher(client, RESEARCHER_SYSTEM_PROMPT, prompt)

        # Override fixed fields to ensure consistency with config
        result["researcher_id"] = researcher_id
        result["academic_rank"] = rank
        result["department"] = primary
        result["self_declared_specialties"] = "|".join(specialties)
        result["specialty_weights"] = ""
        result["num_papers"] = 15
        result["topic_diversity"] = topic_div
        result["activity_index"] = activity_idx

        # ذخیره فوری
        pd.DataFrame([result]).to_csv(
            config.RESEARCHERS_CSV, mode="a", header=False, index=False, encoding="utf-8-sig"
        )
        done_ids.add(researcher_id)
        print(f"  ✅ [{len(done_ids)}/{config.N_RESEARCHERS}] {researcher_id} — {rank} | {primary[:30]}")

    return pd.read_csv(config.RESEARCHERS_CSV)
