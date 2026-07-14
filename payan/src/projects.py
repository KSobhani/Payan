import json
import random
import time
import pandas as pd
from src import config

SYSTEM_PROMPT = (
    "تو یک پژوهشگر ارشد دانشگاهی ایرانی در حوزه علوم کامپیوتر و مهندسی نرم‌افزار هستی. "
    "وظیفه‌ات تولید طرح‌های پژوهشی واقع‌گرایانه به زبان فارسی رسمی دانشگاهی است. "
    "خروجی را فقط به‌صورت JSON معتبر ارائه بده، بدون توضیح اضافه."
)


def _build_phase_a_prompt(
    researcher: dict,
    difficulty: str,
    specialty: str,
    second_specialty: str | None = None,
) -> str:
    keywords = researcher["research_keywords"].replace("|", "، ")

    if difficulty == "easy":
        return (
            f"پژوهشگر {researcher['researcher_id']} در حوزه «{specialty}» تخصص دارد.\n"
            f"کلیدواژه‌های پژوهشی او: {keywords}\n\n"
            "یک طرح پژوهشی آسان بنویس که:\n"
            "- مستقیماً از کلیدواژه‌های ذکرشده استفاده کند (overlap واژگانی بالا)\n"
            f"- در حوزه اصلی «{specialty}» باشد\n\n"
            'خروجی JSON: {"title": "...", "abstract": "...(100-150 کلمه)", '
            '"keywords": [...], "specialty_domain": "..."}'
        )
    elif difficulty == "medium":
        return (
            f"پژوهشگر {researcher['researcher_id']} در حوزه «{specialty}» تخصص دارد.\n"
            f"کلیدواژه‌های پژوهشی او: {keywords}\n\n"
            "یک طرح پژوهشی متوسط بنویس که:\n"
            f"- مفهوماً در حوزه «{specialty}» باشد\n"
            "- از کلیدواژه‌های مستقیم ذکرشده استفاده نکن (واژگان متفاوت، مفهوم مشابه)\n\n"
            'خروجی JSON: {"title": "...", "abstract": "...(100-150 کلمه)", '
            '"keywords": [...], "specialty_domain": "..."}'
        )
    else:
        sp2 = second_specialty or specialty
        return (
            f"یک طرح پژوهشی بین‌رشته‌ای بنویس که حوزه‌های «{specialty}» و «{sp2}» را ترکیب کند.\n"
            "این طرح باید:\n"
            "- ترکیب واقعی و منطقی دو حوزه باشد\n"
            f"- از مفاهیم هر دو حوزه «{specialty}» و «{sp2}» بهره ببرد\n"
            "- چالش اصلی، انگیزه و رویکرد را تشریح کند\n\n"
            'خروجی JSON: {"title": "...", "abstract": "...(100-150 کلمه)", '
            '"keywords": [...], "specialty_domain": "..."}'
        )


def _build_phase_b_prompt(title: str, abstract: str) -> str:
    return (
        f"بر اساس طرح پژوهشی زیر، بخش‌های تکمیلی را به فارسی رسمی بنویس:\n"
        f"عنوان: {title}\n"
        f"چکیده: {abstract}\n\n"
        "خروجی JSON:\n"
        '{"introduction": "...(80-120 کلمه)", '
        '"literature_review": "...(80-120 کلمه)", '
        '"methodology_summary": "...(60-80 کلمه)", '
        '"results_summary": "...(60-80 کلمه)"}'
    )


def _call_gpt(client, system: str, user: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config.GPT_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, Exception):
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def generate_projects(
    researchers_df: pd.DataFrame,
    client=None,
) -> pd.DataFrame:
    from openai import OpenAI
    import os

    if client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.SEED + 1)

    # Build done set keyed by (manager_id, global_seq) so that pre-populated
    # rows with seq=0..12 (global per researcher) are correctly recognised.
    done: set[tuple] = set()
    if config.PROJECTS_CSV.exists():
        existing = pd.read_csv(config.PROJECTS_CSV)
        for _, row in existing.iterrows():
            done.add((row["manager_id"], int(row["seq"])))
    else:
        pd.DataFrame(columns=[
            "project_id", "title", "specialty_domain", "abstract",
            "introduction", "literature_review", "methodology_summary",
            "results_summary", "keywords", "manager_id", "difficulty",
            "year", "seq",
        ]).to_csv(config.PROJECTS_CSV, index=False, encoding="utf-8-sig")

    # Load all rows already written so project_id numbering is correct.
    rows_written: list[dict] = []
    if config.PROJECTS_CSV.exists():
        rows_written = pd.read_csv(config.PROJECTS_CSV).to_dict("records")

    for _, researcher in researchers_df.iterrows():
        rid = researcher["researcher_id"]
        specialties = researcher["self_declared_specialties"].split("|")
        primary = specialties[0]
        secondary = specialties[1] if len(specialties) > 1 else primary

        adjacent = config.SPECIALTIES[primary]["adjacent"]
        hard_pool = [s for s in adjacent if s != primary]
        if not hard_pool:
            hard_pool = [s for s in config.SPECIALTY_LIST if s != primary]
        second_sp = rng.choice(hard_pool)

        # Build a flat task list — global_seq (0..12) is used as the resume key.
        tasks: list[tuple] = []
        for _ in range(5):
            tasks.append(("easy", primary, None))
        for _ in range(5):
            tasks.append(("medium", secondary, None))
        for _ in range(3):
            tasks.append(("hard", primary, second_sp))

        for global_seq, (difficulty, specialty, second_specialty) in enumerate(tasks):
            if (rid, global_seq) in done:
                continue

            phase_a_prompt = _build_phase_a_prompt(
                researcher.to_dict(), difficulty, specialty, second_specialty
            )
            phase_a = _call_gpt(client, SYSTEM_PROMPT, phase_a_prompt)

            phase_b_prompt = _build_phase_b_prompt(
                phase_a.get("title", ""), phase_a.get("abstract", "")
            )
            phase_b = _call_gpt(client, SYSTEM_PROMPT, phase_b_prompt)

            row = {
                "project_id": f"PRJ_{len(rows_written) + 1:04d}",
                "title": phase_a.get("title", ""),
                "specialty_domain": phase_a.get("specialty_domain", specialty),
                "abstract": phase_a.get("abstract", ""),
                "introduction": phase_b.get("introduction", ""),
                "literature_review": phase_b.get("literature_review", ""),
                "methodology_summary": phase_b.get("methodology_summary", ""),
                "results_summary": phase_b.get("results_summary", ""),
                "keywords": "|".join(phase_a.get("keywords", [])),
                "manager_id": rid,
                "difficulty": difficulty,
                "year": rng.randint(1395, 1403),
                "seq": global_seq,
            }
            rows_written.append(row)
            done.add((rid, global_seq))

            pd.DataFrame([row]).to_csv(
                config.PROJECTS_CSV, mode="a", header=False, index=False, encoding="utf-8-sig"
            )

    return pd.read_csv(config.PROJECTS_CSV)


def get_openai_client():
    from openai import OpenAI
    import os
    from dotenv import load_dotenv
    load_dotenv()
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
