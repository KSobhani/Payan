import json
import random
import time
import pandas as pd
from src import config

SYSTEM_PROMPT = (
    "تو یک پژوهشگر ارشد دانشگاهی ایرانی هستی. "
    "وظیفه‌ات تولید طرح‌های پژوهشی واقع‌گرایانه به زبان فارسی رسمی دانشگاهی است. "
    "خروجی را فقط به‌صورت JSON معتبر ارائه بده، بدون توضیح اضافه."
)


def _build_phase_a_prompt(
    researcher: dict,
    difficulty: str,
    specialty: str,
    sub_topic: str,
    second_specialty: str | None = None,
    second_sub_topic: str | None = None,
) -> str:
    keywords = researcher["research_keywords"].replace("|", "، ")

    if difficulty == "easy":
        return (
            f"پژوهشگر {researcher['researcher_id']} در حوزه «{specialty}» تخصص دارد.\n"
            f"زیرموضوع: «{sub_topic}»\n"
            f"کلیدواژه‌های پژوهشی او: {keywords}\n\n"
            "یک پروپوزال پژوهشی آسان بنویس که:\n"
            f"- دقیقاً در زیرموضوع «{sub_topic}» باشد\n"
            "- مستقیماً از کلیدواژه‌های ذکرشده استفاده کند (overlap واژگانی بالا)\n\n"
            'خروجی JSON: {"title": "...", "abstract": "...(100-150 کلمه)", '
            '"keywords": [...], "specialty_domain": "..."}'
        )
    elif difficulty == "medium":
        return (
            f"پژوهشگر {researcher['researcher_id']} در حوزه «{specialty}» تخصص دارد.\n"
            f"زیرموضوع: «{sub_topic}»\n"
            f"کلیدواژه‌های پژوهشی او: {keywords}\n\n"
            "یک پروپوزال پژوهشی متوسط بنویس که:\n"
            f"- مفهوماً در زیرموضوع «{sub_topic}» باشد\n"
            "- از کلیدواژه‌های مستقیم ذکرشده استفاده نکن (واژگان متفاوت، مفهوم مشابه)\n\n"
            'خروجی JSON: {"title": "...", "abstract": "...(100-150 کلمه)", '
            '"keywords": [...], "specialty_domain": "..."}'
        )
    else:
        sp2 = second_specialty or specialty
        st2 = second_sub_topic or sub_topic
        return (
            f"یک پروپوزال پژوهشی بین‌رشته‌ای بنویس که حوزه‌های «{specialty}» "
            f"(زیرموضوع: {sub_topic}) و «{sp2}» (زیرموضوع: {st2}) را ترکیب کند.\n"
            "این طرح باید:\n"
            "- ترکیب واقعی و منطقی دو حوزه باشد\n"
            f"- از مفاهیم هر دو زیرموضوع بهره ببرد\n"
            "- چالش اصلی، انگیزه و رویکرد را تشریح کند\n\n"
            'خروجی JSON: {"title": "...", "abstract": "...(100-150 کلمه)", '
            '"keywords": [...], "specialty_domain": "..."}'
        )


def _build_phase_b_prompt(title: str, abstract: str) -> str:
    return (
        f"بر اساس پروپوزال پژوهشی زیر، بخش‌های تکمیلی را به فارسی رسمی بنویس:\n"
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

    rows_written: list[dict] = []
    if config.PROJECTS_CSV.exists():
        rows_written = pd.read_csv(config.PROJECTS_CSV).to_dict("records")

    for _, researcher in researchers_df.iterrows():
        rid = researcher["researcher_id"]
        specialties = researcher["self_declared_specialties"].split("|")
        primary = specialties[0]
        secondary = specialties[1] if len(specialties) > 1 else primary

        primary_subs = config.SPECIALTIES[primary]["sub_topics"]
        secondary_subs = config.SPECIALTIES[secondary]["sub_topics"]

        adjacent = config.SPECIALTIES[primary]["adjacent"]
        hard_pool = [s for s in adjacent if s != primary]
        if not hard_pool:
            hard_pool = [s for s in config.SPECIALTY_LIST if s != primary]
        cross_specialty = rng.choice(hard_pool)
        cross_sub = rng.choice(config.SPECIALTIES[cross_specialty]["sub_topics"])

        # 7 tasks per researcher: seq 0-2 easy, 3-5 medium, 6 hard
        tasks = [
            (0, "easy",   primary,   primary_subs[0],   None,            None),
            (1, "easy",   primary,   primary_subs[1],   None,            None),
            (2, "easy",   primary,   primary_subs[2],   None,            None),
            (3, "medium", secondary, secondary_subs[0], None,            None),
            (4, "medium", secondary, secondary_subs[1], None,            None),
            (5, "medium", secondary, secondary_subs[2], None,            None),
            (6, "hard",   primary,   primary_subs[0],   cross_specialty, cross_sub),
        ]

        for seq, difficulty, specialty, sub_topic, second_specialty, second_sub_topic in tasks:
            if (rid, seq) in done:
                continue

            phase_a_prompt = _build_phase_a_prompt(
                researcher.to_dict(), difficulty, specialty, sub_topic,
                second_specialty, second_sub_topic,
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
                "seq": seq,
            }
            rows_written.append(row)
            done.add((rid, seq))

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
