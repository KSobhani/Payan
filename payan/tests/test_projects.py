import json
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src import config
from src.researchers import generate_researchers
from src.projects import (
    _build_phase_a_prompt,
    _build_phase_b_prompt,
    _call_gpt,
    generate_projects,
)

FAKE_RESEARCHER = {
    "researcher_id": "Researcher_001",
    "self_declared_specialties": "یادگیری ماشین و داده‌کاوی|پردازش زبان طبیعی",
    "research_keywords": "یادگیری عمیق|شبکه عصبی|تحلیل احساسات|تشخیص موجودیت|BERT|دسته‌بندی|ماشین بردار پشتیبان|خوشه‌بندی|داده‌کاوی|ویژگی‌سازی",
    "paper_titles": "عنوان ۱|عنوان ۲|عنوان ۳|عنوان ۴|عنوان ۵|عنوان ۶|عنوان ۷|عنوان ۸|عنوان ۹|عنوان ۱۰|عنوان ۱۱|عنوان ۱۲|عنوان ۱۳|عنوان ۱۴|عنوان ۱۵",
    "academic_rank": "استادیار",
    "department": "یادگیری ماشین و داده‌کاوی",
}

FAKE_PHASE_A = {
    "title": "پیش‌بینی احساسات با شبکه عصبی",
    "abstract": "این پژوهش روشی برای تحلیل احساسات ارائه می‌دهد.",
    "keywords": ["شبکه عصبی", "تحلیل احساسات"],
    "specialty_domain": "یادگیری ماشین و داده‌کاوی",
}

FAKE_PHASE_B = {
    "introduction": "در این پژوهش مسئله تحلیل احساسات بررسی می‌شود.",
    "literature_review": "پژوهش‌های پیشین نشان داده‌اند که...",
    "methodology_summary": "از شبکه عصبی LSTM استفاده شده است.",
    "results_summary": "نتایج نشان دهنده بهبود ۱۵ درصدی دقت است.",
}


def test_phase_a_prompt_easy_uses_keywords():
    prompt = _build_phase_a_prompt(FAKE_RESEARCHER, "easy", "یادگیری ماشین و داده‌کاوی")
    assert "کلیدواژه" in prompt or "keywords" in prompt.lower()


def test_phase_a_prompt_medium_avoids_direct_keywords():
    prompt = _build_phase_a_prompt(FAKE_RESEARCHER, "medium", "پردازش زبان طبیعی")
    assert "بدون" in prompt or "استفاده نکن" in prompt or "واژگان متفاوت" in prompt


def test_phase_a_prompt_hard_blends_two_specialties():
    prompt = _build_phase_a_prompt(FAKE_RESEARCHER, "hard", "یادگیری ماشین و داده‌کاوی", second_specialty="پردازش زبان طبیعی")
    assert "پردازش زبان طبیعی" in prompt
    assert "یادگیری ماشین" in prompt


def test_phase_b_prompt_references_title_and_abstract():
    prompt = _build_phase_b_prompt(FAKE_PHASE_A["title"], FAKE_PHASE_A["abstract"])
    assert FAKE_PHASE_A["title"] in prompt
    assert "introduction" in prompt or "مقدمه" in prompt


def test_call_gpt_returns_parsed_dict(mock_paths):
    client = MagicMock()
    r = MagicMock()
    r.choices[0].message.content = json.dumps(FAKE_PHASE_A, ensure_ascii=False)
    client.chat.completions.create.return_value = r
    result = _call_gpt(client, "system prompt", "user prompt")
    assert result == FAKE_PHASE_A


def test_call_gpt_retries_on_json_error(mock_paths):
    client = MagicMock()
    bad_response = MagicMock()
    bad_response.choices[0].message.content = "not json"
    good_response = MagicMock()
    good_response.choices[0].message.content = json.dumps(FAKE_PHASE_A)
    client.chat.completions.create.side_effect = [bad_response, good_response]
    result = _call_gpt(client, "sys", "user", max_retries=2)
    assert result == FAKE_PHASE_A


def test_generate_projects_count(mock_paths):
    researchers_df = generate_researchers()
    client = MagicMock()
    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        r = MagicMock()
        if call_count["n"] % 2 == 1:
            r.choices[0].message.content = json.dumps(FAKE_PHASE_A, ensure_ascii=False)
        else:
            r.choices[0].message.content = json.dumps(FAKE_PHASE_B, ensure_ascii=False)
        return r

    client.chat.completions.create.side_effect = side_effect
    df = generate_projects(researchers_df, client=client)
    assert len(df) == 1300


def test_generate_projects_difficulty_distribution(mock_paths):
    researchers_df = generate_researchers()
    client = MagicMock()
    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        r = MagicMock()
        if call_count["n"] % 2 == 1:
            r.choices[0].message.content = json.dumps(FAKE_PHASE_A, ensure_ascii=False)
        else:
            r.choices[0].message.content = json.dumps(FAKE_PHASE_B, ensure_ascii=False)
        return r

    client.chat.completions.create.side_effect = side_effect
    df = generate_projects(researchers_df, client=client)
    assert (df["difficulty"] == "easy").sum() == 500
    assert (df["difficulty"] == "medium").sum() == 500
    assert (df["difficulty"] == "hard").sum() == 300


def test_generate_projects_schema(mock_paths):
    researchers_df = generate_researchers()
    client = MagicMock()
    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        r = MagicMock()
        if call_count["n"] % 2 == 1:
            r.choices[0].message.content = json.dumps(FAKE_PHASE_A, ensure_ascii=False)
        else:
            r.choices[0].message.content = json.dumps(FAKE_PHASE_B, ensure_ascii=False)
        return r

    client.chat.completions.create.side_effect = side_effect
    df = generate_projects(researchers_df, client=client)
    required_cols = [
        "project_id", "title", "specialty_domain", "abstract",
        "introduction", "literature_review", "methodology_summary",
        "results_summary", "keywords", "manager_id", "difficulty", "year",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_generate_projects_resume(mock_paths):
    researchers_df = generate_researchers()
    first_id = researchers_df["researcher_id"].iloc[0]
    existing = pd.DataFrame([{
        "project_id": f"PRJ_{i+1:04d}",
        "title": f"عنوان {i}", "specialty_domain": "یادگیری ماشین و داده‌کاوی",
        "abstract": "چکیده", "introduction": "مقدمه", "literature_review": "پیشینه",
        "methodology_summary": "روش", "results_summary": "نتایج",
        "keywords": "کلید۱", "manager_id": first_id,
        "difficulty": ["easy", "medium", "hard"][i % 3],
        "year": 1400, "seq": i,
    } for i in range(13)])
    existing.to_csv(config.PROJECTS_CSV, index=False, encoding="utf-8-sig")

    call_count = {"n": 0}
    client = MagicMock()
    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        r = MagicMock()
        if call_count["n"] % 2 == 1:
            r.choices[0].message.content = json.dumps(FAKE_PHASE_A, ensure_ascii=False)
        else:
            r.choices[0].message.content = json.dumps(FAKE_PHASE_B, ensure_ascii=False)
        return r
    client.chat.completions.create.side_effect = side_effect

    df = generate_projects(researchers_df, client=client)
    assert len(df) == 1300
    assert call_count["n"] == (1300 - 13) * 2
