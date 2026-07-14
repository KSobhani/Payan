import pytest
import pandas as pd
from src import config
from src.researchers import generate_researchers


@pytest.fixture
def df(mock_paths):
    return generate_researchers()


def test_row_count(df):
    assert len(df) == 100


def test_required_columns(df):
    required = [
        "researcher_id", "name", "academic_rank", "university", "department",
        "self_declared_specialties", "research_keywords", "paper_titles",
        "specialty_weights", "num_papers", "topic_diversity", "activity_index",
    ]
    assert all(c in df.columns for c in required)


def test_researcher_ids(df):
    assert df["researcher_id"].iloc[0] == "Researcher_001"
    assert df["researcher_id"].iloc[-1] == "Researcher_100"


def test_multi_specialty_distribution(df):
    counts = df["self_declared_specialties"].str.split("|").str.len()
    assert (counts == 1).sum() == 70
    assert (counts == 2).sum() == 10
    assert (counts == 3).sum() == 10
    assert (counts == 4).sum() == 5
    assert (counts == 5).sum() == 5


def test_keywords_per_specialty(df):
    for _, row in df.iterrows():
        n_sp = len(row["self_declared_specialties"].split("|"))
        n_kw = len(row["research_keywords"].split("|"))
        assert n_kw == n_sp * 5, f"Researcher {row['researcher_id']}: expected {n_sp*5} keywords, got {n_kw}"


def test_paper_titles_count(df):
    for _, row in df.iterrows():
        assert len(row["paper_titles"].split("|")) == 15


def test_num_papers_fixed(df):
    assert (df["num_papers"] == 15).all()


def test_academic_rank_valid(df):
    assert set(df["academic_rank"]).issubset(set(config.ACADEMIC_RANKS))


def test_specialty_weights_empty_initially(df):
    assert (df["specialty_weights"] == "").all()


def test_topic_diversity_range(df):
    assert df["topic_diversity"].between(0.0, 1.0).all()


def test_activity_index_range(df):
    assert df["activity_index"].between(0.0, 1.0).all()


def test_csv_saved(mock_paths):
    generate_researchers()
    assert config.RESEARCHERS_CSV.exists()


def test_reproducible(mock_paths):
    df1 = generate_researchers()
    df2 = generate_researchers()
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))


def test_additional_specialties_from_adjacent(df):
    for _, row in df.iterrows():
        specs = row["self_declared_specialties"].split("|")
        if len(specs) > 1:
            primary = specs[0]
            adjacent = config.SPECIALTIES[primary]["adjacent"]
            all_valid = set(adjacent) | set(config.SPECIALTY_LIST)
            for sp in specs[1:]:
                assert sp in all_valid
