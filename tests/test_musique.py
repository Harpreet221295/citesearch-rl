"""Offline test for the MuSiQue -> DRTask mapping (GENERALIZATION_EVAL.md). No network."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data as dr_data


def _row():
    return {
        "id": "3hop1__12_34_56",
        "question": "Who is the spouse of the Green performer?",
        "answer": "Miquette Giraudy",
        "answer_aliases": ["Giraudy", "Miquette Giraudy"],
        "answerable": True,
        "paragraphs": [
            {"idx": 0, "title": "Green (Steve Hillage album)", "paragraph_text": "Green is an album by Steve Hillage.", "is_supporting": True},
            {"idx": 1, "title": "Miquette Giraudy", "paragraph_text": "Miquette Giraudy is the partner of Steve Hillage.", "is_supporting": True},
            {"idx": 2, "title": "Green", "paragraph_text": "Green is a colour.", "is_supporting": False},
            {"idx": 3, "title": "Green", "paragraph_text": "Green is also a surname.", "is_supporting": False},
            {"idx": 4, "title": "Steve Hillage", "paragraph_text": "Steve Hillage is a guitarist.", "is_supporting": True},
        ],
    }


def test_musique_row_maps_to_drtask_with_aliases_and_gold():
    t = dr_data.musique_row_to_task(_row(), "musique-validation-7")
    assert t.task_id == "musique-validation-7"
    assert t.gold_answer == "Miquette Giraudy"
    assert t.gold_aliases == ["Giraudy"]                 # alias equal to the answer is dropped
    assert "Miquette Giraudy" in t.answers and "Giraudy" in t.answers
    assert t.supporting_titles == ["Green (Steve Hillage album)", "Miquette Giraudy", "Steve Hillage"]
    assert len(t.passages) == 5
    assert t.meta["dataset"] == "musique" and t.meta["hops"] == "3hop1" and t.meta["n_hops"] == 3


def test_duplicate_titles_are_disambiguated_for_read():
    t = dr_data.musique_row_to_task(_row(), "x")
    titles = [p[0] for p in t.passages]
    assert len(titles) == len(set(titles)), titles
    assert "Green (2)" in titles
    store = t.docstore()
    obs, ok, err, got = __import__("tools").execute(store, "read", {"title": "Green (2)"})
    assert ok and "surname" in obs
    # retrieval hit-rate must work on the disambiguated names too
    assert set(t.supporting_titles) <= set(titles)
