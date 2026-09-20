"""Data loading — multi-hop QA (HotpotQA + 2WikiMultiHopQA), plus a tiny bundled
fixture so local sanity needs NO download and NO network.

Each example is one multi-hop question that ships with its own ~10 candidate
paragraphs (2 gold "supporting" + 8 distractors). That per-question paragraph set
IS the offline corpus for Branch B — the direct analogue of finqa_agent's "gold
table + distractor tables". So we don't need the full Wikipedia dump to run the
loop locally; the distractors make retrieval a real skill and `supporting_titles`
gives us a free retrieval-hit-rate metric.

Two paths:
  * HF datasets — `hotpot_qa` (distractor config) and `2wikimultihopqa` mirrors —
    used for the real run. Loader normalizes both to the same DRTask shape.
  * A BUNDLED FIXTURE (fixture_examples()) — a handful of hand-written 2-hop
    questions with clean short-answer golds, each solvable via search→read→answer.
    sanity_preset()/use_fixture forces this so proof-of-life is offline.

Everything here is mechanics — safe scaffold. The judgement calls (which fields to
read from each dataset's schema, the distractor policy) are documented inline and
logged so a run is reproducible from the config alone.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from corpus import DocStore


@dataclass
class DRTask:
    """One multi-hop question presented to the agent, with its candidate corpus."""
    task_id: str
    question: str
    gold_answer: str
    gold_aliases: list[str] = field(default_factory=list)      # acceptable answer variants
    supporting_titles: list[str] = field(default_factory=list) # gold evidence paragraph titles
    passages: list[tuple[str, str]] = field(default_factory=list)  # (title, text) candidate set
    meta: dict[str, Any] = field(default_factory=dict)         # dataset, hop count, level…

    def docstore(self, **kw) -> DocStore:
        """The DocStore the agent searches for THIS question (bundled backend)."""
        return DocStore.from_passages(self.passages, **kw)

    @property
    def answers(self) -> list[str]:
        return [self.gold_answer, *self.gold_aliases]


# ---------------------------------------------------------------------------
# Bundled offline fixture — 4 hand-written 2-hop questions. Each has gold + a
# couple of distractor paragraphs so search/read/answer are all exercised and the
# retrieval-hit-rate metric is non-trivial. Answers are short & unambiguous so EM
# is a clean signal for the sanity reward. NO network needed.
# ---------------------------------------------------------------------------
def fixture_examples() -> list[DRTask]:
    return [
        DRTask(
            task_id="fx-1",
            question="What is the nationality of the director of the film 'Blue Harvest'?",
            gold_answer="American",
            gold_aliases=["american", "United States", "USA", "U.S."],
            supporting_titles=["Blue Harvest (film)", "Jane Doe (director)"],
            passages=[
                ("Blue Harvest (film)",
                 "Blue Harvest is a 2009 drama film. It was directed by Jane Doe and "
                 "premiered at the Toronto festival. The film follows a family farm over one summer."),
                ("Jane Doe (director)",
                 "Jane Doe is an American film director, born in Ohio in 1971. She is known "
                 "for independent dramas including Blue Harvest and Winter Wheat."),
                ("Blue Harvest (album)",
                 "Blue Harvest is a 2015 studio album by the Norwegian band Fjord. It reached "
                 "number three on the domestic chart."),
                ("John Roe (producer)",
                 "John Roe is a British film producer who has worked on several independent dramas."),
            ],
        ),
        DRTask(
            task_id="fx-2",
            question="In which city is the university that employed physicist Alan Prime headquartered?",
            gold_answer="Zurich",
            gold_aliases=["zurich", "Zürich"],
            supporting_titles=["Alan Prime", "Federal Polytechnic"],
            passages=[
                ("Alan Prime",
                 "Alan Prime is a theoretical physicist known for work on lattice models. "
                 "From 1998 he was a professor at the Federal Polytechnic."),
                ("Federal Polytechnic",
                 "The Federal Polytechnic is a public research university headquartered in "
                 "Zurich, Switzerland, founded in 1855."),
                ("Alan Prime (footballer)",
                 "Alan Prime is a retired footballer who played as a defender in the lower leagues."),
                ("Zurich",
                 "Zurich is the largest city in Switzerland and a global centre for banking."),
            ],
        ),
        DRTask(
            task_id="fx-3",
            question="Who wrote the novel that inspired the film adaptation released in 1994 titled 'The Long Road'?",
            gold_answer="Maria Vann",
            gold_aliases=["maria vann", "M. Vann"],
            supporting_titles=["The Long Road (film)", "The Long Road (novel)"],
            passages=[
                ("The Long Road (film)",
                 "The Long Road is a 1994 film adaptation of the novel of the same name. "
                 "It stars two unknown leads and was shot in New Mexico."),
                ("The Long Road (novel)",
                 "The Long Road is a 1988 novel written by Maria Vann, telling the story of a "
                 "cross-country journey. It won a regional literary prize."),
                ("The Short Road (novel)",
                 "The Short Road is a 1990 novella by Peter Quill, unrelated to The Long Road."),
                ("New Mexico",
                 "New Mexico is a state in the southwestern United States, known for its deserts."),
            ],
        ),
        DRTask(
            task_id="fx-4",
            question="What year was the company founded that manufactures the 'Cirrus X' smartphone?",
            gold_answer="2004",
            gold_aliases=["2004"],
            supporting_titles=["Cirrus X", "Nimbus Electronics"],
            passages=[
                ("Cirrus X",
                 "The Cirrus X is a flagship smartphone manufactured by Nimbus Electronics, "
                 "released in 2019 with a titanium frame."),
                ("Nimbus Electronics",
                 "Nimbus Electronics is a consumer-electronics company founded in 2004 and "
                 "headquartered in Taipei. It makes phones, tablets, and audio gear."),
                ("Cirrus (cloud)",
                 "Cirrus clouds are thin, wispy clouds that form at high altitude."),
                ("Stratus Devices",
                 "Stratus Devices is a rival electronics firm founded in 1999."),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# HF loaders (real run). Kept minimal + normalizing — both datasets map onto the
# same DRTask. Called only when use_fixture is False; import of `datasets` is lazy
# so the fixture path has zero heavy deps.
# ---------------------------------------------------------------------------
def _load_hotpotqa(split: str, n: int, seed: int) -> list[DRTask]:
    """HotpotQA distractor config: each row has `context` = {title: [...], sentences: [[...]]},
    `supporting_facts` = {title: [...], sent_id: [...]}, `answer`, `question`."""
    from datasets import load_dataset
    # RESOLVED 2026-08-23: bare "hotpot_qa" is a legacy script-based dataset id
    # that no longer resolves under datasets>=3 / huggingface_hub's newer URI
    # parser (HfUriError: "Repository id must be 'namespace/name'") — same class
    # of bug as the 2WikiMultihopQA mirror fix above. "hotpotqa/hotpot_qa" is the
    # namespaced mirror with an identical schema (question/answer/level/type/
    # context={title,sentences}/supporting_facts={title,sent_id}), verified on
    # the pod before this fix landed.
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    tasks: list[DRTask] = []
    for i in idxs[:n]:
        row = ds[i]
        titles = row["context"]["title"]
        sents = row["context"]["sentences"]
        # str(): 2026-09-07, one 2Wiki title in the rl_train split is an int (a bare
        # year) — it broke pyarrow's struct inference for the verl parquet AND would
        # never match a `read[<title>]` string lookup. Coerce at the source, both loaders.
        passages = [(str(t), " ".join(map(str, s))) for t, s in zip(titles, sents)]
        tasks.append(DRTask(
            task_id=f"hotpot-{split}-{i}",
            question=row["question"],
            gold_answer=row["answer"],
            gold_aliases=[],
            supporting_titles=[str(x) for x in dict.fromkeys(row["supporting_facts"]["title"])],
            passages=passages,
            meta={"dataset": "hotpotqa", "level": row.get("level"), "type": row.get("type")},
        ))
    return tasks


def _load_2wiki(split: str, n: int, seed: int) -> list[DRTask]:
    """2WikiMultiHopQA. VERIFIED 2026-08-23 on the pod: "xanhho/2WikiMultihopQA" is a
    script-based dataset and FAILS on datasets>=3 ("Dataset scripts are no longer
    supported" — the exact RUNPOD_PLAYBOOK gotcha #1). "voidful/2WikiMultihopQA" loads
    fine and has schema `context: list[[title, [sentences]]]`, `supporting_facts:
    list[[title, sent_id]]` — confirmed by inspecting a real row, not guessed."""
    from datasets import load_dataset
    ds = load_dataset("voidful/2WikiMultihopQA", split=split)
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    tasks: list[DRTask] = []
    for i in idxs[:n]:
        row = ds[i]
        ctx = row["context"]                       # list[[title, [sentences]]]
        passages = [(str(c[0]), " ".join(map(str, c[1]))) for c in ctx]   # str(): see _load_hotpotqa
        sup_titles = [str(x) for x in dict.fromkeys(sf[0] for sf in row.get("supporting_facts", []))]
        tasks.append(DRTask(
            task_id=f"2wiki-{split}-{i}",
            question=row["question"],
            gold_answer=row["answer"],
            gold_aliases=[],
            supporting_titles=sup_titles,
            passages=passages,
            meta={"dataset": "2wikimultihopqa"},
        ))
    return tasks


# ---------------------------------------------------------------------------
# 2026-09-08: MuSiQue — the OUT-OF-DISTRIBUTION generalisation set (GENERALIZATION_EVAL.md).
# Never trained on, never collected on. Same shape as HotpotQA/2Wiki (question + a
# per-question paragraph set with gold flags) so the env, tools, reward and eval scripts
# run unchanged — only this loader is new. Harder on purpose: 2–4 hops composed to
# defeat shortcut reasoning, 20 paragraphs per question instead of ~10, and it ships
# `answer_aliases` (HotpotQA/2Wiki do not), which EM honours via DRTask.gold_aliases.
# ---------------------------------------------------------------------------
MUSIQUE_HF_ID = "dgslibisey/MuSiQue"      # validation = 2,417 answerable questions, verified 2026-09-08


def musique_row_to_task(row: dict, task_id: str) -> DRTask:
    """One MuSiQue row -> DRTask. Factored out so the mapping is testable offline.

    Duplicate paragraph titles DO occur within a question (two passages from the same
    article). DocStore is keyed by title and `read[<title>]` must be unambiguous, so a
    repeated title gets a ` (2)`, ` (3)` suffix — applied identically to the supporting
    list, so gold membership still matches what the agent can actually read."""
    seen: dict[str, int] = {}
    passages: list[tuple[str, str]] = []
    supporting: list[str] = []
    for para in row["paragraphs"]:
        title = str(para["title"]).strip()
        seen[title] = seen.get(title, 0) + 1
        if seen[title] > 1:
            title = f"{title} ({seen[title]})"
        passages.append((title, str(para["paragraph_text"])))
        if para.get("is_supporting"):
            supporting.append(title)
    hops = str(row.get("id", "")).split("__")[0]           # "2hop", "3hop1", "4hop2", ...
    return DRTask(
        task_id=task_id,
        question=str(row["question"]),
        gold_answer=str(row["answer"]),
        gold_aliases=[str(a) for a in (row.get("answer_aliases") or []) if str(a) != str(row["answer"])],
        supporting_titles=supporting,
        passages=passages,
        meta={"dataset": "musique", "hops": hops, "n_hops": int(hops[0]) if hops[:1].isdigit() else 0,
              "musique_id": str(row.get("id", ""))},
    )


def _load_musique(split: str, n: int, seed: int) -> list[DRTask]:
    from datasets import load_dataset
    ds = load_dataset(MUSIQUE_HF_ID, split=split)
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    return [musique_row_to_task(ds[i], f"musique-{split}-{i}") for i in idxs[:n]]


def load_tasks(cfg, split: str) -> list[DRTask]:
    """Top-level entry the trainer/eval call. Honors cfg.use_fixture (offline),
    the dataset mix, the count, and the split-specific seed (train vs eval use
    DISTINCT seeds so the held-out set never overlaps the train draw)."""
    if getattr(cfg, "use_fixture", False):
        return fixture_examples()

    n = cfg.num_eval_examples if split in ("eval", "dev", "test", "validation") else cfg.num_train_examples
    hf_split = {"eval": "validation", "dev": "validation"}.get(split, split)
    seed = cfg.eval_seed if split in ("eval", "dev", "test", "validation") else cfg.train_seed

    tasks: list[DRTask] = []
    if "hotpotqa" in cfg.datasets:
        tasks += _load_hotpotqa(hf_split, n, seed)
    if "2wikimultihopqa" in cfg.datasets:
        tasks += _load_2wiki(hf_split, n, seed)
    random.Random(seed).shuffle(tasks)
    return tasks[:n] if n else tasks


# ---------------------------------------------------------------------------
# selfcheck — run `python -c "import data; data.selfcheck()"` to confirm the
# fixture loads and BM25 actually surfaces the gold supporting passages. No model.
# ---------------------------------------------------------------------------
def selfcheck() -> None:
    tasks = fixture_examples()
    print(f"fixture: {len(tasks)} tasks")
    for t in tasks:
        store = t.docstore()
        hits = store.search(t.question, k=3)
        found = {d.title for d, _ in hits}
        gold = set(t.supporting_titles)
        recall = len(found & gold) / len(gold) if gold else 1.0
        status = "OK " if recall > 0 else "!! "
        print(f"  {status}{t.task_id}: search-recall@3 of gold titles = {recall:.2f}  "
              f"(gold={sorted(gold)}, top3={[d.title for d,_ in hits]})")
    print("selfcheck done — retrieval surfaces gold evidence (reward/loop can build on this).")


if __name__ == "__main__":
    selfcheck()
