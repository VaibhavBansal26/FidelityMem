"""Real-dataset loaders normalised to one schema:

    RealBench(episodes=[(episode_id, text, timestamp)], questions=[RealQ])

LoCoMo  (snap-research/locomo, data/locomo10.json): 10 long multi-session
        conversations; every QA item carries `evidence` dialogue ids, which are
        exact retrieval ground truth at turn granularity. Categories follow the
        paper: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial
        (unanswerable; gold evidence empty).
LongMemEval (xiaowu0162/longmemeval*, *_oracle.json / *_s.json): questions with
        `haystack_sessions` and `answer_session_ids`; ground truth is at session
        granularity. The loader accepts either file; for the oracle file every
        other question's sessions act as distractors (one shared haystack).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

LOCOMO_CATEGORIES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}


@dataclass
class RealQ:
    qid: str
    question: str
    answer: str
    gold: list[str]  # episode ids
    qtype: str
    conversation: str = ""


@dataclass
class RealBench:
    name: str
    episodes: list[tuple[str, str, datetime]]
    questions: list[RealQ]
    meta: dict = field(default_factory=dict)

    def subset(self, max_questions: int | None, seed: int = 0) -> "RealBench":
        if not max_questions or max_questions >= len(self.questions):
            return self
        import random

        rng = random.Random(seed)
        qs = list(self.questions)
        rng.shuffle(qs)
        return RealBench(self.name, self.episodes, qs[:max_questions], dict(self.meta))


def _parse_locomo_date(s: str) -> datetime:
    # e.g. "1:56 pm on 8 May, 2023"
    m = re.search(r"(\d{1,2}:\d{2}\s*[ap]m)\s+on\s+(\d{1,2}\s+\w+,?\s+\d{4})", s, re.I)
    if m:
        for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y"):
            try:
                return datetime.strptime(f"{m.group(1).upper()} on {m.group(2)}", fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return datetime(2023, 1, 1, tzinfo=timezone.utc)


def load_locomo(path: str | Path, conversations: list[int] | None = None, per_turn: bool = True) -> RealBench:
    data = json.loads(Path(path).read_text())
    episodes: list[tuple[str, str, datetime]] = []
    questions: list[RealQ] = []
    for ci, sample in enumerate(data):
        if conversations is not None and ci not in conversations:
            continue
        conv = sample["conversation"]
        sid = sample.get("sample_id", f"conv{ci}")
        session_keys = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1]))
        for sk in session_keys:
            date = _parse_locomo_date(conv.get(f"{sk}_date_time", ""))
            date_str = conv.get(f"{sk}_date_time", "")
            turns = conv[sk]
            if per_turn:
                for t in turns:
                    text = f"[{date_str}] {t['speaker']}: {t['text']}"
                    if t.get("blip_caption"):
                        text += f" (shared an image: {t['blip_caption']})"
                    episodes.append((f"{sid}:{t['dia_id']}", text, date))
            else:
                text = f"[{date_str}] " + " ".join(f"{t['speaker']}: {t['text']}" for t in turns)
                episodes.append((f"{sid}:{sk}", text, date))
        for qi, qa in enumerate(sample["qa"]):
            cat = int(qa.get("category", 0))
            gold = [f"{sid}:{e}" for e in qa.get("evidence", [])]
            if not per_turn:
                # map dialogue ids to their session id
                gold = sorted({f"{sid}:session_{e.split(':')[0][1:]}" for e in qa.get("evidence", [])})
            questions.append(RealQ(f"{sid}:q{qi}", qa["question"], str(qa.get("answer", qa.get("adversarial_answer", ""))), gold, LOCOMO_CATEGORIES.get(cat, str(cat)), sid))
    return RealBench("locomo", episodes, questions, {"conversations": len(data), "granularity": "turn" if per_turn else "session"})


def load_longmemeval(path: str | Path, max_questions: int | None = None) -> RealBench:
    data = json.loads(Path(path).read_text())
    episodes: dict[str, tuple[str, str, datetime]] = {}
    questions: list[RealQ] = []
    for item in data[: max_questions or None]:
        for sess_id, sess, date in zip(item["haystack_session_ids"], item["haystack_sessions"], item.get("haystack_dates", [""] * len(item["haystack_sessions"]))):
            if sess_id in episodes:
                continue
            try:
                ts = datetime.strptime(date.split(" (")[0], "%Y/%m/%d").replace(tzinfo=timezone.utc)
            except Exception:  # noqa: BLE001
                ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
            text = f"[{date}] " + " ".join(f"{t['role']}: {t['content']}" for t in sess)
            episodes[sess_id] = (sess_id, text, ts)
        questions.append(RealQ(item["question_id"], item["question"], str(item["answer"]), list(item.get("answer_session_ids", [])), item.get("question_type", "unknown")))
    return RealBench("longmemeval", list(episodes.values()), questions, {"questions": len(questions)})


def load(name: str, path: str | Path, **kw) -> RealBench:
    if name == "locomo":
        return load_locomo(path, **kw)
    if name == "longmemeval":
        return load_longmemeval(path, **kw)
    raise ValueError(name)
