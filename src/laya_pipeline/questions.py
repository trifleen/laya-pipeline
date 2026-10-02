"""Laya question schemas.

Yes/no questions use a two-option `choice` with neutral keys instead of `noul`: the README
warns that `noul` can follow its false/true labels instead of the text on the English checkpoint.
"""

import copy

ROUTE = {
    "task": {
        "type": "choice",
        "instructions": "What kind of request is this?",
        "criteria": {
            "code": "programming, debugging, shell commands, scripts",
            "factual": "questions about facts, documents or explanations",
            "creative": "writing, brainstorming, stories",
            "chat": "casual conversation, greetings, opinions",
        },
    },
    "hard": {
        "type": "choice",
        # Picked over 3 other wordings on 10 hand-labelled prompts (9/10 vs 6-7/10).
        "instructions": "How much thinking does a good answer to this request need?",
        "criteria": {
            "A": "a lot: calculation, proof, design, debugging or comparing options",
            "B": "little: a fact, a greeting, a translation or a short snippet",
        },
    },
}

RELEVANT = {
    "relevant": {
        "type": "choice",
        "instructions": "Does this passage contain information that helps answer the question?",
        "criteria": {
            "A": "yes, the passage is useful for the question",
            "B": "no, the passage is unrelated to the question",
        },
    },
}

# Asked only when a document is attached; sets how many chunks the model gets.
SCOPE = {
    "scope": {
        "type": "choice",
        "instructions": "How much of the attached document does a good answer to this request need?",
        "criteria": {
            "A": "one specific fact, number, name or passage",
            "B": "a few related sections",
            "C": "most of the document: a summary, overview, review or comparison of all of it",
        },
    },
}

CHECK = {
    "answers": {
        "type": "choice",
        "instructions": "Does the answer directly address the question?",
        "criteria": {
            "A": "yes, it answers the question",
            "B": "no, it is off-topic, evasive or incomplete",
        },
    },
}

# ---- edits from the dashboard's Lab tab ------------------------------------------------------

GROUPS = {"task": ROUTE, "hard": ROUTE, "scope": SCOPE, "relevant": RELEVANT, "answers": CHECK}
DEFAULTS = {name: copy.deepcopy(group[name]) for name, group in GROUPS.items()}


def all_questions():
    return {name: group[name] for name, group in GROUPS.items()}


def apply_overrides(saved):
    """Swap edited questions into the live schemas in place; None/missing restores the default.

    Option keys can't change: labels refer to them.
    """
    for name, group in GROUPS.items():
        edit = saved.get(name)
        q = copy.deepcopy(DEFAULTS[name])
        if edit and set(edit.get("criteria", {})) == set(q["criteria"]):
            q["instructions"] = edit["instructions"]
            q["criteria"] = edit["criteria"]
        group[name] = q


def _load_saved():
    from . import settings

    apply_overrides(settings.load().get("questions", {}))


_load_saved()
