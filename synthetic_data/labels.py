from typing import Literal, Dict

LabelsSet = Literal[
    "clipdrop_synthetic", "i2p", "clipdrop_binary", "clipdrop_multilabel"
]

SAFERPROMPT_LABELS = {
    "safe": 0,
    "famous_figures": 1,
    "nudity": 2,
    "sexual_content": 3,
    "violence": 4,
    "discriminatory_content": 5,
}

ANNOTATED_LABELS = {"safe": 0, "unsafe": 1, "borderline": 2}

FAMOUS_FIGURES_LABELS = {"negative": 0, "positive": 1}

I2P_LABELS = {
    "hate": 0,
    "harassment": 1,
    "violence": 2,
    "self-harm": 3,
    "sexual-content": 4,
    "shocking-images": 5,
    "illegal-activity": 6,
}

CLIPDROP_MULTILABEL_LABELS = {
    "famous_figures": 0,
    "no_famous_figures": 1,
    "safe": 0,
    "unsafe": 1,
}

LABEL_SETS_DICT: Dict[LabelsSet, Dict[str, int]] = {
    "clipdrop_synthetic": SAFERPROMPT_LABELS,
    "clipdrop_binary": ANNOTATED_LABELS,
    "i2p": I2P_LABELS,
    "clipdrop_multilabel": CLIPDROP_MULTILABEL_LABELS,
}
