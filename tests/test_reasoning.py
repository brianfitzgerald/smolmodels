from model.reasoning import (
    parse_groups,
    hard_group_reward,
    soft_group_reward,
    group_size_reward,
    ConnectionsDataModule,
)


FIRST_SAMPLE = {
    "groups": [
        {"reason": "defeat badly", "words": ["crush", "rout", "shellac", "trash"]},
        {"reason": "tops", "words": ["cami", "halter", "tank", "tee"]},
        {"reason": "butt", "words": ["bottom", "buns", "seat", "tail"]},
        {"reason": "kinds of whales", "words": ["blue", "fin", "gray", "right"]},
    ],
    "words": [
        "crush",
        "rout",
        "shellac",
        "trash",
        "cami",
        "halter",
        "tank",
        "tee",
        "bottom",
        "buns",
        "seat",
        "tail",
        "blue",
        "fin",
        "gray",
        "right",
    ],
    "prompt": [
        {
            "content": "\nYou are an expert puzzle solving model.\nFind groups of words that are related to each other. Each group is four words long. There are exactly four groups in total.\nYou may only use each word in one group.\nRespond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n<group>\n...\n</group>\n<group>\n...\n</group>\n</answer>\n\n# Example\n\nUser: apple, orange, banana, pear, corolla, charger,\nAssistant: <reasoning>\nThe first group are all fruits.\nThe second group are all cars.\n</reasoning>\n<answer>\n<group>apple, orange, banana, pear</group>\n<group>corolla, charger</group>\n</answer>\n\n# Example\n\nUser: dog, cat, red, white,\nAssistant: <reasoning>\nThe first group are all animals.\nThe second group are all colors.\n</reasoning>\n<answer>\n<group>dog, cat</group>\n<group>red, white</group>\n</answer>\n",
            "role": "system",
        },
        {
            "content": "crush, rout, shellac, trash, cami, halter, tank, tee, bottom, buns, seat, tail, blue, fin, gray, right",
            "role": "user",
        },
    ],
    "answer": "<answer><group>crush, rout, shellac, trash</group>\n<group>cami, halter, tank, tee</group>\n<group>bottom, buns, seat, tail</group>\n<group>blue, fin, gray, right</group></answer>",
    "answer_groups": [
        ["crush", "rout", "shellac", "trash"],
        ["cami", "halter", "tank", "tee"],
        ["bottom", "buns", "seat", "tail"],
        ["blue", "fin", "gray", "right"],
    ],
}

ANSWERS_BATCH = [FIRST_SAMPLE["answer_groups"]]


def test_parse_groups_single():
    input_str = "<group>apple, banana, cherry</group>"
    expected = [["apple", "banana", "cherry"]]
    assert parse_groups(input_str) == expected


def test_parse_groups_multiple():
    input_str = "<group>apple, banana</group> some text <group>cherry, date</group>"
    expected = [["apple", "banana"], ["cherry", "date"]]
    assert parse_groups(input_str) == expected


def test_parse_groups_whitespace():
    input_str = "<group>  apple  ,  banana  </group>"
    expected = [["apple", "banana"]]
    assert parse_groups(input_str) == expected


def test_parse_groups_empty_string():
    input_str = ""
    expected = []
    assert parse_groups(input_str) == expected


def test_parse_groups_empty_group():
    input_str = "<group></group>"
    expected = [[]]
    assert parse_groups(input_str) == expected


def test_connections_reward_func_correct():
    completion = [[{"content": FIRST_SAMPLE["answer"]}]]
    hard_score = hard_group_reward(
        FIRST_SAMPLE["prompt"], completion, answer_groups=ANSWERS_BATCH
    )
    soft_score = soft_group_reward(
        FIRST_SAMPLE["prompt"], completion, answer_groups=ANSWERS_BATCH
    )
    assert [hard_score[0], soft_score[0]] == [4.0, 1.0]


def test_soft_reward_values():
    completions = [
        "<answer><group>crush, rout, shellac, trash</group></answer>",
        "<answer><group>crush, rout, shellac</group></answer>",
        "<answer><group>rout, shellac</group></answer>",
        "<answer><group>rout</group></answer>",
    ]
    scores = []
    expected_scores = [1.0, 0.75, 0.5, 0.25]
    for c in completions:
        score = soft_group_reward("", [[{"content": c}]], answer_groups=ANSWERS_BATCH)
        scores.append(score[0])
    assert scores == expected_scores


def test_soft_reward_values_multiple_groups():
    completions = [
        "<answer><group>crush, rout, shellac, trash</group>\n<group>cami, halter, tank, tee</group>\n<group>bottom, buns, seat, tail</group>\n<group>blue, fin, gray, right</group></answer>",
        "<answer><group>crush, rout, shellac, trash</group><group>cami, halter, tank, tee</group></answer>",
        "<answer><group>crush, rout, shellac, trash</group></answer>",
    ]
    scores = []
    expected_scores = [1.0, 1.0, 1.0]
    for c in completions:
        score = soft_group_reward("", [[{"content": c}]], answer_groups=ANSWERS_BATCH)
        scores.append(score[0])
    assert scores == expected_scores


def test_hard_reward_values():
    completions = [
        "<answer><group>crush, rout, shellac, trash</group>\n<group>cami, halter, tank, tee</group>\n<group>bottom, buns, seat, tail</group>\n<group>blue, fin, gray, right</group></answer>",
        "<answer><group>crush, rout, shellac, trash</group><group>cami, halter, tank, tee</group></answer>",
        "<answer><group>crush, rout, shellac, trash</group></answer>",
    ]
    scores = []
    for c in completions:
        score = hard_group_reward("", [[{"content": c}]], answer_groups=ANSWERS_BATCH)
        scores.append(score[0])
    assert scores == [4.0, 2.0, 1.0]


def test_group_size_rewards():
    completions = [
        "<answer><group>crush, rout, shellac, trash</group><group>crush, rout, shellac, trash</group><group>crush, rout, shellac, trash</group><group>crush, rout, shellac, trash</group></answer>",
        "<answer><group>crush, rout, shellac</group><group>crush, rout, shellac, rout</group></answer>",
        "<answer><group>crush, rout, shellac</group></answer>",
        "<answer><group>rout, shellac</group></answer>",
        "<answer><group>rout</group></answer>",
    ]
    out = []
    for c in completions:
        out.append(
            group_size_reward("", [[{"content": c}]], answer_groups=ANSWERS_BATCH)
        )
    assert out == [[0.25], [0.125], [0.0], [0.0], [0.0]]
