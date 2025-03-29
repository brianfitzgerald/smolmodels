from model.reasoning import (
    parse_groups,
    connections_soft_group_reward_func,
    connections_hard_group_reward_func,
    score_connections,
    score_connections_soft,
)


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


# Tests for connections_reward_func
def test_connections_reward_func_correct():
    # Define the correct answer groups.
    answer = [["apple", "banana"], ["cherry", "date"]]
    # Create one completion that correctly matches both groups.
    content = "<group>apple, banana</group><group>cherry, date</group>"
    completions = [[{"content": content}]]
    prompts = []  # Not used in the function.
    scores = connections_hard_group_reward_func(prompts, completions, answer=answer)
    # Expecting a score of 2.0.
    assert scores == [2.0]


def test_connections_reward_func_incorrect():
    # Define the correct answer groups.
    answer = [["apple", "banana"], ["cherry", "date"]]
    # Create a completion with a non-matching group.
    content = "<group>apple, grape</group>"
    completions = [[{"content": content}]]
    prompts = []  # Not used in the function.
    scores = connections_hard_group_reward_func(prompts, completions, answer=answer)
    # No submitted group matches any answer group, so score should be 0.0.
    assert scores == [0.0]


def test_connections_reward_func_multiple_completions():
    # Define the correct answer groups.
    answer = [["apple", "banana"], ["cherry", "date"]]
    # First completion: only one matching group.
    content1 = "<group>apple, banana</group>"
    # Second completion: both groups match.
    content2 = "<group>apple, banana</group><group>cherry, date</group>"
    completions = [[{"content": content1}], [{"content": content2}]]
    prompts = []  # Not used in the function.
    scores = connections_hard_group_reward_func(prompts, completions, answer=answer)
    # First completion should yield 1.0 and second should yield 2.0.
    assert scores == [1.0, 2.0]


def test_connections_soft_reward_func_multiple_completions():
    # Define the correct answer groups.
    answer = [["apple", "banana"], ["cherry", "date"]]
    # First completion: only one matching group.
    content1 = "<group>apple, banana</group>"
    # Second completion: both groups match.
    content2 = "<group>apple</group><group>cherry</group>"
    completions = [[{"content": content1}], [{"content": content2}]]
    prompts = []  # Not used in the function.
    scores = connections_soft_group_reward_func(prompts, completions, answer=answer)
    # First completion should yield 1.0 and second should yield 2.0.
    assert scores == [0.5, 0.5]


def test_no_groups():
    # Both solution and submitted groups are empty.
    assert score_connections_soft([], []) == 0.0


def test_full_match():
    # Each submitted group exactly matches a solution group.
    solution_groups = [["a", "b"], ["c", "d"]]
    submitted_groups = [["a", "b"], ["c", "d"]]
    # Each group is fully matched: 2 items * 0.25 = 0.5 per group; total = 1.0.
    assert score_connections_soft(solution_groups, submitted_groups) == 1.0


def test_partial_match():
    # One solution group with four items, and submitted groups partially match.
    solution_groups = [["a", "b", "c", "d"]]
    submitted_groups = [["a", "b"], ["c", "e"]]
    # The best match is from the first submitted group with 2 items matching: 2 * 0.25 = 0.5.
    assert score_connections_soft(solution_groups, submitted_groups) == 0.5


def test_multiple_solution_groups():
    # Two solution groups with overlapping submitted groups.
    solution_groups = [["a", "b", "c"], ["d", "e"]]
    submitted_groups = [["a", "b"], ["c", "d", "f"], ["e"]]
    # For the first group: best match is ['a','b'] (2 matches → 0.5).
    # For the second group: best match is either ['c', 'd', 'f'] or ['e'] (1 match → 0.25).
    # Total score = 0.5 + 0.25 = 0.75.
    assert score_connections_soft(solution_groups, submitted_groups) == 0.75


def test_overlapping_submissions():
    # The same submitted group is used for multiple solution groups.
    solution_groups = [["a", "b"], ["a", "c"]]
    submitted_groups = [["a"]]
    # For each solution group, only 'a' is matched, so each group contributes 0.25.
    # Total score = 0.25 + 0.25 = 0.5.
    assert score_connections_soft(solution_groups, submitted_groups) == 0.5
