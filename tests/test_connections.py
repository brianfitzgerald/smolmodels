from model.reasoning import (
    parse_groups,
    connections_soft_group_reward_func,
    connections_hard_group_reward_func,
    score_connections_soft,
    group_size_reward_func,
    n_groups_reward_func,
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


def test_connections_soft_group_reward_func():
    # Test the soft reward function.
    prompts = []  # Not used by the function.
    # Provide completions that will result in groups:
    # "a,b,c,d" -> splits into 4 items -> default dummy returns 1.0.
    # "high" -> splits into ["high"] -> dummy returns 3.0.
    # "low" -> splits into ["low"] -> dummy returns -1.0.
    completions = ["a,b,c,d", "high", "low"]
    kwargs = {"answer": "dummy"}

    result = connections_soft_group_reward_func(prompts, completions, **kwargs)
    # Expected computation:
    # For "a,b,c,d": 1.0 * 2 = 2.0.
    # For "high": 3.0 * 2 = 6.0, clamped to 5.0.
    # For "low": (-1.0) * 2 = -2.0, clamped to 0.0.
    expected = [2.0, 5.0, 0.0]
    assert result == expected


def test_connections_hard_group_reward_func(capsys):
    # Test the hard reward function.
    prompts = []
    # Using similar completions as before.
    completions = ["a,b,c,d", "high", "low"]
    kwargs = {"answer": "dummy"}

    result = connections_hard_group_reward_func(prompts, completions, **kwargs)
    # Expected:
    # "a,b,c,d" -> splits into 4 items -> default dummy returns 0.2 -> 0.2 * 5 = 1.0.
    # "high" -> ["high"] -> returns 1.0 -> 1.0 * 5 = 5.0.
    # "low" -> ["low"] -> returns -0.5 -> -0.5 * 5 = -2.5, clamped to 0.0.
    expected = [1.0, 5.0, 0.0]
    assert result == expected


def test_group_size_reward_func():
    # Test the group size reward function.
    prompts = []
    # Provide completions that yield groups of different sizes.
    # "a,b,c,d" splits to 4 items (reward 0.5).
    # "one,two,three" splits to 3 items (reward 0.0).
    # "x,y,z,w,v" splits to 5 items (reward 0.0).
    completions = ["a,b,c,d", "one,two,three", "x,y,z,w,v"]
    kwargs = {}

    result = group_size_reward_func(prompts, completions, **kwargs)
    expected = [0.5, 0.0, 0.0]
    assert result == expected


def test_n_groups_reward_func():
    # Test the n groups reward function.
    prompts = []
    # Provide completions:
    # "a,b,c,d" -> 4 items -> reward 0.5.
    # "1,2,3" -> 3 items -> reward 0.0.
    # "x,y,z,w" -> 4 items -> reward 0.5.
    completions = ["a,b,c,d", "1,2,3", "x,y,z,w"]
    kwargs = {}

    result = n_groups_reward_func(prompts, completions, **kwargs)
    expected = [0.5, 0.0, 0.5]
    assert result == expected
