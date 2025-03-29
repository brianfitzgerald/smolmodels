from model.reasoning import parse_groups, connections_soft_group_reward_func


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
    scores = connections_soft_group_reward_func(prompts, completions, answer=answer)
    # Expecting a score of 2.0.
    assert scores == [2.0]


def test_connections_reward_func_incorrect():
    # Define the correct answer groups.
    answer = [["apple", "banana"], ["cherry", "date"]]
    # Create a completion with a non-matching group.
    content = "<group>apple, grape</group>"
    completions = [[{"content": content}]]
    prompts = []  # Not used in the function.
    scores = connections_soft_group_reward_func(prompts, completions, answer=answer)
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
    scores = connections_soft_group_reward_func(prompts, completions, answer=answer)
    # First completion should yield 1.0 and second should yield 2.0.
    assert scores == [1.0, 2.0]
