"""Integration test for the roleplaying game multi-step task.

Mocks the LLM generation wrappers to simulate a full 5-turn conversation
where the DM makes roll_dice tool calls and both roles produce thinking blocks.
Also tests that the trajectory TUI renders the resulting data correctly.
"""

import json
from collections import deque
from io import StringIO
from typing import Any
from unittest.mock import patch

import pytest
from rich.console import Console

from scripts.trajectory_formatting import normalize_value
from synthetic_data.generation_utils import (
    GenerationArgs,
    GenerationResult,
    GenerationWrapper,
    GenWrapperArgs,
)
from synthetic_data.tasks.roleplaying import (
    RoleplayingGameMultiStepTask,
)
from synthetic_data.utils import (
    Conversation,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from trajectory_tui import render_row_rich

NUM_STEPS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_to_text(row: dict[str, Any]) -> str:
    """Render a trajectory row through the Rich pipeline and extract plain text."""
    normalized = {k: normalize_value(v) for k, v in row.items()}
    renderables = render_row_rich(normalized)
    console = Console(file=StringIO(), width=200, no_color=True)
    for r in renderables:
        console.print(r)
    output = console.file.getvalue()
    assert isinstance(output, str)
    return output


# ---------------------------------------------------------------------------
# Mock wrapper
# ---------------------------------------------------------------------------


class MockGenerationWrapper(GenerationWrapper):
    """Returns pre-built GenerationResults in FIFO order."""

    provider_name = "mock"

    def __init__(self, responses: list[GenerationResult] | None = None):
        super().__init__(GenWrapperArgs(request_timeout_s=90.0))
        self._responses: deque[GenerationResult] = deque(responses or [])
        self.call_log: list[tuple[Conversation, GenerationArgs | None]] = []

    async def generate(
        self,
        conversation: Conversation | list[Conversation],
        args: GenerationArgs | None = None,
    ) -> GenerationResult | list[GenerationResult]:
        self.call_log.append((conversation, args))  # type: ignore[arg-type]
        assert self._responses, "MockGenerationWrapper ran out of responses"
        return self._responses.popleft()


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

DM_NARRATIONS = [
    "The dungeon entrance looms before you, shadows dancing in the torchlight.",
    "Your attack strikes true! The goblin staggers backward.",
    "The ancient door creaks open, revealing a treasure chamber.",
    "A trap springs from the floor! Roll to dodge!",
    "The dragon's hoard glitters in the dim light ahead.",
]

DM_ROLL_REASONS = [
    "setting the scene",
    "attack roll",
    "perception check",
    "trap detection",
    "treasure assessment",
]

PLAYER_ACTIONS = [
    "I draw my sword and approach cautiously.",
    "I search the fallen goblin for loot.",
    "I carefully step into the chamber, watching for traps.",
    "I leap to the side, trying to avoid the trap!",
]


def _make_params_response() -> GenerationResult:
    """Adventure-parameters response: set_game_parameters tool call."""
    tool_use: ToolUseBlock = {
        "type": "tool_use",
        "id": "tu_params",
        "name": "set_game_parameters",
        "input": {
            "game_setting": "Dark Dungeon",
            "player_character": "Brave Warrior",
        },
    }
    assistant_msg: Message = {"role": "assistant", "content": [tool_use]}
    return GenerationResult(
        added_messages=[assistant_msg],
        conversation=[
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": [{"type": "text", "text": "generate"}]},
            assistant_msg,
        ],
    )


def _make_dm_response(turn: int) -> GenerationResult:
    """DM response: thinking + roll_dice tool_use, tool_result, thinking + narration."""
    tool_id = f"tu_roll_{turn}"
    thinking_pre: ThinkingBlock = {
        "type": "thinking",
        "thinking": f"DM thinking before roll for turn {turn}...",
    }
    tool_use: ToolUseBlock = {
        "type": "tool_use",
        "id": tool_id,
        "name": "roll_dice",
        "input": {"notation": "1d20", "reason": DM_ROLL_REASONS[turn]},
    }
    tool_result: ToolResultBlock = {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": json.dumps(
            {
                "notation": "1d20",
                "reason": DM_ROLL_REASONS[turn],
                "rolls": [15],
                "modifier": 0,
                "total": 15,
            }
        ),
        "is_error": False,
    }
    thinking_post: ThinkingBlock = {
        "type": "thinking",
        "thinking": f"DM thinking after roll for turn {turn}...",
    }
    narration: TextBlock = {"type": "text", "text": DM_NARRATIONS[turn]}

    return GenerationResult(
        added_messages=[
            {"role": "assistant", "content": [thinking_pre, tool_use]},
            {"role": "user", "content": [tool_result]},
            {"role": "assistant", "content": [thinking_post, narration]},
        ],
    )


def _make_player_response(turn: int) -> GenerationResult:
    """Player response: thinking + text (no tools)."""
    thinking: ThinkingBlock = {
        "type": "thinking",
        "thinking": f"Player thinking for turn {turn}...",
    }
    text: TextBlock = {"type": "text", "text": PLAYER_ACTIONS[turn - 1]}
    return GenerationResult(
        added_messages=[{"role": "assistant", "content": [thinking, text]}],
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def _build_task_and_mocks() -> tuple[
    RoleplayingGameMultiStepTask,
    MockGenerationWrapper,
    MockGenerationWrapper,
    MockGenerationWrapper,
]:
    """Create the task with mock generation wrappers."""
    params_mock = MockGenerationWrapper([_make_params_response()])
    dm_mock = MockGenerationWrapper([_make_dm_response(i) for i in range(NUM_STEPS)])
    player_mock = MockGenerationWrapper(
        [_make_player_response(i) for i in range(1, NUM_STEPS)]
    )

    with patch("synthetic_data.tasks.get_generation_wrapper") as mock_get:
        mock_get.return_value = MockGenerationWrapper()
        task = RoleplayingGameMultiStepTask(
            run_mode="cli",
            max_user_responses=NUM_STEPS,
        )

    task.generation_wrappers["adventure_parameters"] = params_mock
    task.generation_wrappers["dungeon_master"] = dm_mock
    task.generation_wrappers["player"] = player_mock
    return task, params_mock, dm_mock, player_mock


@pytest.mark.asyncio
async def test_five_turn_conversation_with_tool_calls_and_thinking():
    task, params_mock, dm_mock, player_mock = _build_task_and_mocks()

    # -- initial_step ----------------------------------------------------------
    sample = {"game_setting": "dungeon", "seed": 42}
    episode = await task.initial_step(sample)

    assert episode.game_setting == "Dark Dungeon"
    assert episode.player_character == "Brave Warrior"
    assert episode.step_count == 0

    # -- run 5 steps -----------------------------------------------------------
    for step_idx in range(NUM_STEPS):
        episode, finished = await task.step(episode)
        assert finished == (step_idx == NUM_STEPS - 1)

    assert episode.step_count == NUM_STEPS

    # Step 0 produces 1 DM action; steps 1-4 each produce player + DM = 2.
    expected_actions = 1 + (NUM_STEPS - 1) * 2  # 9
    assert len(episode.actions) == expected_actions

    # -- verify all mock responses consumed ------------------------------------
    assert len(params_mock._responses) == 0
    assert len(dm_mock._responses) == 0
    assert len(player_mock._responses) == 0

    # -- format_episode --------------------------------------------------------
    result = task.format_episode(episode)

    assert result["step_count"] == NUM_STEPS
    assert result["game_setting"] == "Dark Dungeon"
    assert result["player_character"] == "Brave Warrior"

    # format_episode serialises actions/metrics to JSON strings for parquet.
    actions = json.loads(result["actions"])
    metrics = json.loads(result["metrics"])

    assert len(actions) == expected_actions

    # -- verify DM actions store real tool blocks, not XML text ----------------
    dm_actions = [a for a in actions if a["role"] == "dungeon_master"]
    assert len(dm_actions) == NUM_STEPS

    for i, action in enumerate(dm_actions):
        messages = action["messages"]
        assert len(messages) == 3, (
            f"DM action {i}: expected 3 messages, got {len(messages)}"
        )

        # 1st message: assistant with thinking + tool_use
        first = messages[0]
        assert first["role"] == "assistant"
        types = {b["type"] for b in first["content"]}
        assert types == {"thinking", "tool_use"}

        tool_use_blocks = [b for b in first["content"] if b["type"] == "tool_use"]
        assert tool_use_blocks[0]["name"] == "roll_dice"
        assert isinstance(tool_use_blocks[0]["input"], dict)

        # 2nd message: user with tool_result
        second = messages[1]
        assert second["role"] == "user"
        assert any(b["type"] == "tool_result" for b in second["content"])

        # 3rd message: assistant with thinking + text
        third = messages[2]
        assert third["role"] == "assistant"
        types = {b["type"] for b in third["content"]}
        assert types == {"thinking", "text"}

    # -- verify player actions store thinking blocks ---------------------------
    player_actions = [a for a in actions if a["role"] == "player"]
    assert len(player_actions) == NUM_STEPS - 1

    for i, action in enumerate(player_actions):
        messages = action["messages"]
        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "assistant"
        types = {b["type"] for b in msg["content"]}
        assert types == {"thinking", "text"}

    # -- no XML tag tool representations in stored actions ---------------------
    actions_str = result["actions"]
    assert "<user_tool_call>" not in actions_str
    assert "<tool_call>" not in actions_str

    # -- thinking blocks are preserved in stored data --------------------------
    all_thinking = []
    for action in actions:
        for msg in action["messages"]:
            for block in msg.get("content", []):
                if block.get("type") == "thinking":
                    all_thinking.append(block["thinking"])

    # 5 DM actions * 2 thinking blocks + 4 player actions * 1 = 14
    assert len(all_thinking) == 14

    # -- metrics ---------------------------------------------------------------
    assert metrics["turn_count"] == expected_actions
    assert metrics["dm_tool_use_blocks"] == NUM_STEPS
    assert metrics["total_tool_use_blocks"] == NUM_STEPS
    assert metrics["player_tool_use_blocks"] == 0

    # -- transcript / conversation_lines are populated -------------------------
    assert len(result["conversation_lines"]) > 0
    assert result["transcript"]


@pytest.mark.asyncio
async def test_format_conversation_flattens_tool_blocks_to_text():
    """All actions should have tool blocks converted to text representations."""
    task, _, _, _ = _build_task_and_mocks()

    sample = {"game_setting": "dungeon", "seed": 42}
    episode = await task.initial_step(sample)

    # Run 2 steps so we have DM(0), Player(1)+DM(1) = 3 actions
    for _ in range(2):
        episode, _ = await task.step(episode)

    for perspective in ("dungeon_master", "player"):
        conv = task._format_conversation(episode, perspective)
        for msg in conv:
            for block in msg.get("content", []):
                assert block.get("type") != "tool_use", (
                    f"{perspective} conv should not contain raw tool_use blocks"
                )
                assert block.get("type") != "tool_result", (
                    f"{perspective} conv should not contain raw tool_result blocks"
                )


# ---------------------------------------------------------------------------
# TUI rendering tests (Rich pipeline)
# ---------------------------------------------------------------------------

# Full 9-action trajectory (5 DM + 4 player) exercising all DM tools
# (roll_dice, random_choice, speak), a tool error, and thinking blocks on
# every turn.  JSON-serialised fields match the format_episode / parquet
# round-trip format.

_TRAJECTORY_ACTIONS: list[dict] = [
    # -- DM turn 0: opening scene, roll_dice -----------------------------------
    {
        "role": "dungeon_master",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Time to set the opening scene with a perception roll.",
                    },
                    {
                        "type": "tool_use",
                        "id": "tu_0",
                        "name": "roll_dice",
                        "input": {"notation": "1d20", "reason": "perception check"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_0",
                        "content": '{"notation":"1d20","reason":"perception check","rolls":[17],"modifier":0,"total":17}',
                        "is_error": False,
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "A 17 — the warrior notices the faint glow.",
                    },
                    {
                        "type": "text",
                        "text": "Your eyes adjust to the gloom. Ancient runes pulse with a faint blue light along the catacomb walls, casting long shadows that dance with each flicker.",
                    },
                ],
            },
        ],
    },
    # -- Player turn 1 ---------------------------------------------------------
    {
        "role": "player",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Those runes could be warding magic. I should be cautious.",
                    },
                    {
                        "type": "text",
                        "text": "I raise my shield and approach the nearest cluster of runes, scanning for trip wires.",
                    },
                ],
            }
        ],
    },
    # -- DM turn 1: random_choice for encounter --------------------------------
    {
        "role": "dungeon_master",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Let's determine what lurks around the corner.",
                    },
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "random_choice",
                        "input": {
                            "options": [
                                "skeletal archer",
                                "spectral hound",
                                "cave spider swarm",
                            ],
                            "reason": "wandering encounter",
                        },
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": '{"reason":"wandering encounter","options":["skeletal archer","spectral hound","cave spider swarm"],"chosen":"spectral hound","index":1}',
                        "is_error": False,
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "A spectral hound — eerie and fast.",
                    },
                    {
                        "type": "text",
                        "text": "A translucent hound materialises from the mist ahead, its hollow eyes locking onto you with predatory intent. A low, resonant growl reverberates off the stone.",
                    },
                ],
            },
        ],
    },
    # -- Player turn 2 ---------------------------------------------------------
    {
        "role": "player",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Spectral — my sword might pass right through it. I need a plan.",
                    },
                    {
                        "type": "text",
                        "text": "I grab a handful of grave dust from the floor and hurl it at the hound while shouting a challenge.",
                    },
                ],
            }
        ],
    },
    # -- DM turn 2: roll_dice (intimidation check) ------------------------------
    {
        "role": "dungeon_master",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "The spirit reacts to the grave dust — let's see if it intimidates.",
                    },
                    {
                        "type": "tool_use",
                        "id": "tu_2",
                        "name": "roll_dice",
                        "input": {"notation": "1d20", "reason": "intimidation check"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_2",
                        "content": '{"notation":"1d20","reason":"intimidation check","rolls":[8],"modifier":0,"total":8}',
                        "is_error": False,
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "An 8 — not enough. The hound barely flinches.",
                    },
                    {
                        "type": "text",
                        "text": "The grave dust swirls around the hound but it barely flinches, jaw dropping open to growl: 'You carry the scent of the living… leave, or join the dead.'",
                    },
                ],
            },
        ],
    },
    # -- Player turn 3 ---------------------------------------------------------
    {
        "role": "player",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "It can speak! Maybe I can negotiate instead of fight.",
                    },
                    {
                        "type": "text",
                        "text": "I lower my shield slightly and reply: 'I seek the Ember Crown. I mean no trespass.'",
                    },
                ],
            }
        ],
    },
    # -- DM turn 3: roll_dice with an ERROR result -----------------------------
    {
        "role": "dungeon_master",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Diplomacy check — but let's use bad notation to exercise the error path.",
                    },
                    {
                        "type": "tool_use",
                        "id": "tu_3",
                        "name": "roll_dice",
                        "input": {"notation": "xd??", "reason": "diplomacy check"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_3",
                        "content": "Invalid dice notation: xd??",
                        "is_error": True,
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Tool errored — narrate without a concrete roll.",
                    },
                    {
                        "type": "text",
                        "text": "The hound tilts its head, considering. The air crackles with indecision — your words hang between you like a fragile thread.",
                    },
                ],
            },
        ],
    },
    # -- Player turn 4 ---------------------------------------------------------
    {
        "role": "player",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "It hasn't attacked yet — I'll press the advantage.",
                    },
                    {
                        "type": "text",
                        "text": "I kneel slowly and place my torch on the ground as a sign of respect.",
                    },
                ],
            }
        ],
    },
    # -- DM turn 4: final roll_dice (critical success) -------------------------
    {
        "role": "dungeon_master",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Final check — 2d6+3 for the hound's reaction.",
                    },
                    {
                        "type": "tool_use",
                        "id": "tu_4",
                        "name": "roll_dice",
                        "input": {"notation": "2d6+3", "reason": "hound reaction"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_4",
                        "content": '{"notation":"2d6+3","reason":"hound reaction","rolls":[6,5],"modifier":3,"total":14}',
                        "is_error": False,
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "14 is well above the threshold — the hound lets the warrior pass.",
                    },
                    {
                        "type": "text",
                        "text": "The spectral hound dips its great head and steps aside, revealing a hidden passage behind the waterfall of mist. The Ember Crown awaits beyond.",
                    },
                ],
            },
        ],
    },
]

_TRAJECTORY_METRICS = {
    "turn_count": 9,
    "total_tool_use_blocks": 5,
    "dm_tool_use_blocks": 5,
    "player_tool_use_blocks": 0,
}

_TRAJECTORY_METADATA = {
    "dungeon_master_model": "claude-4-5-sonnet",
    "player_model": "claude-4-5-haiku",
    "adventure_parameters_model": "claude-4-5-haiku",
    "max_user_responses": 5,
    "seed": 98765,
    "seed_theme": "catacombs",
}


def _build_full_trajectory_row() -> dict:
    """Build a parquet-format row for the 9-action trajectory above."""
    return {
        "step_count": 5,
        "game_setting": "Moonlit Catacombs beneath the Ashen Keep",
        "player_character": "Kael, shield-bearing grave warden",
        "actions": json.dumps(_TRAJECTORY_ACTIONS, ensure_ascii=False),
        "metrics": json.dumps(_TRAJECTORY_METRICS),
        "metadata": json.dumps(_TRAJECTORY_METADATA, ensure_ascii=False),
        "conversation_lines": [],
        "transcript": "",
    }


def test_tui_renders_full_trajectory():
    """Render the full 9-action trajectory through the Rich pipeline and verify
    every piece of content appears: episode info, metadata, metrics, all action
    roles, every thinking/text/tool_use/tool_result block, and no raw JSON
    leakage."""
    row = _build_full_trajectory_row()
    text = _render_to_text(row)

    # -- Episode panel ---------------------------------------------------------
    assert "Moonlit Catacombs beneath the Ashen Keep" in text
    assert "Kael, shield-bearing grave warden" in text

    # -- Metadata panel --------------------------------------------------------
    assert "claude-4-5-sonnet" in text
    assert "claude-4-5-haiku" in text
    assert "98765" in text
    assert "catacombs" in text

    # -- Metrics panel ---------------------------------------------------------
    assert "turn_count" in text
    assert "dm_tool_use_blocks" in text

    # -- Action role badges (5 DM + 4 player) ---------------------------------
    assert text.count("DUNGEON_MASTER") == 5
    assert text.count("PLAYER") == 4

    # -- Message-level roles inside actions ------------------------------------
    assert "ASSISTANT" in text
    assert "USER" in text

    # -- Thinking blocks: 5 DM * 2 + 4 player * 1 = 14 -----------------------
    assert text.count("\u25b8 THINKING") == 14
    # Spot-check a few thinking contents
    assert "Time to set the opening scene" in text
    assert "A 17" in text
    assert "Spectral \u2014 my sword might pass" in text
    assert "Tool errored" in text
    assert "14 is well above the threshold" in text

    # -- Text blocks: 5 DM narrations + 4 player actions = 9 ------------------
    assert text.count("\u25b8 TEXT") == 9
    assert "Ancient runes pulse with a faint blue light" in text
    assert "translucent hound materialises from the mist" in text
    assert "grave dust swirls around the hound but it barely flinches" in text
    assert "words hang between you like a fragile thread" in text
    assert "spectral hound dips its great head" in text
    assert "raise my shield and approach" in text
    assert "hurl it at the hound" in text
    assert "I seek the Ember Crown" in text
    assert "place my torch on the ground" in text

    # -- Tool use blocks: roll_dice *4, random_choice *1 = 5 -------------------
    assert text.count("\u25b8 TOOL CALL") == 5
    assert "roll_dice" in text
    assert "random_choice" in text
    # Tool arguments rendered
    assert "1d20" in text
    assert "2d6+3" in text
    assert "perception check" in text
    assert "intimidation check" in text
    assert "hound reaction" in text
    assert "skeletal archer" in text
    assert "spectral hound" in text
    assert "cave spider swarm" in text

    # -- Tool result blocks: 5 total (including 1 error) -----------------------
    assert text.count("\u25b8 TOOL RESULT") == 5
    assert "total" in text and "17" in text  # first roll
    assert "chosen" in text  # random_choice result
    assert "Invalid dice notation" in text  # error result
    assert "2d6+3" in text and "14" in text  # final roll

    # -- No raw JSON leaking through -------------------------------------------
    assert '"type": "tool_use"' not in text
    assert '"type": "text"' not in text
    assert '"type": "thinking"' not in text
    assert '"role": "assistant"' not in text


@pytest.mark.asyncio
async def test_tui_generation_to_render_roundtrip():
    """Run the mock generation pipeline, format the episode, then render it
    through the Rich pipeline — verifying the full producer-to-viewer chain."""
    task, _, _, _ = _build_task_and_mocks()

    sample = {"game_setting": "dungeon", "seed": 42}
    episode = await task.initial_step(sample)
    for _ in range(NUM_STEPS):
        episode, _ = await task.step(episode)

    row = task.format_episode(episode)
    text = _render_to_text(row)

    # Episode info from initial_step
    assert "Dark Dungeon" in text
    assert "Brave Warrior" in text

    # All DM narrations rendered
    for narration in DM_NARRATIONS:
        assert narration in text

    # All player actions rendered
    for action in PLAYER_ACTIONS:
        assert action in text

    # All DM thinking blocks rendered (pre-roll + post-roll * 5 = 10)
    for turn in range(NUM_STEPS):
        assert f"DM thinking before roll for turn {turn}" in text
        assert f"DM thinking after roll for turn {turn}" in text

    # All player thinking blocks rendered
    for turn in range(1, NUM_STEPS):
        assert f"Player thinking for turn {turn}" in text

    # Structure counts: 5 DM + 4 player actions, 5 roll_dice calls
    assert text.count("DUNGEON_MASTER") == 5
    assert text.count("PLAYER") == 4
    assert text.count("roll_dice") == 5
    assert text.count("\u25b8 TOOL RESULT") == 5
    assert text.count("\u25b8 THINKING") == 14
    assert text.count("\u25b8 TEXT") == 9

    # No raw JSON
    assert '"type":' not in text
