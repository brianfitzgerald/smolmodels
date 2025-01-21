from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Literal, Optional, List


DialogueType = Literal["scene_heading", "action", "dialogue", "transition"]


@dataclass
class DialogueLine:
    character: str
    content: str
    parenthetical: Optional[str] = None
    line_number: int = 0


@dataclass
class SceneElement:
    type: DialogueType
    content: str
    line_number: int
    dialogue_data: Optional[DialogueLine] = None  # Used only for dialogue elements


@dataclass
class Scene:
    heading: SceneElement
    elements: List[SceneElement]


class ScreenplayParser:
    def __init__(self, text: str):
        self.lines = text.split("\n")
        self.current_line = 0
        self.total_lines = len(self.lines)
        self.scenes: List[Scene] = []
        self.character_line_counts = defaultdict(int)

    def parse(self) -> List[Scene]:
        """Parse the entire screenplay and return a list of scenes."""
        # Skip any leading blank lines or title page content
        while self.current_line < self.total_lines:
            if self._is_scene_heading(self.current_line):
                self._parse_scene()
            self.current_line += 1
        return self.scenes

    def _is_scene_heading(self, line_num: int) -> bool:
        """Check if the current line is a scene heading."""
        line = self.lines[line_num].strip()
        return bool(re.match(r"^(INT\.|EXT\.).+", line))

    def _is_character_name(self, line_num: int) -> bool:
        """Check if the current line is a character name."""
        line = self.lines[line_num].strip()
        return bool(
            line
            and line.isupper()
            and not line.startswith("(")
            and not self._is_scene_heading(line_num)
            and not self._is_transition(line_num)
        )

    def _is_parenthetical(self, line_num: int) -> bool:
        """Check if the current line is a parenthetical."""
        line = self.lines[line_num].strip()
        return line.startswith("(") and line.endswith(")")

    def _is_transition(self, line_num: int) -> bool:
        """Check if the current line is a transition."""
        line = self.lines[line_num].strip()
        return bool(re.match(r"^(FADE|DISSOLVE|CUT).+", line))

    def _merge_consecutive_elements(
        self, elements: List[SceneElement]
    ) -> List[SceneElement]:
        """Merge consecutive elements of the same type."""
        if not elements:
            return elements

        merged = []
        current = elements[0]
        current_lines = [current.content]
        current_start_line = current.line_number

        for next_elem in elements[1:]:
            if (
                next_elem.type == current.type
                and next_elem.line_number == current_start_line + len(current_lines)
            ):
                current_lines.append(next_elem.content)
            else:
                merged.append(
                    SceneElement(
                        type=current.type,
                        content=" ".join(current_lines),
                        line_number=current_start_line,
                        dialogue_data=current.dialogue_data,
                    )
                )
                current = next_elem
                current_lines = [current.content]
                current_start_line = current.line_number

        # Don't forget to add the last group
        merged.append(
            SceneElement(
                type=current.type,
                content=" ".join(current_lines),
                line_number=current_start_line,
                dialogue_data=current.dialogue_data,
            )
        )

        return merged

    def _parse_dialogue_block(self) -> SceneElement:
        """Parse a complete dialogue block including character name, parentheticals, and dialogue."""
        character_line = self.current_line
        character = self.lines[self.current_line].strip()
        self.current_line += 1

        parenthetical = None
        dialogue_lines = []

        while self.current_line < self.total_lines:
            line = self.lines[self.current_line].strip()

            if not line:
                break

            if self._is_parenthetical(self.current_line):
                parenthetical = line
            else:
                dialogue_lines.append(line)

            self.current_line += 1

            # Check next line - if blank or new character/scene heading, end dialogue block
            if self.current_line >= self.total_lines:
                break
            next_line = self.lines[self.current_line].strip()
            if (
                not next_line
                or self._is_character_name(self.current_line)
                or self._is_scene_heading(self.current_line)
            ):
                break

        dialogue_content = " ".join(dialogue_lines)

        return SceneElement(
            type="dialogue",
            content=dialogue_content,
            line_number=character_line,
            dialogue_data=DialogueLine(
                character=character,
                content=dialogue_content,
                parenthetical=parenthetical,
                line_number=character_line,
            ),
        )

    def _parse_scene(self):
        """Parse a single scene."""
        # Parse scene heading
        heading = SceneElement(
            type="scene_heading",
            content=self.lines[self.current_line].strip(),
            line_number=self.current_line,
        )

        elements = []
        self.current_line += 1

        # Parse scene content until we hit the next scene heading or end of script
        while self.current_line < self.total_lines:
            if self._is_scene_heading(self.current_line):
                break

            line = self.lines[self.current_line].strip()

            # Skip blank lines
            if not line:
                self.current_line += 1
                continue

            if self._is_character_name(self.current_line):
                dialogue_element = self._parse_dialogue_block()
                assert dialogue_element.type == "dialogue"
                assert dialogue_element.dialogue_data is not None
                character_name = dialogue_element.dialogue_data.character
                self.character_line_counts[character_name] += 1
                elements.append(dialogue_element)
            elif self._is_transition(self.current_line):
                elements.append(
                    SceneElement(
                        type="transition", content=line, line_number=self.current_line
                    )
                )
                self.current_line += 1
            else:
                # Assume it's action description
                elements.append(
                    SceneElement(
                        type="action", content=line, line_number=self.current_line
                    )
                )
                self.current_line += 1

        # Merge consecutive elements before adding to scene
        elements = self._merge_consecutive_elements(elements)
        self.scenes.append(Scene(heading=heading, elements=elements))

    @staticmethod
    def format_conversation(scene: Scene) -> str:
        """Format a scene's dialogue as a conversation."""
        conversation = []
        for element in scene.elements:
            if element.type == "dialogue":
                assert element.dialogue_data is not None
                character = element.dialogue_data.character
                line = element.dialogue_data.content
                if element.dialogue_data.parenthetical:
                    line = f"{element.dialogue_data.parenthetical}\n{line}"
                conversation.append(f"{character}: {line}")
        return "\n".join(conversation)
