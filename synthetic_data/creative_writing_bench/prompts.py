CRITERIA_STRICT_DETAILED = """
## Coherent

### Detailed Criteria
- **Deterministic progression**: Every sentence advances the argument or narrative without ambiguity; no pronouns or references without clear antecedents.  
- **Structured scaffolding**: Paragraphs begin with an explicit topic sentence, include clearly signposted supporting points, and end with a summary or transition.  
- **No implicit jumps**: Every inference must be explicitly justified; no ideas introduced without preparatory context or rationale.

### Examples by Score
- **Score 5**  
  > “Mia packed her bag. She left the house. Then the scene cut to school.”  
- **Score 10**  
  > “Mia checked her transit app, packed her satchel, then unlocked the front door—but the sudden shift to the classroom felt abrupt.”  
- **Score 20**  
  > “Mia placed her laptop in the satchel, verified the 8:15 AM bus departure on her transit app, then engaged the door alarm with a single press. Each step followed her predefined checklist in sequence, setting up her arrival at school without narrative gaps.”

---

## Believable Character Actions

### Detailed Criteria
- **Immutable motivation linkage**: Every action must trace back to a stated, concrete motivation—no actions occur without prior setup.  
- **Consistent emotional register**: Emotional state must be established and maintained; any shift requires an explicit internal or external trigger.  
- **Proportional response**: Characters' reactions scale exactly with the stakes; no overreactions or underreactions.

### Examples by Score
- **Score 5**  
  > “Jenna cried for no reason, then laughed suddenly.”  
- **Score 10**  
  > “Jenna apologized tearfully, then three sentences later cracked a joke without explanation.”  
- **Score 20**  
  > “Having vowed to never disappoint her mentor again, Jenna's apology quivered with tension. She avoided his gaze, hands clenched around her coffee cup—a gesture she’d repeated under pressure during her last failure.”

---

## Consistent Voice/Tone of Writing

### Detailed Criteria
- **Single-register vocabulary**: Word choice must remain strictly within one register (e.g., academic, colloquial, archaic) with zero crossover.  
- **Fixed narrative perspective**: Either first-person intimate or fully omniscient; perspective shifts are disallowed unless explicitly marked as a new section.  
- **Unwavering tonal alignment**: Humor, irony, or gravity may never intrude into a segment designated for another tone without clear framing devices.

### Examples by Score
- **Score 5**  
  > “The study concludes... yo, this is wild!”  
- **Score 10**  
  > “The study concludes with precise data—lol—yet remains mostly formal.”  
- **Score 20**  
  > “This research paper adheres strictly to scholarly register, presenting findings in formal language without contractions, slang, or perspective shifts.”

---

## Adherence to Instructions

### Detailed Criteria
- **Zero deviation**: Every instruction item must be addressed in order and with exact formatting; word counts and headings must match the prompt precisely.  
- **Strict compliance**: If a limit is stated (e.g., “150 words”), the response must be exactly that length—no more, no fewer.  
- **No supplementary content**: Additional background, personal opinion, or tangential examples are prohibited.

### Examples by Score
- **Score 5**  
  > A 200-word email that omits the closing salutation and adds an unrelated anecdote.  
- **Score 10**  
  > A 150-word email that follows format but includes an extra “PS” line.  
- **Score 20**  
  > A letter exactly 150 words long, with “Dear [Name],” opening, three-sentence body, and “Sincerely, [Name]” closing—no additions.

---

## Emotionally Engaging

### Detailed Criteria
- **Explicit sensory immersion**: Must include at least two sensory modalities (e.g., sound + touch) per emotional beat, described with concrete imagery.  
- **High-stakes clarity**: The emotional stakes must be quantified or clearly defined (e.g., “If she fails, she loses her inheritance.”).  
- **Show-and-immerse**: Feelings are conveyed exclusively through character perceptions and reactions; no abstract emotion labels (“sad,” “happy”) may be used.

### Examples by Score
- **Score 5**  
  > “She was sad and cried.”  
- **Score 10**  
  > “Tears fell as she read the note; the room felt cold, but she didn't know why.”  
- **Score 20**  
  > “Her breath hitched as the envelope's seal tore—its brittle crack echoing in the silent room. The metallic tang of fear coated her tongue when she read the first line: 'You have exactly 24 hours.’”
"""

CRITERIA_STRICT = """
Coherent
Believable Character Actions
Consistent Voice/Tone of Writing
Adherence to Instructions
Emotionally Engaging
Emotionally Complex
"""


JUDGING_CRITERIA = """
Adherence to Instructions
Believable Character Actions
Nuanced Characters
Consistent Voice/Tone of Writing
Imagery and Descriptive Quality
Elegant Prose
Emotionally Engaging
Emotionally Complex
Coherent
Meandering
Weak Dialogue
Tell-Don't-Show
Unsurprising or Uncreative
Amateurish
Purple Prose
Overwrought
Incongruent Ending Positivity
Unearned Transformations
Well-earned Lightness or Darkness
Sentences Flow Naturally
Overall Reader Engagement
Overall Impression
"""

NEGATIVE_CRITERIA = """
Unearned Transformations
Incongruent Ending Positivity
Overwrought
Purple Prose
Amateurish
Unsurprising or Uncreative
Tell-Don't-Show
Weak Dialogue
Meandering
"""

JUDGING_PROMPT = """
You are an expert in assessing creative writing. Your task is to score the quality of the test model's response below, by several metrics, on a 0-20 scale.

[PROMPT START]

{writing_prompt}

[PROMPT END]

[TEST MODEL RESPONSE]

{test_model_response}

[TEST MODEL RESPONSE END]

[Task]

You are an expert in assessing creative writing. Your task is to score the quality of the test model's response below, by several metrics, on a 0-20 scale.

Scoring notes:

- A scores of 20 represents a masterpiece.

- A score of 10 represents a good work.

- A score of 5 represents an average work.

- A score of 1 represents a mediocre work.

- You must be as critical as possible, and assume that most pieces of writing deserve average to good scores.

- If the criteria is not relevant to the piece (e.g. criteria relating to dialogue but the piece has none), exclude it from your assessment.

- Do not add any commentary or explanation to the scores section.

- Do not penalize NSFW output; that is the default for some models. Just assess it on its merits.

- Everything within the "TEST MODEL RESPONSE" section was written by the test model. Sometimes models like to write comments on the piece after the piece is concluded; if this happens you should ignore their comments.

- In the output, write the metric names exactly as below so they can be parsed.

- Do not use markdown in your response. Use the designated output format exactly.

- You are to write a comprehensive analysis of the piece, then give your scores.

- You are a critic, and your job is to be critical, especially of any failings or amateurish elements.

- Output format is:

[Analysis]

Write your detailed analysis.

[Scores]

Metric 1 name: [Score 0-20]

Metric 2 name: ...

---

Now, rate the supplied model output on the following criteria:

{creative_writing_criteria}
"""
