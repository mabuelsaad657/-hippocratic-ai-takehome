# Hippocratic AI - Bedtime Story Generator

A multi-agent bedtime story generator for children ages 5–10, built with GPT-3.5-turbo.

## How it works

1. **Planner** — generates a structured story outline (characters, setting, challenge, resolution, lesson) from the user's request. Automatically reframes inappropriate requests into age-appropriate ones.
2. **Storyteller** — writes a 300–400 word story from the outline with a 3-act arc and calming bedtime tone.
3. **LLM Judge** — scores the story on 5 criteria (age-appropriateness, structure, engagement, calming quality, fulfills request) and returns structured feedback.
4. **Quality gate** — if the overall score is below 7.0/10, the story is automatically refined using the full score breakdown. Up to 2 refinement passes.
5. **User feedback loop** — after reading, the user can accept, quit, or request changes.

## Setup

```bash
export OPENAI_API_KEY=your_key_here
python main.py
```

## Files
- `main.py` — full pipeline
- `DIAGRAM.svg` — system block diagram
