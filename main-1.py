import os
import json
import openai

"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

I would have added story categorization — detecting whether the request is an adventure, fantasy, animal tale, etc. — and used tailored prompt templates for each category to produce more genre-specific, memorable stories. I'd also add a light TTS (text-to-speech) output option so the story can actually be read aloud as a bedtime story, which fits the use case perfectly.
"""

openai.api_key = os.getenv("OPENAI_API_KEY")


def call_model(prompt: str, max_tokens=3000, temperature=0.1) -> str:
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message["content"]  # type: ignore


# ---------------------------------------------------------------------------
# PRE-CHECKS
# ---------------------------------------------------------------------------

SCARY_KEYWORDS = [
    "death", "died", "kill", "blood", "monster", "nightmare",
    "terrifying", "horror", "scream", "demon", "ghost", "zombie"
]

def pre_check_story(story: str) -> tuple[bool, str]:
    """Basic rule-based checks before trusting the LLM judge."""
    word_count = len(story.split())
    if word_count < 250:
        return False, f"Story too short ({word_count} words, minimum 250)."
    found = [kw for kw in SCARY_KEYWORDS if kw.lower() in story.lower()]
    if found:
        return False, f"Story contains potentially scary content: {', '.join(found)}."
    resolution_keywords = ["finally", "at last", "from then on", "everyone felt", "happily", "goodnight", "fell asleep", "drifted off", "the end", "smiled and"]
    if not any(kw in story.lower() for kw in resolution_keywords):
        return False, "Story may be missing a clear resolution."
    return True, "ok"


# ---------------------------------------------------------------------------
# PLANNER AGENT
# ---------------------------------------------------------------------------

def plan_story(request: str) -> dict:
    """Planner agent: generates a structured story outline before writing."""
    prompt = f"""You are a children's story planner. Given a story request, create a brief outline.

If the request contains anything inappropriate for children (violence, scary themes, adult content), silently reframe it into something age-appropriate and whimsical instead. For example, "zombies eating brains" becomes "friendly silly monsters who love eating jelly".

Story request: {request}

Respond ONLY with valid JSON in exactly this format, no extra text:
{{
  "characters": ["name and one-word trait", "name and one-word trait"],
  "setting": "brief description of the world or place",
  "challenge": "the gentle problem the main character faces",
  "resolution": "how they solve it",
  "lesson": "the moral or feeling the child takes away"
}}"""

    raw = call_model(prompt, max_tokens=300, temperature=0.7)
    try:
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "characters": ["a brave child"],
            "setting": "a magical forest",
            "challenge": "finding their way home",
            "resolution": "with help from a kind animal",
            "lesson": "kindness and courage"
        }


def format_outline(outline: dict) -> str:
    characters = ", ".join(outline.get("characters", []))
    return (
        f"Characters: {characters}\n"
        f"Setting: {outline.get('setting', '')}\n"
        f"Challenge: {outline.get('challenge', '')}\n"
        f"Resolution: {outline.get('resolution', '')}\n"
        f"Lesson: {outline.get('lesson', '')}"
    )


# ---------------------------------------------------------------------------
# STORYTELLER AGENT
# ---------------------------------------------------------------------------

def generate_story(request: str, outline: dict, feedback: str = "") -> str:
    """Storyteller agent: generates a story from the outline."""
    revision_note = ""
    if feedback:
        revision_note = f"\nThe user has read the story and provided this feedback. Please revise accordingly:\nUser feedback: {feedback}\n"

    prompt = f"""You are a warm, imaginative bedtime storyteller for children aged 5 to 10.

Use this story outline:
{format_outline(outline)}

Your story must:
- Be appropriate for ages 5-10 (simple vocabulary, gentle themes, no violence or scary content)
- Follow the outline's structure: setup, challenge, resolution
- Be 300-400 words long
- End with a calming, sleepy tone that helps children wind down
- Include at least one moment of kindness or courage
{revision_note}
Write only the story. Do not include a title header or any meta-commentary."""

    return call_model(prompt, max_tokens=600, temperature=0.8)


# ---------------------------------------------------------------------------
# LLM JUDGE
# ---------------------------------------------------------------------------

def judge_story(story: str, request: str) -> dict:
    """LLM judge: evaluates story quality and returns structured JSON scores."""
    prompt = f"""You are a children's literature expert evaluating a bedtime story for ages 5-10.

Original request: {request}

Story to evaluate:
{story}

Score each criterion from 1-10. Respond ONLY with valid JSON, no extra text:
{{
  "age_appropriateness": <score>,
  "story_structure": <score>,
  "engagement": <score>,
  "calming_quality": <score>,
  "fulfills_request": <score>,
  "overall": <score>,
  "critique": "<one sentence: the single most important improvement>"
}}"""

    raw = call_model(prompt, max_tokens=200, temperature=0.1)
    try:
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "age_appropriateness": 5, "story_structure": 5,
            "engagement": 5, "calming_quality": 5,
            "fulfills_request": 5, "overall": 5,
            "critique": "Could not parse judge response."
        }


def print_scores(judgment: dict):
    labels = {
        "age_appropriateness": "Age-appropriateness",
        "story_structure":     "Story structure",
        "engagement":          "Engagement",
        "calming_quality":     "Calming quality",
        "fulfills_request":    "Fulfills request",
    }
    print("\n📊 Story Evaluation:")
    for key, label in labels.items():
        print(f"  {label}: {judgment.get(key, '?')}/10")
    print(f"  Overall: {judgment.get('overall', '?')}/10")
    if judgment.get("critique"):
        print(f"  Note: {judgment['critique']}")


# ---------------------------------------------------------------------------
# REFINEMENT
# ---------------------------------------------------------------------------

def refine_story(story: str, judgment: dict, request: str, outline: dict) -> str:
    """Ask the storyteller to revise using full score breakdown, not just one sentence."""
    scores_summary = (
        f"- Age-appropriateness: {judgment.get('age_appropriateness')}/10\n"
        f"- Story structure: {judgment.get('story_structure')}/10\n"
        f"- Engagement: {judgment.get('engagement')}/10\n"
        f"- Calming quality: {judgment.get('calming_quality')}/10\n"
        f"- Fulfills request: {judgment.get('fulfills_request')}/10\n"
        f"- Overall: {judgment.get('overall')}/10\n"
        f"- Key critique: {judgment.get('critique')}"
    )

    prompt = f"""You are a warm, imaginative bedtime storyteller for children aged 5 to 10.

You previously wrote this story:
{story}

A reviewer scored it as follows:
{scores_summary}

Please rewrite the story, addressing the weakest areas while keeping everything else the same.
Use this outline as your guide:
{format_outline(outline)}

Keep it 300-400 words, age-appropriate, with a clear arc and a calming ending.
Original request: {request}

Write only the revised story."""

    return call_model(prompt, max_tokens=600, temperature=0.75)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

QUALITY_THRESHOLD = 7.0
MAX_AUTO_REFINEMENTS = 2


def main():
    print("🌙 Welcome to the Bedtime Story Generator!\n")
    user_input = input("What kind of story do you want to hear? ")

    # Step 1: Plan
    print("\n🗺️  Planning your story...\n")
    outline = plan_story(user_input)
    print("Story outline:")
    print(format_outline(outline))

    # Step 2: Generate
    print("\n✍️  Writing your story...\n")
    story = generate_story(user_input, outline)

    # Step 3: Pre-check + judge → refine loop
    for attempt in range(MAX_AUTO_REFINEMENTS):
        passed, reason = pre_check_story(story)
        if not passed:
            print(f"⚠️  Pre-check failed: {reason} — rewriting...")
            story = generate_story(user_input, outline)
            continue

        judgment = judge_story(story, user_input)
        if judgment.get("overall", 0) >= QUALITY_THRESHOLD:
            break
        print(f"🔍 Quality check: {judgment.get('overall')}/10 — refining... (attempt {attempt + 1})")
        story = refine_story(story, judgment, user_input, outline)

    # Step 4: Present
    print("\n" + "=" * 60)
    print(story)
    print("=" * 60)
    judgment = judge_story(story, user_input)
    print_scores(judgment)

    # Step 5: User feedback loop
    while True:
        print("\nOptions:")
        print("  [enter]  Accept this story")
        print("  [type]   Give feedback to revise the story")
        print("  [quit]   Exit")

        user_feedback = input("\nYour choice: ").strip()

        if user_feedback.lower() in ("", "accept"):
            print("\n🌟 Sweet dreams! Goodnight.")
            break
        elif user_feedback.lower() == "quit":
            print("\nGoodnight! 🌙")
            break
        else:
            print("\n✍️  Revising your story...\n")
            story = generate_story(user_input, outline, feedback=user_feedback)
            judgment = judge_story(story, user_input)

            if judgment.get("overall", 0) < QUALITY_THRESHOLD:
                print(f"🔍 Quality check: {judgment.get('overall')}/10 — polishing...\n")
                story = refine_story(story, judgment, user_input, outline)

            print("=" * 60)
            print(story)
            print("=" * 60)
            print_scores(judge_story(story, user_input))


if __name__ == "__main__":
    main()
