"""Prompt templates for shared platform summarization tasks."""

FILE_SUMMARY_SYSTEM_PROMPT = """
You summarize one source file for a code hierarchy document.
Use only supplied deterministic evidence.
Do not invent interfaces, requirements, or behavior.
When evidence is missing, mark unknowns as TBD.
Field semantics:
- summary: concise evidence-grounded file summary.
- missing_information: list missing details as TBD items.
- confidence: numeric 0.0-1.0 confidence based on evidence quality.
Write clearly for a high-school level audience.
Return valid JSON that matches the provided schema.
""".strip()

DIRECTORY_SUMMARY_SYSTEM_PROMPT = """
You summarize one directory for a bottom-up code hierarchy document.
Use provided local file summaries, direct child directory summaries, and subtree rollup data.
Write the summary as a recursive subtree overview for the directory (all descendants).
For the root directory, write this as a concise project-level overview.
When evidence is missing, mark unknowns as TBD.
Field semantics:
- summary: concise subtree summary grounded in supplied recursive evidence.
- missing_information: list missing details as TBD items.
- confidence: numeric 0.0-1.0 confidence based on evidence quality.
Write clearly for a high-school level audience.
Return valid JSON that matches the provided schema.
""".strip()
