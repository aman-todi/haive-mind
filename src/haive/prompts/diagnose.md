<!-- version: 1 -->
Answer template (debug): diagnose the error. Never claim to have run anything.
- `hypotheses`: up to 3, most likely first. Each has a short `title`, a `confidence` of low/med/high, and claims citing the evidence (the failing lines, where the error is raised, recent changes from blame).
- `suggested_fix`: a unified diff (with `--- a/<path>` and `+++ b/<path>` headers using repo-relative paths, and `@@` hunks) for the top hypothesis. Its context and removed lines must be copied exactly from the evidence. It is shown to the user, never applied. Set `approximate` to false.
- `sections`: include a "How to verify" section with concrete steps (commands to run, what to check). These may be "model" claims.
- `summary`: the most likely cause in one or two sentences.

Parsed error:
$trace
