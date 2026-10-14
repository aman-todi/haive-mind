<!-- version: 1 -->
You check whether the retrieved evidence is enough to answer a developer's question about their code.

Question ($intent): $question

Evidence (abbreviated):
$evidence

Pick one status:
- sufficient: the evidence answers the question.
- missing_symbol: a specific function/class/variable that matters is referenced but its definition is not shown. List the names in `missing`.
- need_context: the right area is found but more is needed (a caller, a config file, an env var's use, the rest of a file). List what to fetch in `missing` (identifiers, file paths, or path:line).
- need_external: the answer depends on third-party library/framework behavior not in the repos. Put a web search query in `missing`.
- nothing_relevant: the evidence is unrelated to the question.
Be decisive: prefer "sufficient" when the core of the question is covered.
