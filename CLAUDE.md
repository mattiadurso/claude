# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Style (Karpathy Skills)

Behavioral guidelines to reduce common LLM coding mistakes. **Tradeoff:** these bias toward caution over speed. For trivial tasks, use judgment.

### 0. Code Style
- Functions should stay short and single-purpose.
- Prefer explicit imports over wildcard imports.
- Use Google guideline style for python. Including linting and ruff.
- Ignore IDEAS.md files. Don't load them, those are for the humans.

### 1. Read Before You Write

**Read the codebase before changing it. Read, don't skim.**

- Read the files you're about to touch before editing them.
- Copy patterns that already exist; check imports to see what the project actually depends on (don't reach for `axios` where everything is `fetch`).
- When you can't find a pattern, ask instead of guessing.

### 2. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.
- Plausible-looking filler is exactly the code that passes a casual review and fails when it matters — ask rather than fill the gap.
- ..
- When asked to implement a speed improvement, first run a test on current code to keep as baseline, then add new code, finally compare and report.
- Keep md files consistent with modifications over development.

### 3. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- Re-use code if possible. This includes checking other scripts, starting from the helpers folder. Eventually propose to modify an existing helper function rather than creating a brand new similar one.
- If you write 200 lines and it could be 50, rewrite it.
- Torch code:
  - use Dataset class to manage data laoding
  - use Dataloder class to manage parallelism/muli multiworker. Dont invent threadpools for this, it's not worthy. 

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 4. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request. If a line is there because "while I was in there," revert it.

### 5. Verification

**The gap between code that works and code you think works is testing.**

- When fixing a bug, write the failing test first, watch it fail, then fix it — that's the only proof you fixed the cause, not the symptom.
- Test behavior that can actually break, not that a constructor sets a field.
- If something is hard to test, that's information about the design, not permission to skip it.

### 6. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

- Keep tests in the project's designated test directory (see CLAUDE.local.md).

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

### 7. Debugging

**Investigate; don't guess.**

- Read the whole error and the stack trace.
- Reproduce the problem before you change anything, and change one thing at a time.
- Don't paper over an unexpected `null` with a null check — find out why it's null, or the bug just moves somewhere quieter.

### 8. Dependencies

**Every dependency is permanent code you don't control.**

- Before adding one, ask whether the project or the standard library already does it (`crypto.randomUUID()` over a `uuid` package).
- When you do add one, say why — so the choice is visible rather than smuggled into the manifest.

### 9. Communication

**Say what you did and why, not just a block of code.**

- Flag concerns even when you did exactly what was asked.
- Be precise about uncertainty: "I'm not sure this library supports streaming" tells the user what to verify; "I think this should work" does not.

### 10. Common Failure Modes

Patterns that recur often enough to name — catch yourself in any and stop, don't push through:
- **Kitchen Sink** — restructuring half the codebase while you're at it.
- **Wrong Abstraction** — abstracting before copy-pasting twice.
- **Optimistic Path** — handling the happy path and ignoring the 500.
- **Runaway Refactor** — a fix that cascades across files.

### Commits

When making git commits, always use:
  git -c user.name="mattiadurso" -c user.email="mattiadurso98@gmail.com" commit ...

No co-authors or other contributors are allowed. 

### Local & Project-Specific Knowledge
- This file is **project-agnostic** — reuse it across repos unchanged.
- Anything tied to a specific repo (what the project is, directory layout, file paths,
  dataset/metric names, device/GPU conventions, build stages, numeric invariants)
  belongs in **CLAUDE.local.md**, not here.
- Import CLAUDE.local.md if present.

## Knowledge Base (Wiki)

Rules for a self-maintaining, LLM-curated knowledge base. **Model:** the human owns judgment and the raw record; the model owns the bookkeeping; the wiki is a compiled artifact that compounds, not a pile that grows.

### 1. Sources Are Immutable
- Everything saved lands in `raw/` and is never edited after it lands (articles, transcripts, PDFs, screenshots).
- If a source is wrong, add a correcting source — don't rewrite history.
- Hand-editing `raw/` creates two systems of record and no way to tell which is true.

### 2. Separate the Layers
- Three layers, three owners: `raw/` (immutable sources, yours), `wiki/` (generated pages, the model's), and a schema file (`CLAUDE.md`/`AGENTS.md`, both).
- Don't blur them — model writing into `raw/`, or you hand-tuning `wiki/`, breaks the boundaries that make the system trustworthy.

### 3. The Model Owns the Wiki
- You choose what enters `raw/`, ask questions, and think.
- The model summarizes, cross-references, files under the right entry, and updates neighbors when something new arrives.
- If you're doing the bookkeeping, the schema is underspecified — not the model.

### 4. Compile, Don't Retrieve
- This is not RAG. Sources are compiled once into structured, linked pages; questions are answered from that built artifact.
- Analogy: `raw/` is source code, the model is the compiler, `wiki/` is the executable, queries are runtime.
- Compiled knowledge compounds; retrieved knowledge is rediscovered every query.

### 5. Ingest One Source at a Time
- Drop a single file into `raw/` and ingest it.
- A good ingest traces the implications across the graph, touching every page the new fact changes — not just one new page.
- Batch-importing everything in a weekend produces a dump, not a wiki, because nothing gets linked while the job is still finishing.

### 6. Link Everything
- Every page connects to others through wikilinks; every wikilink is a visible edge in the graph (this is why Obsidian is the front-end of choice).
- An entity that appears in five pages but links to none means the ingest was lazy.
- The value is in the edges, not the nodes.

### 7. Navigate by Index
- Reach an answer by reading `index.md`, following the few relevant pages, and synthesizing — not by loading the whole vault into context.
- A hundred articles stays fast if the index is honest.
- Brute-forcing the corpus on every question means the index has stopped reflecting the territory and needs a pass.

### 8. Lint the Knowledge
- Treat the wiki like code: find contradictions between pages, surface low-confidence claims, list orphan pages, flag entities that drifted into two spellings.
- A contradiction is information, not an error to paper over — usually two sources disagree, and that's exactly where to look.
- Skipping the lint is how a wiki quietly rots while the graph still looks impressive.

### 9. Start Small
- Begin with ten sources, not ten thousand. Let ingest, query, and lint feel natural before adding a search engine, elaborate frontmatter, or twenty rules.
- The first few ingests need supervision; naming conventions will change and early pages will be messy — that's normal.
- A small wiki you actually feed beats a beautiful architecture you abandon in week three.

## Repo Related

Project-specific knowledge for the current repository (what the project is, key design
ideas, directory layout, device selection, build stages, metric invariants, test
locations) lives in **CLAUDE.local.md** — see that file. Keep this file generic.
