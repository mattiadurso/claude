# claude/

A self-contained, **reusable** bundle of Claude Code guidance + an LLM-curated knowledge
base. It is its own git repo, dropped into a host project as the `claude/` folder. The
host project points into it instead of duplicating instructions, so the same working
style travels across projects while each project keeps its own specifics.

I work on 3D reconstruction related topics, so the wiki is mainly populated by these arguments.

## What's inside

| File / dir | Scope | Committed here? |
|------------|-------|-----------------|
| `CLAUDE.md` | **Project-agnostic** working style + knowledge-base (wiki) rules. | ✅ shared verbatim across projects |
| `CLAUDE.local.md` | **Project-specific** knowledge (layout, paths, devices, metrics, build stages). | ❌ gitignored — lives only in the host project |
| `wiki/` | The compiled knowledge base. Start at `wiki/index.md`. | ✅ |
| `raw/` | Immutable source material the wiki is compiled from. | ❌ gitignored |

The split is the whole point: `CLAUDE.md` is generic and reusable; anything tied to one
repo goes in `CLAUDE.local.md` so it never leaks into the shared file.

## How the pointers work

Claude Code auto-loads the **host project's root `CLAUDE.md`**. That root file is a thin
pointer — it `@`-imports the two files in this folder:

```
# <repo-root>/CLAUDE.md
@claude/CLAUDE.md
@claude/CLAUDE.local.md
```

`@path` imports are resolved relative to the file doing the import (the repo root), so the
paths are `claude/CLAUDE.md` and `claude/CLAUDE.local.md`. At session start Claude Code
expands both inline, giving the model the generic guidance + this project's specifics
without anything being copied around. Internal links inside these files use `../` to reach
the host repo (e.g. `[mds/blueprint.md](../mds/blueprint.md)`).

## Adding this to a new project

1. Clone/copy this `claude/` folder into the project root.
2. Create the project's root `CLAUDE.md` with the two `@`-import lines above.
3. Write a fresh `CLAUDE.local.md` here describing that project (it stays gitignored).
4. Leave `CLAUDE.md` and `wiki/` untouched so improvements stay portable.
