# Status line — how it works and how to add a field

Notes for the Claude Code status line on this machine. Lives next to the script it documents
(`~/.claude/`), not in any repo — nothing here is project-specific.

## The two files

| file | role |
|---|---|
| `~/.claude/settings.json` | wires it up: `statusLine.command` + `refreshInterval` |
| `~/.claude/statusline-command.sh` | the renderer — reads session JSON on stdin, prints **one line** to stdout |

```json
"statusLine": {
  "type": "command",
  "command": "sh ~/.claude/statusline-command.sh",
  "refreshInterval": 10
}
```

Current output:

```
📁 folder  (branch)   Opus 5 (xhigh)  context [██████░░░░] 64%  used tokens: 32%  resets 19:20
```

## How the script is built

Claude Code pipes a JSON blob to the command on every refresh. Rather than shell out to `jq`
seven times (and `jq` isn't installed here anyway), one `python3` call reads the JSON and
prints `KEY=value` lines with `shlex.quote`, which the shell `eval`s into variables:

```sh
eval "$(printf '%s' "$input" | /usr/bin/python3 -c '...
fields = {
    "model":  g("model", "display_name"),
    "effort": g("effort", "level"),
    ...
}
for k, v in fields.items():
    print("%s=%s" % (k, shlex.quote(str(v))))
' 2>/dev/null)"
```

`g(*path)` walks the dict and returns `""` for any missing or null key, so **every optional
field degrades to an empty string** — that's what makes the `[ -n "$x" ] && printf ...` guards
at the bottom work.

## Recipe: add a field to the line

1. **Confirm the field exists in the payload.** The schema is documented inside the CLI binary,
   so it's always current for the installed version:
   ```sh
   grep -a -n -B60 -A70 'context_window_size": number' \
     ~/.local/share/claude/versions/<version>   # e.g. 2.1.223
   ```
2. **Add one line to the `fields` dict** in the python block: `"name": g("path", "to", "key"),`.
3. **Print it** in the assembly section at the bottom, guarded by `[ -n "$name" ]`.
4. **Test without restarting Claude** (see below).

Worked example — model effort level, added 2026-08-06:

```sh
# in fields:
"effort": g("effort", "level"),

# in the assembly section, replacing the old one-line model printf:
if [ -n "$model" ]; then
  if [ -n "$effort" ]; then
    printf "  ${MAGENTA} %s (%s)${RESET}" "$model" "$effort"
  else
    printf "  ${MAGENTA} %s${RESET}" "$model"
  fi
fi
```

The if/else rather than a bare `%s (%s)` is deliberate: `effort` is **only present when the
current model supports reasoning effort**, so a model without one must still render cleanly.

## Testing

Feed it a payload by hand — no restart, no waiting for a refresh. Always test the field
*present*, *absent*, and the empty-payload case:

```sh
echo '{"cwd":"/home/mattia/Desktop/Repos/cvpr2027","model":{"display_name":"Opus 5"},
       "effort":{"level":"xhigh"},"context_window":{"used_percentage":64.2},
       "rate_limits":{"five_hour":{"used_percentage":31.7,"resets_at":1785000000}}}' \
  | sh ~/.claude/statusline-command.sh

echo '{"model":{"display_name":"Haiku 4.5"},"context_window":{"used_percentage":12}}' \
  | sh ~/.claude/statusline-command.sh

echo '{}' | sh ~/.claude/statusline-command.sh
```

## Payload reference

Top level: `session_id`, `session_name`, `prompt_id`, `transcript_path`, `cwd`, `version`.

| path | notes |
|---|---|
| `model.id`, `model.display_name` | e.g. `claude-opus-5`, `Opus 5` |
| `effort.level` | **optional** — `low`/`medium`/`high`/`xhigh`/`max`; live session value, tracks `/effort`, not the `effortLevel` default in settings.json |
| `thinking.enabled` | bool |
| `output_style.name` | |
| `workspace.current_dir`, `.project_dir`, `.added_dirs`, `.git_worktree` | |
| `workspace.repo.{host,owner,name}` | **optional** — from the origin remote |
| `context_window.used_percentage` / `.remaining_percentage` | pre-computed 0–100, **null until the first message** |
| `context_window.total_input_tokens` / `.total_output_tokens` / `.context_window_size` | input count includes cache reads/writes |
| `context_window.current_usage.{input,output,cache_creation_input,cache_read_input}_tokens` | null if no messages yet |
| `rate_limits.five_hour.{used_percentage,resets_at}` | **optional** — subscribers only, and only after the first API response; `resets_at` is unix epoch seconds |
| `rate_limits.seven_day.{used_percentage,resets_at}` | same, weekly limit — not currently shown |
| `vim.mode` | **optional** — only when vim mode is on |
| `agent.{name,type}` | **optional** — only under `--agent` |
| `pr.{number,url,review_state}` | **optional** — open PR for the branch |
| `worktree.{name,path,branch,original_cwd,original_branch}` | **optional** — only in a `--worktree` session |

## Gotchas

- **`sh`, not bash.** The command runs under `sh`; no arrays, no `[[ ]]`.
- **Dynamic text goes through `printf` `%s`, never the format string.** A `%` in a folder or
  branch name would otherwise be read as a format specifier.
- **Git calls use `--no-optional-locks`** so the status line can't contend with a real git
  command in the same repo.
- **Optional fields are genuinely absent**, not empty — guard every one, or a stray separator
  shows up on the line.
- `session_tokens` is computed but not printed; add `%s` + `"$session_tokens"` to the printf
  to surface it.
- There is a `statusline-setup` agent for this; it's not required, but it's the sanctioned path
  if a change gets fiddly.
