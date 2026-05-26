# L2P CLI — Command-Line Interface

> PDDL generation, validation, planning, and LLM agent tooling - all from your terminal.

```
l2p/cli/
├── main.py               # Entry point & argument dispatch
├── commands/
│   ├── init.py           # l2p init — model configuration
│   ├── models.py         # l2p models — list, switch, test
│   ├── generate.py       # l2p generate — interactive pipelines
│   ├── generators/
│   │   ├── domain.py     # Interactive domain generator
│   │   └── problem.py    # Interactive problem generator
│   ├── set.py            # l2p set — inject components from JSON
│   ├── build.py          # l2p build — assemble domain/problem PDDL
│   ├── validate.py       # l2p validate — validate components & files
│   ├── plan.py           # l2p plan — run planners
│   ├── schema.py         # l2p schema — output Pydantic JSON schemas
│   ├── config.py         # l2p config — show, edit, reset
│   ├── templates.py      # l2p templates — list, show, find
│   ├── new.py            # l2p new — blank PDDL files
│   ├── chat.py           # l2p chat — interactive LLM session
│   └── mcp.py            # l2p mcp — MCP server entry point
├── mcp/
│   └── server.py         # MCP server implementation
└── utils/
    ├── config.py         # ConfigManager (YAML-based)
    ├── errors.py         # CLIError with troubleshooting
    ├── helpers.py        # Terminal styling, input helpers
    └── templates.py      # Template discovery
```

---

## Installation

```bash
pip install l2p

# For LLM backends:
pip install openai tiktoken        # OpenAI SDK backend
pip install llm tiktoken           # Unified backend (simonw/llm)

# For MCP support:
pip install l2p[mcp]
```

---

## Quick Reference

| Command | Description | Audience |
|---------|-------------|----------|
| `l2p init` | Configure LLM provider & model | Human + Agent |
| `l2p set <component>` | Inject a PDDL component from JSON | **Agent** |
| `l2p build domain` | Assemble & render full domain PDDL | **Agent** |
| `l2p build problem` | Assemble & render full problem PDDL | **Agent** |
| `l2p validate <component>` | Validate JSON component against rules | **Agent** |
| `l2p validate domain <file>` | Validate a `.pddl` domain file | **Agent** |
| `l2p validate problem <file>` | Validate a `.pddl` problem file | **Agent** |
| `l2p plan` | Run planner on domain/problem strings | **Agent** |
| `l2p schema <component>` | Output JSON Schema for LLM reference | **Agent** |
| `l2p generate domain` | Interactive domain generation | Human |
| `l2p generate problem` | Interactive problem generation | Human |
| `l2p models list` | List available models | Human + Agent |
| `l2p models test` | Test model connection | Human + Agent |
| `l2p config show` | Display configuration | Human + Agent |
| `l2p new` | Create blank PDDL file | Human + Agent |
| `l2p chat` | Interactive LLM chat | Human |
| `l2p mcp` | Start MCP server for agent frameworks | **Agent** |

---

## Agentic Workflow (non-interactive commands)

These commands accept structured JSON input and produce machine-readable output — designed for LLM tool-calling agents and automation scripts.

### `l2p set` — Inject & validate a component

Inject individual PDDL components from JSON. Each call validates the data against L2P's semantic rules and optionally outputs the formatted result.

```bash
# Inject types
l2p set types --data '[{"name":"block","parent":"object"}]'

# Inject predicates, output PDDL-formatted
l2p set predicates --data '[
  {"name":"clear","params":[{"variable":"?x","type":"block"}]}
]' --pddl

# Inject from file, output validated JSON
l2p set actions --file actions.json --json

# Read from stdin (for piping)
l2p set types --data '[...]' --json | l2p set predicates --stdin

# Show the JSON Schema an LLM should follow
l2p set types --schema
```

Available components: `requirements`, `types`, `constants`, `predicates`, `functions`, `derived-predicates`, `actions`, `durative-actions`, `events`, `processes`, `constraints`, `objects`, `initial-state`, `goal-state`, `metric`.

### `l2p build` — Assemble and render PDDL

Build the final PDDL string from a complete `DomainDetails` or `ProblemDetails` JSON, or from individual component files.

```bash
# Full JSON — one-shot domain
l2p build domain --data '{"name":"bw","types":[...],"predicates":[...],"actions":[...]}' -o domain.pddl

# Full JSON — one-shot problem
l2p build problem --data '{"name":"pb1","domain_name":"bw","objects":[...],"initial_state":{...},"goal_state":{...}}' -o problem.pddl

# Individual component files
l2p build domain --name blocksworld \
  --types types.json --predicates preds.json \
  --actions actions.json -o domain.pddl

# Output the assembled DomainDetails as JSON (not PDDL)
l2p build domain --data '{...}' --json
```

### `l2p validate` — Semantic & syntax checking

Validate individual JSON components or entire `.pddl` files against L2P's rule engine (naming conventions, type inheritance, variable scope, arity matching, undeclared symbols).

```bash
# Validate a component JSON snippet
l2p validate types --data '[{"name":"block","parent":"object"}]'

# Validate a .pddl domain file (parses PDDL → Pydantic → checks rules)
l2p validate domain domain.pddl

# Validate a .pddl problem file
l2p validate problem problem.pddl

# Validate against JSON
l2p validate domain --data '{"name":"bw","types":[],"predicates":[],"actions":[]}'
```

### `l2p plan` — Run a planner

Execute FastDownward or Unified Planning on domain/problem PDDL strings directly (no temp files needed).

```bash
# Run FastDownward with raw PDDL strings
l2p plan \
  --domain '(define (domain test) ...)' \
  --problem '(define (problem p) ...)' \
  --planner fast-downward --alias lama-first

# Read from files
l2p plan --domain @domain.pddl --problem @problem.pddl --json

# Use Unified Planning backend
l2p plan --domain @d.pddl --problem @p.pddl \
  --planner unified --engine aries --json
```

Output with `--json` returns a structured `PlanningResult`:

```json
{
  "is_successful": true,
  "plan": ["(pickup b1)", "(stack b1 b2)"],
  "error_message": null,
  "raw_output": "...",
  "metrics": {}
}
```

### `l2p schema` — JSON Schema reference for LLMs

Output the Pydantic JSON Schema for any PDDL component. LLMs read this to know the exact JSON structure expected by `l2p set` and `l2p build`.

```bash
# Schema for a single component
l2p schema types
l2p schema predicates
l2p schema action
l2p schema domain
l2p schema problem

# Include example JSON
l2p schema domain --examples

# List all available schemas
l2p schema list
```

### End-to-end agent workflow

```bash
# 1. Configure
l2p init --backend openai --provider openai --model gpt-4o-mini

# 2. Get the JSON schema the LLM should follow
l2p schema domain --examples

# 3. Build domain
l2p build domain --data '{
  "name":"blocksworld","types":[{"name":"block","parent":"object"}],
  "predicates":[...],"actions":[...]
}' -o domain.pddl

# 4. Validate
l2p validate domain domain.pddl

# 5. Build problem
l2p build problem --data '{
  "name":"pb1","domain_name":"blocksworld",
  "objects":[{"name":"b1","type":"block"}],
  "initial_state":{"facts":["(on-table b1)"]},
  "goal_state":{"conditions":["(holding b1)"]}
}' -o problem.pddl

# 6. Plan
l2p plan --domain @domain.pddl --problem @problem.pddl --json
```

---

## MCP Server

L2P implements the [Model Context Protocol](https://modelcontextprotocol.io), allowing any MCP-compatible AI application (Claude Desktop, Claude Code, VS Code extensions) to use L2P as a native tool.

### Setup

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "l2p": {
      "command": "l2p",
      "args": ["mcp"]
    }
  }
}
```

For **Claude Desktop**, this goes in `claude_desktop_config.json`. For **Claude Code**, add to `.claude/settings.json` or the project-level MCP config.

### Exposed Tools

| Tool | Description |
|------|-------------|
| `validate_component` | Validate a PDDL JSON component against L2P rules |
| `format_component` | Parse JSON and output PDDL-formatted component |
| `build_domain` | Build a complete PDDL domain from DomainDetails JSON |
| `build_problem` | Build a complete PDDL problem from ProblemDetails JSON |
| `run_planner` | Execute FastDownward or Unified Planning planner |
| `get_schema` | Get JSON Schema for any PDDL component |

### Exposed Resources

| Resource | Description |
|----------|-------------|
| `schema://{component}` | JSON Schema for the specified component |

### Example (Claude Desktop)

Once configured, you can ask Claude:

> "Create a blocksworld domain with stack and unstack actions, validate it, generate a problem with 2 blocks, and run the planner."

Claude will call the MCP tools in sequence: `get_schema` → `build_domain` → `validate_component` → `build_problem` → `run_planner`, with full results returned at each step.

### Usage without MCP (stdio)

The MCP server uses stdio transport. For testing:

```bash
# Start the server (it will wait for JSON-RPC messages on stdin)
l2p mcp
```

---

## Interactive Commands (for humans)

### Setup: `l2p init`

```bash
# Interactive
l2p init

# Non-interactive
l2p init --backend openai --provider openai --model gpt-4o-mini
```

Configuration is stored at `~/.l2p/config.yaml`.

### PDDL Generation: `l2p generate`

Interactive pipelines for human-driven domain and problem generation:

```bash
# Step-by-step domain generation
l2p generate domain

# Step-by-step problem generation
l2p generate problem

# Increase LLM retries
l2p generate domain --max-retries 5
```

Each component (types, predicates, actions, etc.) can be generated by the LLM, entered manually, or reviewed and fixed before proceeding.

### Model Management: `l2p models`

```bash
l2p models list                    # List configured models
l2p models list --details          # Show context length, cost, params
l2p models list --provider openai  # Filter by provider
l2p models switch                  # Interactively switch models
l2p models test                    # Test connection
l2p models test --simple           # Config validation only
```

### Configuration: `l2p config`

```bash
l2p config show          # Display config
l2p config show --raw    # Raw YAML
l2p config edit          # Open in $EDITOR
l2p config validate      # Validate settings
l2p config reset --force # Reset to defaults
```

### Create Blank Files: `l2p new`

```bash
l2p new blocksworld.pddl
l2p new pb1.pddl --type problem --domain-name blocksworld
```

### Interactive Chat: `l2p chat`

```bash
l2p chat
```

REPL-style session with `/edit`, `/validate`, `/exit` commands.

---

## Project Structure

```
l2p/cli/
├── main.py               # Entry point & argument dispatch
├── commands/
│   ├── set.py            # l2p set <component> — JSON injection
│   ├── build.py          # l2p build domain|problem — PDDL assembly
│   ├── validate.py       # l2p validate — PDDL file & JSON validation
│   ├── plan.py           # l2p plan — planner execution
│   ├── schema.py         # l2p schema — JSON Schema output
│   ├── mcp.py            # l2p mcp — MCP server entry
│   ├── ...               # Existing interactive commands
├── mcp/
│   └── server.py         # MCP server implementation
└── utils/
    └── ...
```
