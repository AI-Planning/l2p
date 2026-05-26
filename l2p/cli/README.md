# L2P CLI — Command-Line Interface

> Interactive PDDL generation, model management, and utilities — all from your terminal.

```
l2p/ cli/
├── main.py               # Entry point & argument dispatch
├── commands/
│   ├── init.py           # l2p init — model configuration setup
│   ├── generate.py       # l2p generate — PDDL generation pipeline
│   ├── generators/
│   │   ├── domain.py     # Interactive domain generator
│   │   └── problem.py    # Interactive problem generator
│   ├── models.py         # l2p models — list, switch, test models
│   ├── config.py         # l2p config — show, edit, reset, validate
│   ├── templates.py      # l2p templates — list, show, find
│   ├── new.py            # l2p new — create blank PDDL files
│   └── chat.py           # l2p chat — interactive LLM session
└── utils/
    ├── config.py         # ConfigManager (YAML-based)
    ├── errors.py         # CLIError with troubleshooting tips
    ├── helpers.py        # Terminal styling, diff display, input helpers
    └── templates.py      # TemplateManager for prompt file discovery
```

---

## Installation

Ensure the CLI is installed:

```bash
pip install l2p
```

For LLM backends, you may need extra packages:

```bash
# OpenAI SDK backend
pip install openai tiktoken

# Unified backend (simonw/llm)
pip install llm tiktoken

# Ollama
pip install llm && llm install llm-ollama
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `l2p init` | Interactive LLM configuration (provider, model, API key) |
| `l2p models list` | List available models for a provider |
| `l2p models switch` | Interactively select a different model |
| `l2p models test` | Test connection to the configured model |
| `l2p generate domain` | Interactive PDDL domain generation pipeline |
| `l2p generate problem` | Interactive PDDL problem generation pipeline |
| `l2p config show` | Display current configuration |
| `l2p config edit` | Open configuration in `$EDITOR` |
| `l2p config validate` | Validate configuration settings |
| `l2p config reset` | Reset configuration to defaults |
| `l2p templates list` | List available prompt templates |
| `l2p templates show` | Show template content |
| `l2p new` | Create a blank PDDL domain/problem file |
| `l2p chat` | Interactive chat with the configured LLM |

---

## Setup: `l2p init`

Configure your LLM provider and model interactively or non-interactively:

```bash
# Interactive mode (recommended)
l2p init

# Non-interactive
l2p init --backend openai --provider openai --model gpt-4o-mini
l2p init --backend unified --provider ollama --model llama3.1:8b

# Force overwrite existing config
l2p init --force
```

Both backends are supported:
- **unified** — uses [simonw/llm](https://llm.datasette.io) (supports OpenAI, Google, Anthropic, Ollama, and 30+ more via plugins)
- **openai** — uses OpenAI SDK directly (for OpenAI-compatible APIs)

Configuration is stored at `~/.l2p/config.yaml`.

---

## PDDL Generation: `l2p generate`

### Domain Generation

```bash
l2p generate domain
```

Interactive pipeline that guides you through generating a complete PDDL domain:

```
━━━ L2P Interactive Domain Generator ━━━
  Type /exit at any prompt to quit

Domain name: blocksworld
Domain description: Blocksworld with stacking

─── Types ───
  Include types? (Y/n): y
  LLM generates types... Is this correct? (y/N): y

─── Predicates ───
  Include predicates? (Y/n): y
  LLM generates predicates... Is this correct? (y/N): y

─── Actions ───
  1 - LLM extracts action names
  2 - You type action names
  Choice: 1
  LLM extracted: pickup, putdown, stack, unstack
  Accept? (Y/n): y
  Generating all actions...

─── Assembling Domain ───

Output file [blocksworld-domain.pddl]:
```

For each component (types, constants, predicates, functions, actions) you can:
- Ask the LLM to generate from your domain description
- Enter data manually with guided prompts
- Review, request fixes, and confirm before proceeding

### Problem Generation

```bash
l2p generate problem
```

Generates a PDDL problem instance with the same interactive flow:

- Choose domain source (NL description or existing `.pddl` file)
- Generate/enter problem objects
- Generate initial state from description
- Generate goal state
- Output assembled `.pddl` file

### Generation Configuration

```bash
# Increase LLM retry attempts
l2p generate domain --max-retries 5
l2p generate problem --max-retries 5
```

---

## Model Management: `l2p models`

```bash
# List available models for the configured provider
l2p models list

# Show model details (context length, cost, params)
l2p models list --details

# List models for a specific provider
l2p models list --provider openai

# Switch model interactively
l2p models switch

# Test connection to the configured model
l2p models test

# Test without full model load (config validation only)
l2p models test --simple
```

---

## Configuration: `l2p config`

```bash
# View current config
l2p config show

# View raw YAML
l2p config show --raw

# Open in editor (respects $EDITOR env var)
l2p config edit

# Validate configuration
l2p config validate

# Reset to defaults
l2p config reset --force
```

---

## Templates: `l2p templates`

```bash
# List all available prompt templates
l2p templates list

# Show template content
l2p templates show --name prompt_predicates.md --category domain

# Find template file location
l2p templates find --name prompt_actions.md
```

---

## Create Blank Files: `l2p new`

```bash
# Create a domain file
l2p new blocksworld.pddl

# Create a problem file
l2p new pb1.pddl --type problem --domain-name blocksworld

# Overwrite existing
l2p new my-domain.pddl --force
```

---

## Interactive Chat: `l2p chat`

Start a REPL-style chat session with the configured LLM:

```bash
l2p chat
```

```
━━━ L2P Chat ━━━
  openai/gpt-4o-mini
  /exit              Quit
  /edit <file>       Edit a PDDL file with LLM assistance
  /validate <file>   Validate PDDL file syntax

>>> How do I generate predicates for a rover domain?
```

`/edit` loads a PDDL file, prompts for an edit description, sends to the LLM, shows a diff, and asks for confirmation before overwriting. `/validate` checks PDDL syntax using the `pddl` parser.
