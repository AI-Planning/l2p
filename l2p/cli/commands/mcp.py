"""Start the L2P MCP server.

Usage:
  l2p mcp
"""

import argparse
import sys


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "mcp",
        help="Start the L2P MCP server",
        description="Start the Model Context Protocol server for L2P. "
                    "Configure your MCP client to run this command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Configuration (Claude Desktop):
  Add to your claude_desktop_config.json:
    "mcpServers": {
      "l2p": {
        "command": "l2p",
        "args": ["mcp"]
      }
    }

Configuration (Claude Code):
  Add to your .claude/settings.json:
    "mcpServers": {
      "l2p": {
        "command": "l2p",
        "args": ["mcp"]
      }
    }
        """,
    )
    parser.set_defaults(func=mcp_command)


def mcp_command(args):
    try:
        from l2p.mcp.server import run
        run()
    except ImportError as e:
        print(f"[ERROR] MCP dependencies not installed: {e}", file=sys.stderr)
        print("\nInstall with: pip install l2p[mcp]", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] MCP server failed: {e}", file=sys.stderr)
        sys.exit(1)
