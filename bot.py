"""Top-level Pipecat Runner entrypoint (`uv run bot.py`)."""

from pipecat_demo.server import app, bot, run

__all__ = ["app", "bot", "run"]

if __name__ == "__main__":
    run()
