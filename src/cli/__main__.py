from __future__ import annotations

import typer

from src.cli.commands import (
    cmd_config,
    cmd_doctor,
    cmd_health,
    cmd_ingest,
    cmd_query,
    cmd_up,
)

app = typer.Typer(
    name="rag-engine",
    help="Modular RAG engine — index and query your documents.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("ingest")(cmd_ingest)
app.command("query")(cmd_query)
app.command("up")(cmd_up)
app.command("health")(cmd_health)
app.command("doctor")(cmd_doctor)
app.command("config")(cmd_config)

if __name__ == "__main__":
    app()
