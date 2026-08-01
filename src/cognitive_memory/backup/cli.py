"""CM Backup CLI — `cm backup` subgroup.

Subcommands: create, restore, verify.
Registered into the main cm CLI via cli.add_command(backup_group).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from .exporter import export_backup
from .verifier import verify_backup, VerifyError
from .importer import import_backup, ImportError
from .pruner import prune_backups, DEFAULT_BACKUPS_ROOT

logger = logging.getLogger(__name__)


@click.group("backup")
def backup_group() -> None:
    """Backup and restore CM memory data."""
    pass


# ---------------------------------------------------------------------------
# cm backup create
# ---------------------------------------------------------------------------

@backup_group.command("create")
@click.option("--out", "out_dir", default=None, type=click.Path(),
              help="Parent directory for the backup artifact. "
                   "Default: ~/.synaptra/backups/")
@click.option("--name", default=None,
              help="Backup directory name. Default: cm-<timestamp>Z")
@click.option("--skip-cm-stop", is_flag=True, default=False, hidden=True,
              help="Skip stop/restart of CM service (for testing).")
def create(out_dir: str | None, name: str | None, skip_cm_stop: bool) -> None:
    """Create a full CM backup.

    Stops the CM service, exports all memory data to a NDJSON artifact,
    restarts CM, and runs the retention pruner.
    """
    _setup_logging()
    out_path = Path(out_dir) if out_dir else None
    try:
        backup_dir = export_backup(
            out_dir=out_path,
            name=name,
            skip_cm_restart=skip_cm_stop,
        )
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if code == 2:
            click.echo("ERROR: Could not open SurrealKV database. Is CM still running?", err=True)
        elif code == 3:
            click.echo("ERROR: Disk/write error during backup. Check disk space.", err=True)
        else:
            click.echo(f"ERROR: Backup failed (exit {code}).", err=True)
        sys.exit(code)

    click.echo(str(backup_dir))

    # Run retention pruner after successful backup
    backups_root = out_path or DEFAULT_BACKUPS_ROOT
    pruned = prune_backups(backups_dir=backups_root)
    if pruned:
        click.echo(f"Pruned {len(pruned)} old backup(s).", err=True)


# ---------------------------------------------------------------------------
# cm backup verify
# ---------------------------------------------------------------------------

@backup_group.command("verify")
@click.argument("backup_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--strict", is_flag=True, default=False,
              help="Fail on schema hash mismatch (exit 7).")
@click.option("--deep", is_flag=True, default=False,
              help="Load into a temp SurrealKV instance and run an HNSW query. ~30 s.")
def verify(backup_dir: Path, strict: bool, deep: bool) -> None:
    """Verify a CM backup artifact without restoring it."""
    _setup_logging()
    try:
        result = verify_backup(backup_dir, strict=strict, deep=deep)
    except VerifyError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(e.exit_code)

    for w in result["warnings"]:
        click.echo(f"WARNING: {w}", err=True)

    if result["errors"]:
        for err in result["errors"]:
            click.echo(f"ERROR: {err}", err=True)
        sys.exit(result.get("exit_code", 1))

    click.echo(f"OK: {backup_dir}")
    if result["warnings"]:
        click.echo(f"({len(result['warnings'])} warning(s); see stderr)")


# ---------------------------------------------------------------------------
# cm backup restore
# ---------------------------------------------------------------------------

@backup_group.command("restore")
@click.argument("backup_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--target", default=None, type=click.Path(),
              help="Target data directory (file mode). Default: ~/.synaptra/restore-<ts>/")
@click.option("--target-mode", default="file", type=click.Choice(["file", "ws"]),
              help="Import target: 'file' (embedded SurrealKV, default) or 'ws' (SurrealDB server).")
@click.option("--target-url", default=None,
              help="SurrealDB server URL for ws mode, e.g. ws://127.0.0.1:8000/rpc.")
@click.option("--force", is_flag=True, default=False,
              help=(
                  "File mode: allow restore into a non-empty directory.  "
                  "Ws mode: full reset — schema OVERWRITE + records replace + index rebuild."
              ))
@click.option("--strict", is_flag=True, default=False,
              help="Fail on schema hash mismatch.")
def restore(
    backup_dir: Path,
    target: str | None,
    target_mode: str,
    target_url: str | None,
    force: bool,
    strict: bool,
) -> None:
    """Restore a CM backup into a target data directory or SurrealDB server.

    File mode (default): restores into an embedded SurrealKV directory.
    Ws mode: imports into a running SurrealDB server via WebSocket.
      Requires --target-mode=ws and --target-url=ws://...
      Use --force when the target already has data (full reset semantics).
    """
    _setup_logging()

    if target_mode == "ws" and not target_url:
        click.echo("ERROR: --target-url is required when --target-mode=ws", err=True)
        sys.exit(1)

    target_path = Path(target) if target else None
    try:
        result = import_backup(
            backup_dir=backup_dir,
            target_dir=target_path,
            force=force,
            strict=strict,
            target=target_mode,
            target_url=target_url,
        )
    except ImportError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(e.exit_code)

    click.echo(f"OK: {result}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
