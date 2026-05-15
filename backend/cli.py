"""Typer CLI: digest, history, profile subcommands."""

import asyncio
import logging

import typer

app = typer.Typer(name="paper-scout", help="Weekly AI paper discovery agent.")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logging.getLogger("google_genai").setLevel(logging.WARNING)


@app.command()
def digest(
    dry_run: bool = typer.Option(False, "--dry-run", help="Run pipeline without DB write or email."),
) -> None:
    """Run the weekly digest pipeline."""
    from backend.digest import run_weekly_digest
    import backend.db.repository as repo
    from backend.config import get_settings
    from backend.db.migrations import run as run_migrations
    import sqlite3

    settings = get_settings()
    run_migrations(settings.db_path)
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    repo.set_connection(conn)

    try:
        digest_id = asyncio.run(run_weekly_digest(dry_run=dry_run))
        if dry_run:
            typer.echo("Dry run complete — no DB writes or email sent.")
        elif digest_id > 0:
            typer.echo(f"Digest {digest_id} complete.")
            history = repo.get_digest_history(limit=1)
            if history:
                for item in history[0]["items"]:
                    label = "Wildcard" if item["is_wildcard"] else "Exploit"
                    typer.echo(f"  [{label}] {item['title']}")
    except Exception as exc:
        typer.echo(f"Digest failed: {exc}", err=True)
        raise typer.Exit(1)
    finally:
        conn.close()


@app.command()
def history(
    limit: int = typer.Option(10, "--limit", help="Number of past digests to show."),
) -> None:
    """Show past digests with items and ratings."""
    import backend.db.repository as repo
    from backend.config import get_settings
    from backend.db.migrations import run as run_migrations
    import sqlite3

    settings = get_settings()
    run_migrations(settings.db_path)
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    repo.set_connection(conn)

    try:
        digests = repo.get_digest_history(limit=limit)
        if not digests:
            typer.echo("No digests found.")
            return
        for d in digests:
            typer.echo(f"\nDigest {d['id']} — week of {d['week_start']}")
            for item in d["items"]:
                rating = item.get("rating") or "—"
                label = "W" if item["is_wildcard"] else "E"
                typer.echo(f"  [{label}{item['position']}] {item['title']}  (rated: {rating})")
    finally:
        conn.close()


@app.command()
def profile(
    history_flag: bool = typer.Option(False, "--history", help="Show last 5 profiles."),
    rebuild: bool = typer.Option(False, "--rebuild", help="Trigger full profile rebuild."),
) -> None:
    """Show or rebuild the interest profile."""
    import backend.db.repository as repo
    from backend.config import get_settings
    from backend.db.migrations import run as run_migrations
    import sqlite3

    settings = get_settings()
    run_migrations(settings.db_path)
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    repo.set_connection(conn)

    try:
        if rebuild:
            from backend.agents.profiler import update_profile
            from backend.llm.factory import get_provider

            provider = get_provider(settings)
            rated = repo.get_rated_items_with_summaries()
            if not rated:
                typer.echo("No rated items with summaries — cannot rebuild profile.")
                return
            up_items = [(it, r) for it, r in rated if r == "up"][: settings.profile.full_rebuild_up_limit]
            down_items = [(it, r) for it, r in rated if r == "down"][: settings.profile.full_rebuild_down_limit]
            prose = asyncio.run(
                update_profile(up_items + down_items, None, provider, mode="full_rebuild")
            )
            repo.insert_profile(prose, "full_rebuild", len(rated), settings.llm_provider)
            typer.echo("Profile rebuilt.")
            typer.echo(prose)
            return

        if history_flag:
            profiles = repo.get_profile_history(limit=5)
            if not profiles:
                typer.echo("No profiles found.")
                return
            for p in profiles:
                typer.echo(f"\n[{p['created_at']}] mode={p['mode']} rated_count={p['rated_items_count']}")
                typer.echo(p["prose"][:300] + ("..." if len(p["prose"]) > 300 else ""))
            return

        # Default: show current profile
        p = repo.get_latest_profile()
        if p is None:
            typer.echo("No profile yet. Rate some papers first.")
            return
        typer.echo(f"Profile (mode={p['mode']}, rated={p['rated_items_count']}, provider={p['llm_provider']}):")
        typer.echo(p["prose"])
    finally:
        conn.close()
