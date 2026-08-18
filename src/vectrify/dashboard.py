import contextlib
import logging
import math
import threading
import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from vectrify.search.stats import SearchStats

_REFRESH_INTERVAL = 0.25


def _bar(fraction: float, width: int = 12) -> str:
    filled = round(min(1.0, max(0.0, fraction)) * width)
    return "█" * filled + "░" * (width - filled)


def _threshold_color(fraction: float) -> str:
    """Green while there is headroom, red as a stop threshold is approached."""
    if fraction > 0.8:
        return "red"
    if fraction > 0.5:
        return "yellow"
    return "green"


def _stop_row(
    label: str, fraction: float, value: str, note: str = ""
) -> tuple[str, str]:
    """A progress-toward-stop-criterion row: colored bar, value, optional note."""
    color = _threshold_color(fraction)
    suffix = f"  [dim]{note}[/dim]" if note else ""
    return label, f"  [{color}]{_bar(fraction, width=20)}[/{color}]  {value}{suffix}"


def _fmt_score(score: float) -> str:
    return f"{score:.6f}" if score < math.inf else "—"


def _fmt_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_renderable(stats: SearchStats) -> Panel:
    s = stats

    phase_color = "magenta" if s.phase == "seed" else "cyan"
    header = (
        f"[bold]{s.strategy_name or '—'}[/bold]"
        f"  ·  model: [cyan]{s.model_name or '—'}[/cyan]"
        f"  ·  epoch [bold]{s.epoch}{f'/{s.epochs}' if s.epochs else ''}[/bold]"
        f" [{phase_color}]{s.phase}[/{phase_color}]"
        f"  ·  [dim]{_fmt_elapsed(s.elapsed())}[/dim]"
    )

    # The evaluator's verdict is the only score a run has, and it speaks a few
    # dozen times rather than once per task -- so how long since it last
    # approved anything belongs next to the number. A row labelled "score" that
    # moves three times in an hour reads as a stalled run rather than a rare
    # judgement.
    since_gain = (
        f"   since gain [{'yellow' if s.eval_patience_fraction() > 0.6 else 'dim'}]"
        f"{s.eval_checks_without_gain}/{s.eval_patience}[/]"
        if s.eval_patience > 0
        else ""
    )
    eval_line = (
        f"  [bold green]{_fmt_score(s.best_score)}[/bold green]"
        f"   checks [dim]{s.eval_checks}[/dim]{since_gain}"
    )

    # Unchanged is shown apart from invalid because they call for opposite
    # responses: a rising invalid rate means candidates are breaking, a rising
    # unchanged rate means the operators are spending the run standing still.
    unchanged_pct = (
        100.0 * s.unchanged_count / s.tasks_completed if s.tasks_completed else 0.0
    )
    tasks_line = (
        f"  completed [bold]{s.tasks_completed:,}[/bold]"
        f"   accept [green]{s.accept_rate() * 100:.1f}%[/green]"
        f"   pool-rej [yellow]{s.pool_rejected_rate() * 100:.1f}%[/yellow]"
        f"   unchanged [{'red' if unchanged_pct > 25 else 'yellow'}]"
        f"{unchanged_pct:.1f}%[/]"
        f"   invalid [red]{s.invalid_rate() * 100:.1f}%[/red]"
    )

    in_flight_str = (
        f" [dim](+{s.llm_calls_in_flight} in flight)[/dim]"
        if s.llm_calls_in_flight
        else ""
    )
    batch_str = (
        f"   batch [yellow]{s.seeds_completed}/{s.seeds_target}[/yellow]"
        if s.phase == "seed" and s.seeds_target > 0
        else ""
    )
    llm_line = (
        f"  calls [bold]{s.llm_call_count:,}[/bold]{in_flight_str}"
        f"   valid [green]{s.llm_valid_rate() * 100:.1f}%[/green]"
        f"   pool-acc [cyan]{s.llm_accept_rate() * 100:.1f}%[/cyan]"
        f"{batch_str}"
    )

    # Pool stats: single line with diversity + variance values

    pool_line = f"  diversity [dim]{s.pool_diversity:.3f}[/dim]"

    # Which criterion is actually going to end this epoch. A run has three that
    # can fire and no way to tell them apart from the outside: across four runs
    # the answer moved from the evaluator to staleness with no setting changed,
    # only the acceptance rate.
    nearest, closeness = s.nearest_epoch_end()
    epoch_bits = [f"stale [dim]{s.epoch_no_improve:,}/{s.epoch_patience or '—'}[/dim]"]
    if s.epoch_max_tasks > 0:
        epoch_bits.append(f"budget [dim]{s.epoch_tasks:,}/{s.epoch_max_tasks:,}[/dim]")
    if closeness > 0:
        epoch_bits.append(
            f"ends on [{_threshold_color(closeness)}]{nearest}[/]"
            f" [{_threshold_color(closeness)}]{_bar(closeness, width=10)}[/]"
        )
    epoch_line = "  " + "   ".join(epoch_bits)

    # Stop criteria rows (only when enabled)
    stop_rows: list[tuple[str, str]] = []

    if s.phase == "seed" and s.seeds_target > 0:
        # Not a stop criterion, but the epoch is waiting on it all the same.
        stop_rows.append(
            _stop_row(
                "seeding",
                s.seed_fraction(),
                f"{s.seeds_completed}/{s.seeds_target}",
            )
        )

    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold dim", width=10)
    table.add_column()

    table.add_row("evaluator", Text.from_markup(eval_line))
    table.add_row("epoch", Text.from_markup(epoch_line))
    table.add_row("tasks", Text.from_markup(tasks_line))
    table.add_row("llm", Text.from_markup(llm_line))
    table.add_row("pool", Text.from_markup(pool_line))
    for label, content in stop_rows:
        table.add_row(label, Text.from_markup(content))

    with s._lock:
        events = list(s.recent_events)

    if events:
        table.add_row("", "")
        for evt in events:
            table.add_row("", Text(evt, style="dim", overflow="ellipsis", no_wrap=True))

    return Panel(
        table,
        title=Text.from_markup(header),
        title_align="left",
        border_style="blue",
    )


class DashboardLogHandler(logging.Handler):
    """Appends formatted log records to stats.recent_events for display."""

    def __init__(self, stats: SearchStats, level: int = logging.INFO) -> None:
        super().__init__(level)
        self.stats = stats
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with self.stats._lock:
                self.stats.recent_events.append(msg)
        except Exception:
            self.handleError(record)


class Dashboard:
    """Live terminal dashboard backed by Rich Live."""

    def __init__(self, stats: SearchStats) -> None:
        self.stats = stats
        self.log_handler = DashboardLogHandler(stats, level=logging.INFO)
        self._console = Console(highlight=False)
        self._live = Live(
            console=self._console,
            auto_refresh=False,
            redirect_stderr=True,
        )
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def __enter__(self) -> "Dashboard":
        self._stop.clear()
        self._live.__enter__()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="DashboardThread"
        )
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        # Final render
        with contextlib.suppress(Exception):
            self._live.update(_build_renderable(self.stats), refresh=True)
        self._live.__exit__(*exc_info)

    def _loop(self) -> None:
        while not self._stop.is_set() and not self.stats.shutting_down:
            with contextlib.suppress(Exception):
                self._live.update(_build_renderable(self.stats), refresh=True)
            time.sleep(_REFRESH_INTERVAL)
