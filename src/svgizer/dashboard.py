import contextlib
import logging
import threading
import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from svgizer.search.models import INVALID_SCORE
from svgizer.search.stats import SearchStats

_REFRESH_INTERVAL = 0.25


def _bar(fraction: float, width: int = 12) -> str:
    filled = round(min(1.0, max(0.0, fraction)) * width)
    return "█" * filled + "░" * (width - filled)


def _fmt_score(score: float) -> str:
    return f"{score:.6f}" if score < INVALID_SCORE else "—"


def _fmt_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_renderable(stats: SearchStats) -> Panel:
    s = stats

    header = (
        f"[bold]{s.strategy_name or '—'}[/bold]"
        f"  ·  model: [cyan]{s.model_name or '—'}[/cyan]"
        f"  ·  epoch [bold]{s.epoch}[/bold]"
        f"  ·  [dim]{_fmt_elapsed(s.elapsed())}[/dim]"
    )

    score_line = f"  [bold green]{_fmt_score(s.best_score)}[/bold green]"

    tasks_line = (
        f"  completed [bold]{s.tasks_completed:,}[/bold]"
        f"   accept [green]{s.accept_rate() * 100:.1f}%[/green]"
        f"   pool-rej [yellow]{s.pool_rejected_rate() * 100:.1f}%[/yellow]"
        f"   invalid [red]{s.invalid_rate() * 100:.1f}%[/red]"
    )

    in_flight_str = (
        f" [dim](+{s.llm_calls_in_flight} in flight)[/dim]"
        if s.llm_calls_in_flight
        else ""
    )
    llm_line = (
        f"  calls [bold]{s.llm_call_count:,}[/bold]{in_flight_str}"
        f"   valid [green]{s.llm_valid_rate() * 100:.1f}%[/green]"
        f"   pool-acc [cyan]{s.llm_accept_rate() * 100:.1f}%[/cyan]"
        f"   rate [yellow]{s.effective_llm_rate() * 100:.2f}%[/yellow]"
    )

    mut_line = (
        f"  calls [bold]{s.mutation_call_count:,}[/bold]"
        f"   pool-acc [cyan]{s.mutation_accept_rate() * 100:.1f}%[/cyan]"
    )

    div_bar = _bar(s.pool_diversity, width=20)
    if s.epoch_diversity > 0:
        if s.pool_diversity < s.epoch_diversity:
            div_color = "red"
        elif s.pool_diversity < s.epoch_diversity * 2:
            div_color = "yellow"
        else:
            div_color = "green"
        div_threshold = f"  [dim]epoch at < {s.epoch_diversity:.3f}[/dim]"
    else:
        div_color = "cyan"
        div_threshold = ""
    div_line = (
        f"  [{div_color}]{div_bar}[/{div_color}]  {s.pool_diversity:.3f}{div_threshold}"
    )

    if s.epoch_variance > 0:
        # Bar scaled so full = 4x threshold (threshold is expected convergence point)
        var_frac = min(1.0, s.pool_score_std / (s.epoch_variance * 4))
        if s.pool_score_std < s.epoch_variance:
            var_color = "red"
        elif s.pool_score_std < s.epoch_variance * 2:
            var_color = "yellow"
        else:
            var_color = "green"
        var_threshold = f"  [dim]epoch at < {s.epoch_variance:.4f}[/dim]"
    else:
        var_frac = 0.0
        var_color = "cyan"
        var_threshold = ""
    var_bar = _bar(var_frac, width=20)
    var_line = (
        f"  [{var_color}]{var_bar}[/{var_color}]  {s.pool_score_std:.4f}{var_threshold}"
    )

    if s.epoch_patience > 0:
        stag_frac = s.stagnation_fraction()
        if stag_frac > 0.8:
            bar_color = "red"
        elif stag_frac > 0.5:
            bar_color = "yellow"
        else:
            bar_color = "green"
        stag_bar = _bar(stag_frac, width=20)
        stag_line = (
            f"  [{bar_color}]{stag_bar}[/{bar_color}]"
            f"  {s.epoch_no_improve}/{s.epoch_patience}"
        )
    else:
        stag_line = f"  [dim]{s.epoch_no_improve:,} tasks since last improvement[/dim]"

    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold dim", width=10)
    table.add_column()

    table.add_row("score", Text.from_markup(score_line))
    table.add_row("tasks", Text.from_markup(tasks_line))
    table.add_row("llm", Text.from_markup(llm_line))
    table.add_row("mutation", Text.from_markup(mut_line))
    table.add_row("diversity", Text.from_markup(div_line))
    table.add_row("variance", Text.from_markup(var_line))
    table.add_row("stagnation", Text.from_markup(stag_line))

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
