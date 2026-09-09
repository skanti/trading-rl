"""Shared Rich console, progress, and completed-work summaries for download CLIs."""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import logging
import threading
import time

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text


CONSOLE = Console()
_REPORT = ContextVar("download_report", default=None)


def current_report():
    return _REPORT.get()


def info(message: str) -> None:
    CONSOLE.print(Text(message))


class DownloadReport:
    def __init__(self, title: str):
        self.title = title
        self.mode = "Download"
        self.output = None
        self.metrics: dict[str, tuple[int, str]] = {}
        self.outcomes: dict[str, str] = {}
        self.messages: dict[tuple[str, str], list[str]] = {}
        self.message_counts: Counter = Counter()
        self.lock = threading.RLock()
        self.started = time.monotonic()
        self.failure = None
        self.details: list[str] = []

    def set(self, label: str, count: int, unit: str) -> None:
        with self.lock:
            self.metrics[label] = (count, unit)

    def record(self, symbol: str, outcome: str) -> None:
        """Count a symbol only after its operation finishes, including failed saves."""
        with self.lock:
            self.outcomes[symbol] = outcome

    def message(self, level: str, message: str, group: str | None = None) -> None:
        with self.lock:
            key = (level, group or message)
            self.message_counts[key] += 1
            examples = self.messages.setdefault(key, [])
            if len(examples) < 3 and message not in examples:
                examples.append(message)

    def render(self) -> None:
        if not self.metrics and not self.failure and not self.messages:
            return
        counts = Counter(self.outcomes.values())
        for label in ("Updated", "Unchanged", "New downloads", "Full redownloads", "Skipped", "Failed"):
            if label in self.metrics and self.metrics[label][1] == "symbols":
                self.metrics[label] = (max(self.metrics[label][0], counts[label]), "symbols")
        warnings = sum(n for (level, _), n in self.message_counts.items() if level == "Warning")
        errors = sum(n for (level, _), n in self.message_counts.items() if level == "Error")
        status = "FAILED" if self.failure else "Completed with warnings" if warnings or errors else "Completed"
        table = Table(title=f"{self.title} — {status}")
        table.add_column("Result")
        table.add_column("Count", justify="right")
        table.add_column("Unit")
        for label, (count, unit) in self.metrics.items():
            table.add_row(label, f"{count:,}", unit)
        table.add_row("Warnings", f"{warnings:,}", "events")
        table.add_row("Errors", f"{errors:,}", "events")
        CONSOLE.print(table)
        info(f"Mode: {self.mode} · Elapsed: {time.monotonic() - self.started:.1f}s")
        if self.output:
            info(f"Output: {self.output}")
        info("Update/download counts reflect saved data; progress counts processed work.")
        for detail in self.details:
            info(detail)
        for key, examples in self.messages.items():
            level, _ = key
            count = self.message_counts[key]
            label = f"{level} ({count:,} event{'s' if count != 1 else ''}): "
            text = Text(label, style="yellow" if level == "Warning" else "red")
            text.append(examples[0])
            CONSOLE.print(text)
            for example in examples[1:]:
                info(f"  {example}")
            if count > len(examples):
                info(f"  … {count - len(examples):,} more events")


class _ConsoleLogHandler(logging.Handler):
    def __init__(self, report: DownloadReport):
        super().__init__()
        self.report = report

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if record.exc_info and record.exc_info[1]:
            message += f": {record.exc_info[1]}"
        if record.levelno >= logging.WARNING:
            self.report.message(
                "Error" if record.levelno >= logging.ERROR else "Warning",
                message, str(record.msg),
            )
        else:
            info(message)


def download_output(title: str, logger: logging.Logger, *, exit_on_error: bool = False):
    """Scope output to one command; never configure the application's root logger."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            report = DownloadReport(title)
            token = _REPORT.set(report)
            previous = (logger.handlers[:], logger.level, logger.propagate)
            logger.handlers = [_ConsoleLogHandler(report)]
            logger.setLevel(logging.INFO)
            logger.propagate = False
            try:
                return function(*args, **kwargs)
            except Exception as error:
                report.failure = str(error)
                report.message("Error", str(error))
                if exit_on_error:
                    raise SystemExit(1) from None
                raise
            except KeyboardInterrupt:
                report.failure = "Interrupted"
                report.message("Warning", "Interrupted; unfinished work is not counted as saved.")
                raise
            except SystemExit as error:
                if error.code:
                    report.failure = "Invalid arguments"
                raise
            finally:
                logger.handlers, logger.level, logger.propagate = previous
                _REPORT.reset(token)
                report.render()
        return wrapped
    return decorate


@contextmanager
def download_progress(*, total: int, description: str, unit: str = "symbols"):
    with Progress(
        TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
        TextColumn(unit), TimeElapsedColumn(), TimeRemainingColumn(), console=CONSOLE,
    ) as progress:
        task = progress.add_task(description, total=total)

        class Advance:
            def update(self, count: int = 1):
                progress.advance(task, count)

        yield Advance()


def track_download(iterable, *, total: int, description: str, unit: str = "symbols"):
    with download_progress(total=total, description=description, unit=unit) as progress:
        for item in iterable:
            yield item
            progress.update()
