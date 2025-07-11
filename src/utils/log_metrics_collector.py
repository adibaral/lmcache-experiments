import re
import threading
from pathlib import Path
from collections import defaultdict


class LogMetricsCollector:
    """
    A class to collect and store metrics from a log file.
    It continuously monitors the log file for specific patterns
    and updates the metrics_by_rid dictionary.
    """

    LMCACHE_PATTERN = re.compile(
        r"Reqid: (\w+), Total tokens (\d+), LMCache hit tokens: (\d+), need to load: (\d+)"
    )

    THROUGHPUT_PATTERN = re.compile(
        r"Avg prompt throughput:\s*([0-9.]+)\s*tokens/s,\s*Avg generation throughput:\s*([0-9.]+)\s*tokens/s"
    )

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.metrics_by_rid = defaultdict(dict)
        self.throughput_stats = defaultdict(dict)
        self._start_log_monitor()

    def parse_log_line(self, line: str):
        m = self.LMCACHE_PATTERN.search(line)
        if m:
            rid, total, hit, need = m.groups()
            self.metrics_by_rid[rid] = {
                "lmcache_total_tokens": int(total),
                "lmcache_hit_tokens": int(hit),
                "lmcache_needed_tokens": int(need),
            }
            return

        m = self.THROUGHPUT_PATTERN.search(line)
        if m:
            self.throughput_stats = {
                "avg_prompt_tps": float(m.group(1)),
                "avg_generation_tps": float(m.group(2)),
            }

    def _tail_log_file(self):
        """Continuously tail the log file and feed each line to parse_log_line."""
        with Path(self.log_path).open("r") as f:
            f.seek(0, 2)  # Seek to the end of the file
            while True:
                line = f.readline()
                if not line:
                    continue
                self.parse_log_line(line)

    def _start_log_monitor(self):
        t = threading.Thread(target=self._tail_log_file, daemon=True)
        t.start()
