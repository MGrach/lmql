import os
import builtins
import tempfile
import pathlib
import inspect
import pytest


# import sys
# print(sys.path)

# # sys.path.insert(0, "/home/maria/Documents/lmql/lmql/src")

import lmql
# print(lmql.__file__)

# disabling persistent cache
os.environ["NO_CACHE"] = "1"

TEMP_DIR = tempfile.gettempdir()
PROJECT_DIR = os.getcwd()

# call‑site aware I/O Tracker to build read, write map
class IOTracker:
    """Captures all file I/O operations with call‑site and semantic info."""

    def __init__(self):
        self.log = []  # list of dicts: op, path, mode, domain, caller, classification

    def _get_caller(self, skip_frames=3):
        """Get the function name of the caller (skip tracker internals)."""
        frame = inspect.currentframe()
        for _ in range(skip_frames):
            if frame is not None:
                frame = frame.f_back
        if frame is not None:
            return frame.f_code.co_name
        return "<unknown>"

    def _classify_operation(self, op, path, mode, caller):
        """Semantic classification based on caller and path."""
        if "temp" in path.lower() or path.startswith(TEMP_DIR):
            return "TEMP_ARTIFACT"
        if caller and ("compile" in caller.lower() or "serialize" in caller.lower()):
            return "SERIALIZES_CODE"
        if caller and ("exec" in caller.lower() or "eval" in caller.lower()):
            return "INVOKES_EXECUTION"
        if mode and any(m in mode for m in ("w", "a", "+")):
            return "WRITES_ARTIFACT"
        return "READ"

    def add(self, op, path, mode=None):
        path = str(path)
        caller = self._get_caller()
        classification = self._classify_operation(op, path, mode, caller)

        if path.startswith(TEMP_DIR):
            domain = "temp"
        elif path.startswith(PROJECT_DIR):
            domain = "project"
        else:
            domain = "external"

        self.log.append({"op": op,
                         "path": path,
                         "mode": mode,
                         "domain": domain,
                         "caller": caller,
                         "classification": classification
                         })

    def get_writes(self):
        return [e for e in self.log if e["op"] in ("open", "os.open", "pathlib.open")
                and e.get("mode") and any(m in e["mode"] for m in ("w", "a", "+"))]

    def get_reads(self):
        return [e for e in self.log if e["op"] in ("open", "os.open", "pathlib.open")
                and e.get("mode") and e["mode"] == "r"]

    def get_temp_ops(self):
        return [e for e in self.log if e["domain"] == "temp"]

    def build_responsibility_map(self):
        return {
            "reads": self.get_reads(),
            "writes": self.get_writes(),
            "temp": self.get_temp_ops(),
            "all": self.log
        }

    def format_report(self):
        lines = ["\n" + ">"*70, "I/O RESPONSIBILITY REPORT", ">"*70]
        rmap = self.build_responsibility_map()

        sections = [
            ("WRITES (should be empty)", "writes"),
            ("READS", "reads"),
            ("TEMPORARY FILE ACCESS (should be empty)", "temp"),
        ]

        for title, key in sections:
            lines.append(f"\n-- {title} --")
            if not rmap[key]:
                lines.append("  (none)")
            else:
                for e in rmap[key]:
                    writer = e["caller"]
                    consumer = "compiler" if e["classification"] == "SERIALIZES_CODE" else "runtime"
                    lines.append(f"  [{e['domain']}] {e['op']} mode='{e['mode']}'\n"
                                 f"    path: {e['path']}\n"
                                 f"    caller: {writer} | classification: {e['classification']} | consumer: {consumer}"
                                 )

        lines.append("\n" + ">"*70 + "\n")
        return "\n".join(lines)


@pytest.fixture
def io_tracker(monkeypatch):
    """Fixture providing an IOTracker with call‑site capture."""
    tracker = IOTracker()

    # patching builtins.open
    original_open = builtins.open

    def tracked_open(file, mode='r', *args, **kwargs):
        tracker.add("open", file, mode)
        return original_open(file, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", tracked_open)

    # patching os.open
    if hasattr(os, "open"):
        original_os_open = os.open

        def tracked_os_open(path, flags, *args, **kwargs):
            mode = "w" if flags & os.O_WRONLY or flags & os.O_RDWR else "r"
            tracker.add("os.open", path, mode)
            return original_os_open(path, flags, *args, **kwargs)
        monkeypatch.setattr(os, "open", tracked_os_open)

    # patching pathlib.Path.open
    original_path_open = pathlib.Path.open

    def tracked_path_open(self, mode='r', *args, **kwargs):
        tracker.add("pathlib.open", str(self), mode)
        return original_path_open(self, mode, *args, **kwargs)
    monkeypatch.setattr(pathlib.Path, "open", tracked_path_open)

    # patching tempfile functions
    for func_name in ["NamedTemporaryFile", "TemporaryFile", "mkstemp", "mkdtemp", "mktemp"]:
        if hasattr(tempfile, func_name):
            original = getattr(tempfile, func_name)

            def make_wrapper(orig, name):
                def wrapper(*args, **kwargs):
                    result = orig(*args, **kwargs)
                    if hasattr(result, "name"):
                        tracker.add(f"tempfile.{name}", result.name, "w")
                    elif isinstance(result, (tuple, list)) and len(result) > 0:
                        if isinstance(result[0], int):
                            tracker.add(f"tempfile.{name}", result[1], "w")
                    elif isinstance(result, str):
                        tracker.add(f"tempfile.{name}", result, "w")
                    return result
                return wrapper
            monkeypatch.setattr(tempfile, func_name,
                                make_wrapper(original, func_name))

    return tracker


def assert_no_writes(io_tracker):
    writes = io_tracker.get_writes()
    temp_ops = io_tracker.get_temp_ops()
    if writes or temp_ops:
        report = io_tracker.format_report()
        raise AssertionError(f"Unexpected I/O detected.\n{report}")

# tests 
def test_string_query_no_file_io(io_tracker):
    query_string = '''
    "Say 'Hello': [RESPONSE]"
    '''
    q = lmql.query(query_string, is_async=False)
    q(return_prompt_string=True) #type:ignore
    assert_no_writes(io_tracker)


def test_decorator_query_no_file_io(io_tracker):
    @lmql.query(is_async=False)
    def hello():
        '''lmql
        "Say 'Hello': [RESPONSE]"
        '''
    hello(return_prompt_string=True)  # type:ignore
    assert_no_writes(io_tracker)


def test_async_query_no_file_io(io_tracker):
    @lmql.query(is_async=True)
    async def hello_async():
        '''lmql
        "Say 'Hello': [RESPONSE]"
        '''
    import asyncio
    asyncio.run(hello_async(return_prompt_string=True))  # type:ignore
    assert_no_writes(io_tracker)


def test_complex_query_with_where_no_file_io(io_tracker):
    q = lmql.query('"Say [WORD]" where len(TOKENS(WORD)) == 1', is_async=False)
    q(return_prompt_string=True)  # type:ignore
    assert_no_writes(io_tracker)
