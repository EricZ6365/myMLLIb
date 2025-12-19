import time
import sys
class SimpleProfiler:
    def __init__(self):
        self.timings = {}

    def _trace(self, frame, event, arg):
        if event not in ("call", "return"):
            return self._trace
        code = frame.f_code
        func_name = code.co_name

        filename = code.co_filename
        key = (filename, func_name)

        if event == "call":
            frame.f_locals["__start_time__"] = time.perf_counter()
        elif event == "return":
            start = frame.f_locals.get("__start_time__")
            if start is not None:
                elapsed = time.perf_counter() - start
                self.timings[key] = self.timings.get(key, 0) + elapsed

        return self._trace

    def start(self):
        sys.setprofile(self._trace)

    def stop(self):
        sys.setprofile(None)

    def report(self, limit=20):
        print(f"\nTop {limit} functions by total time:")
        sorted_funcs = sorted(self.timings.items(), key=lambda x: -x[1])
        for (filename, func), total_time in sorted_funcs[:limit]:
            print(f"{func:30s} ({filename}) - {total_time:.6f} sec")

__all__ = ["SimpleProfiler"]