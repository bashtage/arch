"""
Isolate the access violations in numpy.linalg on Windows on ARM.

The tests that crashed in the first Windows on ARM run are run one at a time, each in
its own process. Before every call to numpy.linalg.inv or solve the arguments are
saved, so that after a crash the file holds the matrix that crashed. Each of these
matrices is then passed to inv or solve in new processes that only import numpy, with
the OpenBLAS kernels that OpenBLAS picks and with others, and with 1 and 4 threads.
A sweep over random matrices of sizes 1 to 160 runs in each of these configurations.

    python windows_arm_linalg.py OUTPUT_DIR

The saved matrices of the crashes are left in OUTPUT_DIR. Exits with 1 if any test
crashes.

    python windows_arm_linalg.py --sizes OUTPUT_DIR

maps the sizes at which inv of the identity, and inv and solve of a random positive
definite matrix, crash with 1 OpenBLAS thread, each call in its own process that only
imports numpy, and then tries other numbers of threads at the sizes that crash. Exits
with 1 if any call crashes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os

# Runs this script in new processes, so that a crash does not stop the others
import subprocess  # nosec B404
import sys

# The tests that crashed in the first run, as paths in the tests of the installed arch
TESTS = [
    "unitroot/test_dynamic_ols.py::test_smoke[True-aic-None-False-None-None-ct]",
    "unitroot/test_dynamic_ols.py::test_smoke[True-aic-None-False-None-None-ctt]",
    "unitroot/test_dynamic_ols.py::test_smoke[False-aic-None-False-None-None-ct]",
    "unitroot/test_dynamic_ols.py::test_smoke[False-aic-None-False-None-None-ctt]",
    "unitroot/test_dynamic_ols.py::test_smoke[True-aic-None-False-None-2-n]",
    "unitroot/test_dynamic_ols.py::test_smoke[False-aic-None-False-None-2-n]",
    "unitroot/test_dynamic_ols.py::test_smoke[True-aic-None-False-None-2-c]",
    "unitroot/test_dynamic_ols.py::test_smoke[False-aic-None-False-None-2-c]",
    "unitroot/test_dynamic_ols.py::test_smoke[True-aic-None-False-3-None-ctt]",
    "unitroot/test_dynamic_ols.py::test_smoke[False-aic-None-False-3-None-ctt]",
    "unitroot/test_dynamic_ols.py::test_direct_eviews[pandas-config3]",
    "unitroot/test_dynamic_ols.py::test_direct_eviews[numpy-config3]",
    "unitroot/test_unitroot.py::test_autolag_warning",
    "unitroot/test_unitroot.py::TestUnitRoot::test_dfgls_auto_low_memory",
]
# Tests that passed, to compare with: larger and smaller matrices than the crashes
CONTROLS = [
    "unitroot/test_dynamic_ols.py::test_smoke[True-aic-None-False-None-None-c]",
    "unitroot/test_dynamic_ols.py::test_smoke[True-bic-None-False-None-None-ct]",
]
# OPENBLAS_CORETYPE, where None lets OpenBLAS pick the kernels, and the threads
CORETYPES = [None, "ARMV8", "NEOVERSEN1", "CORTEXA57"]
THREADS = ["1", "4"]
REPEATS = 50
SWEEP_PASSES = 3
SWEEP_MAX = 160
# The map of the sizes that crash: every size to 64, then every 8th
SIZES = list(range(1, 65)) + list(range(72, SWEEP_MAX + 1, 8))
OPS = ["inv-identity", "inv-random", "solve-random"]
# OPENBLAS_NUM_THREADS, where "" leaves it unset so that OpenBLAS picks the number
THREAD_COUNTS = ["", "1", "2", "3", "4"]


def blas_info() -> str:
    """The BLAS libraries that numpy loaded"""
    from threadpoolctl import threadpool_info  # noqa: PLC0415

    # threadpoolctl only finds the BLAS of the modules that are loaded
    importlib.import_module("numpy")
    keys = ("internal_api", "version", "architecture", "threading_layer", "num_threads")
    return json.dumps([{k: i.get(k) for k in keys} for i in threadpool_info()])


def run_test(nodeid: str, dump: str) -> None:
    """Run a test and save the arguments of every inv and solve before the call"""
    import numpy as np  # noqa: PLC0415
    import pytest  # noqa: PLC0415

    from arch.unitroot import unitroot  # noqa: PLC0415

    calls = [0]

    def wrap(name, func):
        def wrapper(a, *args, **kwargs):
            calls[0] += 1
            arrays = {"a": np.asarray(a)}
            if args:
                arrays["b"] = np.asarray(args[0])
            with open(dump, "wb") as f:
                np.savez(f, func=np.array(name), call=np.array(calls[0]), **arrays)
                f.flush()
                os.fsync(f.fileno())
            return func(a, *args, **kwargs)

        return wrapper

    inv = wrap("inv", np.linalg.inv)
    solve = wrap("solve", np.linalg.solve)
    # cointegration calls np.linalg.inv and unitroot imports inv and solve by name
    np.linalg.inv = inv
    np.linalg.solve = solve
    unitroot.inv = inv
    unitroot.solve = solve
    print("BLAS", blas_info(), flush=True)
    sys.exit(pytest.main([nodeid, "-v", "-p", "no:cacheprovider", "-p", "no:randomly"]))


def replay(dump: str) -> None:
    """Call inv or solve on a saved matrix in a process that only imports numpy"""
    import numpy as np  # noqa: PLC0415

    print("BLAS", blas_info(), flush=True)
    data = np.load(dump)
    func, a = str(data["func"]), data["a"]
    for _ in range(REPEATS):
        if func == "inv":
            np.linalg.inv(a)
        else:
            np.linalg.solve(a, data["b"])
    print("ok", flush=True)


def sweep() -> None:
    """inv and solve of random positive definite matrices of increasing size"""
    import numpy as np  # noqa: PLC0415

    print("BLAS", blas_info(), flush=True)
    rng = np.random.default_rng(0)
    for _ in range(SWEEP_PASSES):
        for n in range(1, SWEEP_MAX + 1):
            x = rng.standard_normal((max(2 * n, 200), n))
            a = x.T @ x / x.shape[0]
            print(f"inv {n}", flush=True)
            np.linalg.inv(a)
            print(f"solve {n}", flush=True)
            np.linalg.solve(a, rng.standard_normal(n))
            print(f"solve (n, 3) {n}", flush=True)
            np.linalg.solve(a, rng.standard_normal((n, 3)))
    print("ok", flush=True)


def run_op(op: str, n: int) -> None:
    """One call of inv or solve with an n by n matrix"""
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(n)
    x = rng.standard_normal((max(2 * n, 200), n))
    a = x.T @ x / x.shape[0]
    if op == "inv-identity":
        np.linalg.inv(np.eye(n))
    elif op == "inv-random":
        np.linalg.inv(a)
    elif op == "solve-random":
        np.linalg.solve(a, rng.standard_normal(n))
    else:
        raise ValueError(op)
    print("ok", flush=True)


def child(args: list[str], coretype: str | None = None, threads: str | None = None):
    """
    Run this script in a new process and return its outcome and output

    threads is the number of OpenBLAS threads, "" to leave it to OpenBLAS, and None
    to keep the number of the job.
    """
    env = os.environ.copy()
    env.pop("OPENBLAS_CORETYPE", None)
    if coretype is not None:
        env["OPENBLAS_CORETYPE"] = coretype
    if threads == "":
        env.pop("OPENBLAS_NUM_THREADS", None)
        env.pop("OMP_NUM_THREADS", None)
    elif threads is not None:
        env["OPENBLAS_NUM_THREADS"] = env["OMP_NUM_THREADS"] = threads
    cmd = [sys.executable, "-X", "faulthandler", os.path.abspath(__file__), *args]
    try:
        # Only this script, with arguments from this script, and without a shell
        proc = subprocess.run(  # nosec B603
            cmd, capture_output=True, text=True, env=env, timeout=900, check=False
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return "timeout", out
    code = proc.returncode
    if code == 0:
        outcome = "ok"
    elif code in (1, 2, 3, 4, 5):
        outcome = f"failed ({code})"
    else:
        outcome = f"crashed (0x{code & 0xFFFFFFFF:08X})"
    return outcome, proc.stdout + proc.stderr


def architecture(out: str) -> str:
    """The OpenBLAS kernels from the BLAS line of a child's output"""
    for line in out.splitlines():
        if line.startswith("BLAS "):
            info = json.loads(line[5:])
            return ", ".join(f"{i['architecture']} ({i['version']})" for i in info)
    return "?"


def last_line(out: str) -> str:
    """The last call of a sweep, or ok, rather than the end of a traceback"""
    lines = [
        line for line in out.splitlines() if line.startswith(("inv ", "solve ", "ok"))
    ]
    return lines[-1] if lines else ""


def ranges(values: list[int]) -> str:
    """Sizes as ranges, such as 1-5, 8"""
    parts: list[list[int]] = []
    for v in sorted(values):
        if parts and parts[-1][1] == v - 1:
            parts[-1][1] = v
        else:
            parts.append([v, v])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in parts)


def map_sizes(outdir: str) -> int:
    # Only to read the build configuration: this process does not call numpy.linalg
    import numpy as np  # noqa: PLC0415

    os.makedirs(outdir, exist_ok=True)
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    summary = ["## Sizes at which numpy.linalg crashes on Windows on ARM", ""]
    summary += [f"numpy {np.__version__}, {blas.get('openblas configuration')}", ""]
    summary += [f"Sizes: {ranges(SIZES)}. Each call in its own process.", ""]
    summary += ["| Call | Sizes that crash with 1 thread |", "|---|---|"]
    crashed: dict[str, list[int]] = {}
    for op in OPS:
        crashed[op] = []
        for n in SIZES:
            outcome, _ = child(["--op", op, str(n)], threads="1")
            print(f"{op} {n}: {outcome}", flush=True)
            if not outcome.startswith("ok"):
                crashed[op].append(n)
        summary.append(f"| {op} | {ranges(crashed[op]) or 'none'} |")

    # The smallest sizes that crash, and the largest
    found = sorted({n for sizes in crashed.values() for n in sizes})
    check = sorted(set(found[:2] + found[-1:])) if found else [23, 30]
    summary += ["", "### Numbers of threads at these sizes", ""]
    summary += ["| Call | Size | OPENBLAS_NUM_THREADS | Outcome |", "|---|---|---|---|"]
    for op in OPS:
        for n in check:
            for threads in THREAD_COUNTS:
                outcome, _ = child(["--op", op, str(n)], threads=threads)
                row = f"| {op} | {n} | {threads or 'unset'} | {outcome} |"
                print(row, flush=True)
                summary.append(row)

    text = "\n".join(summary) + "\n"
    with open(os.path.join(outdir, "sizes.md"), "w") as f:
        f.write(text)
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text)
    print(text)
    return 1 if found else 0


def describe(dump: str) -> str:
    """The function, shape and call number of a saved matrix"""
    import numpy as np  # noqa: PLC0415

    data = np.load(dump)
    a = data["a"]
    text = f"{data['func']} #{data['call']} of {a.shape} {a.dtype}"
    if "b" in data:
        text += f", b {data['b'].shape}"
    return text


def main(outdir: str) -> int:
    # Only to read the saved matrices: this process does not call numpy.linalg
    import numpy as np  # noqa: PLC0415

    os.makedirs(outdir, exist_ok=True)
    spec = importlib.util.find_spec("arch")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("arch is not installed")
    tests_dir = os.path.join(spec.submodule_search_locations[0], "tests")
    summary = ["## numpy.linalg on Windows on ARM", ""]

    _, out = child(["--blas"])
    summary += ["BLAS: " + out.strip(), ""]

    summary += ["### Each test in its own process", ""]
    summary += ["| Test | Outcome | Last inv or solve |", "|---|---|---|"]
    crashes: dict[str, str] = {}
    for i, test in enumerate(TESTS + CONTROLS):
        dump = os.path.join(outdir, f"test-{i:02d}.npz")
        outcome, out = child(["--test", os.path.join(tests_dir, test), dump])
        last = describe(dump) if os.path.exists(dump) else ""
        print(f"{test}: {outcome}; last call {last}", flush=True)
        if outcome.startswith("crashed"):
            print("\n".join(out.splitlines()[-40:]), flush=True)
            if last:
                # Replay each matrix once, however many tests crashed on it
                data = np.load(dump)
                key = data["a"].tobytes() + data.get("b", np.empty(0)).tobytes()
                crashes.setdefault(hashlib.sha256(key).hexdigest(), dump)
        elif os.path.exists(dump):
            os.remove(dump)
        summary.append(f"| `{test}` | {outcome} | {last} |")

    summary += ["", "### The saved matrices of the crashes, with only numpy", ""]
    summary += ["| Matrix | OPENBLAS_CORETYPE | Threads | Kernels | Outcome |"]
    summary += ["|---|---|---|---|---|"]
    for dump in crashes.values():
        for coretype in CORETYPES:
            for threads in THREADS:
                outcome, out = child(["--replay", dump], coretype, threads)
                row = (
                    f"| {os.path.basename(dump)}: {describe(dump)} | {coretype} "
                    f"| {threads} | {architecture(out)} | {outcome} |"
                )
                print(row, flush=True)
                summary.append(row)

    summary += ["", "### Random matrices of sizes 1 to 160, with only numpy", ""]
    summary += ["| OPENBLAS_CORETYPE | Threads | Kernels | Outcome | Last call |"]
    summary += ["|---|---|---|---|---|"]
    for coretype in CORETYPES:
        for threads in THREADS:
            outcome, out = child(["--sweep"], coretype, threads)
            row = (
                f"| {coretype} | {threads} | {architecture(out)} | {outcome} "
                f"| {last_line(out)} |"
            )
            print(row, flush=True)
            summary.append(row)

    text = "\n".join(summary) + "\n"
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write(text)
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text)
    print(text)
    return 1 if crashes else 0


if __name__ == "__main__":
    if sys.argv[1] == "--blas":
        print(blas_info())
    elif sys.argv[1] == "--test":
        run_test(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "--replay":
        replay(sys.argv[2])
    elif sys.argv[1] == "--sweep":
        sweep()
    elif sys.argv[1] == "--op":
        run_op(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "--sizes":
        sys.exit(map_sizes(sys.argv[2]))
    elif sys.argv[1].startswith("-"):
        sys.exit(f"Unknown option {sys.argv[1]}")
    else:
        sys.exit(main(sys.argv[1]))
