"""
Experiments on the access violations in numpy.linalg on Windows on ARM.

Every call runs in a child process, so that a crash ends only that process. A child
walks through the sizes and prints ``start CALL N`` before each call and ``ok CALL N``
after it. When a child crashes or hangs, its size is recorded and a new child continues
with the next size. Each size that crashed is then tried alone, in a new process.
ctypes turns an access violation in the openblas-* calls, which call numpy's
OpenBLAS directly, into an OSError with the address.

    python winarm_blas.py info

prints the builds of Python, numpy and SciPy, and what their OpenBLAS reports.

    python winarm_blas.py map REPORT [options]

maps the sizes at which calls crash. The results are printed as they come, and written
to REPORT and to the job summary. It exits with 0 when the map is complete, whether or
not anything crashed. The options are

    --calls numpy,inv       calls from CALLS, or the groups numpy, scipy and openblas
    --sizes 1-64,72-160:8   the sizes, as ranges with an optional step
    --threads 1             OPENBLAS_NUM_THREADS of the children, or unset
    --set-threads 1         call openblas_set_num_threads in the children
    --stack 64              make the calls in a thread with this stack, in MB
    --no-faulthandler       run the children without -X faulthandler
    --repeats 3             the number of calls at each size
    --budget 20             minutes, after which the remaining calls are skipped
"""

from __future__ import annotations

import concurrent.futures
import ctypes
import glob
import importlib
import json
import os
import platform
import re

# Runs this script in new processes, so that a crash does not end the experiment
import subprocess  # nosec B404
import sys
import threading
import time

import numpy as np

DEFAULT_SIZES = "1-64,72-160:8"
# Seconds for a child to walk through every size. A walk takes a few seconds.
WALK_TIMEOUT = 60
# A call is skipped after this many hangs in a row
MAX_HANGS = 2
EXCEPTIONS = {
    0xC0000005: "access violation",
    0xC00000FD: "stack overflow",
    0xC0000409: "fail fast",
    0xC000001D: "illegal instruction",
    0xC0000374: "heap corruption",
    0x80000003: "breakpoint",
}


def matrices(n: int):
    """A positive definite matrix and two right-hand sides, without calling BLAS"""
    rng = np.random.default_rng(n)
    x = rng.standard_normal((max(2 * n, 200), n))
    # einsum rather than x.T @ x, so that the only BLAS call is the one under test
    a = np.einsum("ki,kj->ij", x, x) / x.shape[0]
    return a, rng.standard_normal(n), rng.standard_normal((n, 3))


def lu_without_blas(a):
    """
    LU with partial pivoting in numpy alone, as getrf returns it

    The pivots are 0-based: row k was swapped with row piv[k].
    """
    lu = np.array(a, dtype=float, order="F")
    n = lu.shape[0]
    piv = np.zeros(n, dtype=np.int64)
    for k in range(n):
        p = k + int(np.argmax(np.abs(lu[k:, k])))
        piv[k] = p
        if p != k:
            lu[[k, p], :] = lu[[p, k], :]
        lu[k + 1 :, k] /= lu[k, k]
        lu[k + 1 :, k + 1 :] -= lu[k + 1 :, k][:, None] * lu[k, k + 1 :][None, :]
    return lu, piv


def mul(a, b):
    """a @ b without BLAS"""
    return np.einsum("ik,k...->i...", a, b)


def check(ok: bool, what: str) -> None:
    if not ok:
        raise RuntimeError(f"wrong result: {what}")


def close(x, y) -> bool:
    return bool(np.max(np.abs(x - y)) <= 1e-8 * max(1.0, float(np.max(np.abs(y)))))


class OpenBLAS:
    """The OpenBLAS of numpy or SciPy, called directly with ctypes"""

    def __init__(self, package: str = "numpy") -> None:
        module = importlib.import_module(package)
        site = os.path.dirname(os.path.dirname(module.__file__))
        pattern = os.path.join(site, f"{package}.libs", "*openblas*")
        paths = sorted(p for p in glob.glob(pattern) if p.endswith((".dll", ".so")))
        if not paths:
            raise RuntimeError(f"no OpenBLAS in {package}.libs")
        self.path = paths[0]
        # The library that the package loaded, so this shares its state
        self.lib = ctypes.CDLL(self.path)
        # The ILP64 builds add 64_ to the names and use 64-bit integers
        self.ilp64 = self.fortran("dgetrf").__name__.endswith("64_")
        self.int = ctypes.c_int64 if self.ilp64 else ctypes.c_int32
        self.np_int = np.int64 if self.ilp64 else np.int32

    def _find(self, names: list[str]):
        for name in names:
            if hasattr(self.lib, name):
                return getattr(self.lib, name)
        raise RuntimeError(f"none of {names} in {self.path}")

    def fortran(self, name: str):
        """A Fortran BLAS or LAPACK function"""
        return self._find(
            [f"scipy_{name}_64_", f"scipy_{name}_", f"{name}_64_", f"{name}_"]
        )

    def c(self, name: str, restype=ctypes.c_int):
        """A C function of OpenBLAS, such as openblas_get_config"""
        func = self._find([f"scipy_{name}64_", f"scipy_{name}", f"{name}64_", name])
        func.restype = restype
        return func

    def runtime(self) -> dict:
        def text(name: str) -> str:
            return self.c(name, ctypes.c_char_p)().decode().strip()

        return {
            "library": os.path.basename(self.path),
            "ilp64": self.ilp64,
            "config": text("openblas_get_config"),
            "corename": text("openblas_get_corename"),
            "num_threads": self.c("openblas_get_num_threads")(),
            "parallel": self.c("openblas_get_parallel")(),
        }

    def set_num_threads(self, threads: int) -> None:
        func = self.c("openblas_set_num_threads", None)
        func.argtypes = [ctypes.c_int]
        func(threads)

    # Arguments: Fortran passes everything by reference, and the length of each
    # character argument after the others
    def i(self, value: int):
        return ctypes.byref(self.int(value))

    @staticmethod
    def d(value: float):
        return ctypes.byref(ctypes.c_double(value))

    @staticmethod
    def ch(value: str):
        return ctypes.c_char_p(value.encode())

    @staticmethod
    def ptr(arr):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def iptr(self, arr):
        return arr.ctypes.data_as(ctypes.POINTER(self.int))

    def pivots(self, piv):
        """0-based pivots as LAPACK's 1-based ipiv"""
        return np.asarray(piv + 1, dtype=self.np_int)

    def _info(self, info, name: str) -> None:
        if info.value != 0:
            raise RuntimeError(f"{name} returned info={info.value}")

    def call(self, name: str, a, b, bb, lu=None, piv=None) -> None:
        n = a.shape[0]
        one = ctypes.c_size_t(1)
        info = self.int(0)
        if name == "dgetrf":
            out = np.array(a, order="F")
            ipiv = np.zeros(n, dtype=self.np_int)
            self.fortran("dgetrf")(
                self.i(n), self.i(n), self.ptr(out), self.i(n), self.iptr(ipiv),
                ctypes.byref(info),
            )  # fmt: skip
            self._info(info, name)
            perm = np.array(a)
            for k, p in enumerate(ipiv - 1):
                perm[[k, p], :] = perm[[p, k], :]
            low = np.tril(out, -1) + np.eye(n)
            check(close(mul(low, np.triu(out)), perm), "P L U != A")
        elif name == "dgetrs":
            x = np.array(b, order="F")
            self.fortran("dgetrs")(
                self.ch("N"), self.i(n), self.i(1), self.ptr(lu), self.i(n),
                self.iptr(self.pivots(piv)), self.ptr(x), self.i(n), ctypes.byref(info),
                one,
            )  # fmt: skip
            self._info(info, name)
            check(close(mul(a, x), b), "A x != b")
        elif name == "dgesv":
            work = np.array(a, order="F")
            x = np.array(b, order="F")
            ipiv = np.zeros(n, dtype=self.np_int)
            self.fortran("dgesv")(
                self.i(n), self.i(1), self.ptr(work), self.i(n), self.iptr(ipiv),
                self.ptr(x), self.i(n), ctypes.byref(info),
            )  # fmt: skip
            self._info(info, name)
            check(close(mul(a, x), b), "A x != b")
        elif name == "dgetri":
            inv = np.array(lu, order="F")
            lwork = max(1, 64 * n)
            work = np.zeros(lwork)
            self.fortran("dgetri")(
                self.i(n), self.ptr(inv), self.i(n), self.iptr(self.pivots(piv)),
                self.ptr(work), self.i(lwork), ctypes.byref(info),
            )  # fmt: skip
            self._info(info, name)
            check(close(mul(a, inv), np.eye(n)), "A inv(A) != I")
        elif name == "dpotrf":
            low = np.array(a, order="F")
            self.fortran("dpotrf")(
                self.ch("L"), self.i(n), self.ptr(low), self.i(n), ctypes.byref(info),
                one,
            )  # fmt: skip
            self._info(info, name)
            low = np.tril(low)
            check(close(mul(low, low.T), a), "L L' != A")
        elif name == "dtrsm":
            # The unit lower triangle of the LU, as getrs uses it
            x = np.array(bb, order="F")
            self.fortran("dtrsm")(
                self.ch("L"), self.ch("L"), self.ch("N"), self.ch("U"), self.i(n),
                self.i(3), self.d(1.0), self.ptr(lu), self.i(n), self.ptr(x),
                self.i(n), one, one, one, one,
            )  # fmt: skip
            low = np.tril(lu, -1) + np.eye(n)
            check(close(mul(low, x), bb), "L X != B")
        elif name == "dlaswp":
            x = np.array(bb, order="F")
            self.fortran("dlaswp")(
                self.i(3), self.ptr(x), self.i(n), self.i(1), self.i(n),
                self.iptr(self.pivots(piv)), self.i(1),
            )  # fmt: skip
            expected = np.array(bb)
            for k, p in enumerate(piv):
                expected[[k, p], :] = expected[[p, k], :]
            check(bool(np.array_equal(x, expected)), "rows not swapped")
        elif name == "dgemm":
            c = np.zeros((n, n), order="F")
            af = np.array(a, order="F")
            self.fortran("dgemm")(
                self.ch("N"), self.ch("N"), self.i(n), self.i(n), self.i(n),
                self.d(1.0), self.ptr(af), self.i(n), self.ptr(af), self.i(n),
                self.d(0.0), self.ptr(c), self.i(n), one, one,
            )  # fmt: skip
            check(close(c, mul(a, a)), "C != A A")
        elif name == "dgemv":
            y = np.zeros(n)
            af = np.array(a, order="F")
            x = np.array(b)
            self.fortran("dgemv")(
                self.ch("N"), self.i(n), self.i(n), self.d(1.0), self.ptr(af),
                self.i(n), self.ptr(x), self.i(1), self.d(0.0), self.ptr(y), self.i(1),
                one,
            )  # fmt: skip
            check(close(y, mul(a, b)), "y != A x")
        else:
            raise ValueError(name)


def numpy_call(name: str, a, b, bb) -> None:
    linalg = np.linalg
    if name == "inv-identity":
        linalg.inv(np.eye(a.shape[0]))
    elif name == "inv":
        linalg.inv(a)
    elif name == "solve":
        linalg.solve(a, b)
    elif name == "solve-3":
        linalg.solve(a, bb)
    elif name == "det":
        linalg.det(a)
    elif name == "cholesky":
        linalg.cholesky(a)
    elif name == "qr":
        linalg.qr(a)
    elif name == "eigh":
        linalg.eigh(a)
    elif name == "svd":
        linalg.svd(a)
    elif name == "lstsq":
        linalg.lstsq(a, b, rcond=None)
    elif name == "matmul":
        np.matmul(a, a)
    elif name == "matvec":
        np.matmul(a, b)
    else:
        raise ValueError(name)


def scipy_call(name: str, a, b, bb, lu=None, piv=None) -> None:
    import scipy.linalg  # noqa: PLC0415
    from scipy.linalg import blas, lapack  # noqa: PLC0415

    info = 0
    if name == "inv":
        scipy.linalg.inv(a)
    elif name == "solve":
        scipy.linalg.solve(a, b)
    elif name == "lu_factor":
        scipy.linalg.lu_factor(a)
    elif name == "lu_solve":
        scipy.linalg.lu_solve((lu, piv), b)
    elif name == "cho_factor":
        scipy.linalg.cho_factor(a)
    elif name == "dgetrf":
        *_, info = lapack.dgetrf(a)
    elif name == "dgetrs":
        _, info = lapack.dgetrs(lu, piv, b)
    elif name == "dgesv":
        *_, info = lapack.dgesv(a, b)
    elif name == "dgetri":
        _, info = lapack.dgetri(lu, piv)
    elif name == "dpotrf":
        _, info = lapack.dpotrf(a, lower=1)
    elif name == "dtrsm":
        blas.dtrsm(1.0, lu, bb, side=0, lower=1, trans_a=0, diag=1)
    elif name == "dlaswp":
        lapack.dlaswp(bb, piv)
    elif name == "dgemm":
        blas.dgemm(1.0, a, a)
    elif name == "dgemv":
        blas.dgemv(1.0, a, b)
    else:
        raise ValueError(name)
    if info != 0:
        raise RuntimeError(f"{name} returned info={info}")


NUMPY_CALLS = [
    "inv-identity",
    "inv",
    "solve",
    "solve-3",
    "det",
    "cholesky",
    "qr",
    "eigh",
    "svd",
    "lstsq",
    "matmul",
    "matvec",
]
SCIPY_CALLS = [
    "inv",
    "solve",
    "lu_factor",
    "lu_solve",
    "cho_factor",
    "dgetrf",
    "dgetrs",
    "dgesv",
    "dgetri",
    "dpotrf",
    "dtrsm",
    "dlaswp",
    "dgemm",
    "dgemv",
]
OPENBLAS_CALLS = [
    "dgetrf",
    "dgetrs",
    "dgesv",
    "dgetri",
    "dpotrf",
    "dtrsm",
    "dlaswp",
    "dgemm",
    "dgemv",
]
# numpy.linalg as it is, SciPy's OpenBLAS through scipy.linalg, and numpy's OpenBLAS
# called directly. selftest crashes at 5 on purpose, to check that crashes are found.
CALLS = (
    NUMPY_CALLS
    + [f"scipy-{c}" for c in SCIPY_CALLS]
    + [f"openblas-{c}" for c in OPENBLAS_CALLS]
    + ["selftest"]
)
GROUPS = {
    "numpy": NUMPY_CALLS,
    "scipy": [f"scipy-{c}" for c in SCIPY_CALLS],
    "openblas": [f"openblas-{c}" for c in OPENBLAS_CALLS],
}
NEEDS_LU = {"scipy-lu_solve", "scipy-dgetrs", "scipy-dgetri", "scipy-dtrsm"}
NEEDS_LU |= {"scipy-dlaswp", "openblas-dgetrs", "openblas-dgetri"}
NEEDS_LU |= {"openblas-dtrsm", "openblas-dlaswp"}


def parse_sizes(text: str) -> list[int]:
    sizes: list[int] = []
    for part in text.split(","):
        rng, _, step = part.partition(":")
        first, _, last = rng.partition("-")
        sizes += range(int(first), int(last or first) + 1, int(step or 1))
    return sizes


def ranges(values) -> str:
    """Sizes as ranges, such as 1-5, 8"""
    parts: list[list[int]] = []
    for v in sorted(values):
        if parts and parts[-1][1] == v - 1:
            parts[-1][1] = v
        else:
            parts.append([v, v])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in parts)


def runtime(package: str) -> dict:
    try:
        return OpenBLAS(package).runtime()
    except (OSError, RuntimeError, ValueError) as exc:
        return {"error": repr(exc)}


def show_info() -> None:
    """What the builds are, and what OpenBLAS reports at run time"""
    import sysconfig  # noqa: PLC0415

    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    result = {
        "python": sys.version.split()[0],
        "platform": sysconfig.get_platform(),
        "os": platform.platform(),
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "env": {
            k: v
            for k, v in sorted(os.environ.items())
            if k.startswith(("OPENBLAS", "OMP"))
        },
        "numpy": np.__version__,
        "numpy blas": f"{blas.get('name')} {blas.get('version')}",
        "numpy openblas": runtime("numpy"),
    }
    try:
        import scipy  # noqa: PLC0415

        result["scipy"] = scipy.__version__
        result["scipy openblas"] = runtime("scipy")
    except ImportError:
        pass
    print("INFO", json.dumps(result), flush=True)


def walk(call: str, sizes: list[int], repeats: int, set_threads: int | None) -> None:
    """Make the call at each size, in this process"""
    if call in NUMPY_CALLS or call == "selftest":
        library, name = "numpy", call
    else:
        library, _, name = call.partition("-")
    openblas = OpenBLAS("numpy") if library == "openblas" else None
    if set_threads is not None:
        packages = ["numpy", "scipy"] if library == "scipy" else ["numpy"]
        for package in packages:
            lib = OpenBLAS(package)
            before = lib.c("openblas_get_num_threads")()
            lib.set_num_threads(set_threads)
            after = lib.c("openblas_get_num_threads")()
            print(
                f"{package}: openblas_set_num_threads {before} -> {after}", flush=True
            )
    for n in sizes:
        a, b, bb = matrices(n)
        lu, piv = lu_without_blas(a) if call in NEEDS_LU else (None, None)
        print(f"start {call} {n}", flush=True)
        for _ in range(repeats):
            if call == "selftest":
                if n == 5:
                    import faulthandler  # noqa: PLC0415

                    faulthandler._read_null()  # pylint: disable=protected-access
            elif library == "numpy":
                numpy_call(name, a, b, bb)
            elif library == "scipy":
                scipy_call(name, a, b, bb, lu, piv)
            else:
                openblas.call(name, a, b, bb, lu, piv)
        print(f"ok {call} {n}", flush=True)


def walk_in_thread(stack_mb: float, *args) -> None:
    """walk in a thread with a stack of stack_mb MB"""
    threading.stack_size(int(stack_mb * 1024 * 1024))
    # result() raises the exceptions of the thread in this one
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(walk, *args).result()


class Options:
    def __init__(self, argv: list[str]) -> None:
        self.calls = NUMPY_CALLS
        self.sizes = parse_sizes(DEFAULT_SIZES)
        self.threads: str | None = None
        self.set_threads: int | None = None
        self.stack: float | None = None
        self.faulthandler = True
        self.repeats = 3
        self.budget = 20.0
        args = iter(argv)
        for arg in args:
            if arg == "--calls":
                names = next(args).split(",")
                self.calls = [c for name in names for c in GROUPS.get(name, [name])]
                unknown = sorted(set(self.calls) - set(CALLS))
                if unknown:
                    raise SystemExit(f"unknown calls: {unknown}")
            elif arg == "--sizes":
                self.sizes = parse_sizes(next(args))
            elif arg == "--threads":
                self.threads = next(args)
            elif arg == "--set-threads":
                self.set_threads = int(next(args))
            elif arg == "--stack":
                self.stack = float(next(args))
            elif arg == "--no-faulthandler":
                self.faulthandler = False
            elif arg == "--repeats":
                self.repeats = int(next(args))
            elif arg == "--budget":
                self.budget = float(next(args))
            else:
                raise SystemExit(f"unknown option: {arg}")

    def walk_args(self) -> list[str]:
        args = ["--repeats", str(self.repeats)]
        if self.set_threads is not None:
            args += ["--set-threads", str(self.set_threads)]
        if self.stack is not None:
            args += ["--stack", str(self.stack)]
        return args

    def describe(self) -> str:
        parts = [f"OPENBLAS_NUM_THREADS {self.threads or 'as in the job'}"]
        if self.set_threads is not None:
            parts.append(f"openblas_set_num_threads({self.set_threads})")
        if self.stack is not None:
            parts.append(f"in a thread with a {self.stack:g} MB stack")
        parts.append("-X faulthandler" if self.faulthandler else "no faulthandler")
        parts.append(f"{self.repeats} calls at each size")
        return ", ".join(parts)


def child(args: list[str], opts: Options, timeout: float) -> tuple[str, str]:
    """Run this script in a new process and return its outcome and output"""
    env = os.environ.copy()
    if opts.threads == "unset":
        env.pop("OPENBLAS_NUM_THREADS", None)
        env.pop("OMP_NUM_THREADS", None)
    elif opts.threads is not None:
        env["OPENBLAS_NUM_THREADS"] = env["OMP_NUM_THREADS"] = opts.threads
    cmd = [sys.executable, os.path.abspath(__file__), *args]
    if opts.faulthandler:
        cmd[1:1] = ["-X", "faulthandler"]
    try:
        # Only this script, with arguments from this script, and without a shell
        proc = subprocess.run(  # nosec B603
            cmd, capture_output=True, text=True, env=env, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return "hang", out
    out = proc.stdout + proc.stderr
    code = proc.returncode & 0xFFFFFFFF
    if code == 0:
        return "ok", out
    if code < 256:
        errors = [line for line in out.splitlines() if line.strip()]
        return f"error: {errors[-1] if errors else code}", out
    return f"0x{code:08X} {EXCEPTIONS.get(code, '')}".strip(), out


def is_crash(outcome: str) -> bool:
    """An exception of Windows, also when ctypes catches it and raises OSError"""
    return outcome.startswith("0x") or "OSError: exception" in outcome


def map_call(call: str, opts: Options) -> dict[int, str]:
    """The outcome at each size, walking on after each crash"""
    outcomes: dict[int, str] = {}
    todo = list(opts.sizes)
    hangs = 0
    while todo:
        sizes = ",".join(map(str, todo))
        outcome, out = child(
            ["walk", call, sizes, *opts.walk_args()], opts, WALK_TIMEOUT
        )
        for line in out.splitlines():
            if line.startswith("openblas_set_num_threads") or ": openblas_set" in line:
                print(f"  {line}", flush=True)
        done = [
            int(n)
            for n in re.findall(rf"^ok {re.escape(call)} (\d+)$", out, re.MULTILINE)
        ]
        started = [
            int(n)
            for n in re.findall(rf"^start {re.escape(call)} (\d+)$", out, re.MULTILINE)
        ]
        outcomes.update(dict.fromkeys(done, "ok"))
        if outcome == "ok":
            break
        if not started or started[-1] in done:
            # Not in a call: an error of this script, or of an import
            print(f"{call}: {outcome}\n{out[-2000:]}", flush=True)
            outcomes.update(dict.fromkeys(todo[len(done) :], f"not run ({outcome})"))
            break
        n = started[-1]
        outcomes[n] = outcome
        print(f"{call} {n}: {outcome}", flush=True)
        hangs = hangs + 1 if outcome == "hang" else 0
        todo = todo[todo.index(n) + 1 :]
        if hangs >= MAX_HANGS:
            outcomes.update(dict.fromkeys(todo, "not run (hangs)"))
            break
    return outcomes


def run_map(report: str, opts: Options) -> int:
    start = time.monotonic()
    outcome, out = child(["info"], opts, WALK_TIMEOUT)
    lines = [line for line in out.splitlines() if line.startswith("INFO ")]
    details = json.loads(lines[0][5:]) if lines else {"info": outcome}
    print(json.dumps(details, indent=2), flush=True)
    header = [
        "## numpy.linalg on Windows on ARM",
        "",
        f"{opts.describe()}. Sizes {ranges(opts.sizes)}.",
        "",
        "```",
        json.dumps(details, indent=2),
        "```",
        "",
        "| Call | Crashes | Alone, in a new process | Other |",
        "|---|---|---|---|",
    ]
    rows: list[str] = []
    for call in opts.calls:
        if time.monotonic() - start > 60 * opts.budget:
            rows.append(f"| {call} | | | not run (out of time) |")
            continue
        outcomes = map_call(call, opts)
        crashes = {n: o for n, o in outcomes.items() if is_crash(o)}
        alone = []
        for n in sorted(crashes):
            result, _ = child(
                ["walk", call, str(n), *opts.walk_args()], opts, WALK_TIMEOUT
            )
            print(f"{call} {n} alone: {result}", flush=True)
            if result != "ok":
                alone.append(n)
        kinds = sorted(set(crashes.values()))
        other = sorted(
            {
                f"{o} at {n}"
                for n, o in outcomes.items()
                if o != "ok" and n not in crashes
            }
        )
        row = (
            f"| {call} | {ranges(crashes) or 'none'} {' '.join(kinds)} "
            f"| {ranges(alone) if crashes else ''} | {'; '.join(other[:5])} |"
        )
        print(row, flush=True)
        rows.append(row)
        # Written after every call, so that a job that times out leaves the results
        with open(report, "w", encoding="utf-8") as f:
            f.write("\n".join(header + rows) + "\n")
    text = "\n".join(header + rows) + "\n"
    with open(report, "w", encoding="utf-8") as f:
        f.write(text)
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text)
    print(text, flush=True)
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["info"]:
        show_info()
        return 0
    if argv[:1] == ["walk"] and len(argv) >= 3:
        opts = Options(argv[3:])
        args = (argv[1], parse_sizes(argv[2]), opts.repeats, opts.set_threads)
        if opts.stack is not None:
            walk_in_thread(opts.stack, *args)
        else:
            walk(*args)
        return 0
    if argv[:1] == ["map"] and len(argv) >= 2:
        return run_map(argv[1], Options(argv[2:]))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
