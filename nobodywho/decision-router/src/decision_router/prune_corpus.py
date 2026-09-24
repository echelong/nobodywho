"""Representative long tool outputs with the facts a pruner must never drop.

Used by `decision prune test` and the test suite. Each case is
(name, command, output, must_keep) where every must_keep string has to
survive pruning verbatim.
"""

from __future__ import annotations

import random


def _pytest() -> tuple[str, str, str, list[str]]:
    r = random.Random(1)
    lines = ["============================= test session starts ==============================",
             "platform linux -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0",
             "rootdir: /home/dev/app", "collected 313 items", ""]  # fmt: skip
    for i in range(312):
        mod = r.choice(["test_models", "test_views", "test_utils", "test_forms"])
        lines.append(f"tests/{mod}.py::test_case_{i:03d} PASSED{' ' * 20}[{i * 100 // 313:3d}%]")
    lines += [
        "tests/test_api.py::test_login FAILED                                      [ 99%]",
        "", "=================================== FAILURES ===================================",
        "_________________________________ test_login __________________________________",
        "", "    def test_login(client):",
        '        response = client.post("/login", data={"user": "a", "pw": "b"})',
        ">       assert response.status_code == 200",
        "E       assert 500 == 200",
        "E        +  where 500 = <Response [500]>.status_code",
        "", "tests/test_api.py:42: AssertionError",
        "=========================== short test summary info ============================",
        "FAILED tests/test_api.py::test_login - assert 500 == 200",
        "======================== 1 failed, 312 passed in 12.31s ========================",
    ]  # fmt: skip
    return ("pytest", "python -m pytest -q tests", "\n".join(lines),
            ["tests/test_api.py::test_login", "assert 500 == 200", "tests/test_api.py:42",
             "1 failed, 312 passed"])  # fmt: skip


def _typecheck() -> tuple[str, str, str, list[str]]:
    lines = [f"[{i:4d}/1200] Checking src/module_{i}.ts" for i in range(1, 1200)]
    lines.insert(640, "src/auth/session.ts(88,14): error TS2322: Type 'string | undefined' "
                      "is not assignable to type 'string'.")  # fmt: skip
    lines.insert(900, "src/api/client.ts(12,3): error TS2554: Expected 2 arguments, but got 1.")
    lines += ["", "Found 2 errors in 2 files.", "", "Errors  Files",
              "     1  src/auth/session.ts:88", "     1  src/api/client.ts:12"]  # fmt: skip
    return ("typecheck", "npx tsc --noEmit", "\n".join(lines),
            ["src/auth/session.ts(88,14): error TS2322", "src/api/client.ts(12,3): error TS2554",
             "Found 2 errors in 2 files."])  # fmt: skip


def _rust_build() -> tuple[str, str, str, list[str]]:
    lines = [f"   Compiling crate_{i} v0.{i % 9}.{i % 7}" for i in range(400)]
    lines += [
        "warning: unused variable: `retries`",
        "  --> src/net/retry.rs:57:9",
        "   |",
        "57 |     let retries = 3;",
        "   |         ^^^^^^^ help: if this is intentional, prefix it with an underscore: `_retries`",
        "",
        "error[E0308]: mismatched types",
        "  --> src/main.rs:120:18",
        "    |",
        "120 |     let port: u16 = config.port;",
        "    |               ---   ^^^^^^^^^^^ expected `u16`, found `u32`",
        "",
        'error: could not compile `server` (bin "server") due to 1 previous error; 1 warning emitted',
    ]
    return ("rust-build", "cargo build --release", "\n".join(lines),
            ["error[E0308]: mismatched types", "src/main.rs:120:18", "src/net/retry.rs:57:9",
             "could not compile `server`", "warning: unused variable: `retries`"])  # fmt: skip


def _npm_install() -> tuple[str, str, str, list[str]]:
    lines = [f"npm http fetch GET 200 https://registry.npmjs.org/pkg-{i} {i % 90}ms (cache hit)"
             for i in range(900)]  # fmt: skip
    lines.insert(300, "npm WARN deprecated request@2.88.2: request has been deprecated")
    lines += [
        "npm ERR! code ERESOLVE",
        "npm ERR! ERESOLVE unable to resolve dependency tree",
        "npm ERR! Found: react@19.1.0",
        'npm ERR! Could not resolve dependency: peer react@"^18.0.0" from legacy-ui@3.2.1',
        "npm ERR! A complete log of this run can be found in: /home/dev/.npm/_logs/2026-09-24-debug.log",
    ]
    return ("npm-install", "npm install", "\n".join(lines),
            ["npm ERR! code ERESOLVE", "peer react@\"^18.0.0\" from legacy-ui@3.2.1",
             "npm WARN deprecated request@2.88.2"])  # fmt: skip


def _java_trace() -> tuple[str, str, str, list[str]]:
    lines = [f"2026-09-24 10:00:{i % 60:02d}.{i:03d} INFO  [main] c.e.app.Loader - loaded bean #{i}"
             for i in range(700)]  # fmt: skip
    lines += [
        "2026-09-24 10:01:02.001 ERROR [main] c.e.app.Server - startup failed",
        "java.lang.IllegalStateException: pool not initialised",
        "\tat com.example.db.Pool.acquire(Pool.java:88)",
        "\tat com.example.app.Server.start(Server.java:41)",
        "Caused by: java.lang.NullPointerException: config.url is null",
        "\tat com.example.db.Pool.<init>(Pool.java:23)",
        "Process finished with exit code 1",
    ]
    return ("java-trace", "./gradlew run", "\n".join(lines),
            ["java.lang.IllegalStateException: pool not initialised", "Pool.java:88",
             "Caused by: java.lang.NullPointerException: config.url is null",
             "Process finished with exit code 1"])  # fmt: skip


def _git_diff() -> tuple[str, str, str, list[str]]:
    lines = [" src/app/views.py        | 120 ++++++++++-----", " src/app/models.py       |  14 +-",
             " tests/test_views.py     |  60 ++++++", " 3 files changed, 170 insertions(+), 24 deletions(-)",
             "", "diff --git a/src/app/views.py b/src/app/views.py", "index 3f1c2aa..9b0d411 100644",
             "--- a/src/app/views.py", "+++ b/src/app/views.py"]  # fmt: skip
    for i in range(0, 600, 20):
        lines.append(f"@@ -{i + 1},12 +{i + 1},14 @@ def view_{i}(request):")
        lines += [f"     context_{i}_{j} = build({j})" for j in range(8)]
        lines += [f"+    added_{i} = compute({i})", f"-    removed_{i} = legacy({i})"]
    return ("git-diff", "git diff --stat && git diff", "\n".join(lines),
            ["3 files changed, 170 insertions(+), 24 deletions(-)", "diff --git a/src/app/views.py",
             "src/app/models.py"])  # fmt: skip


def _make_fail() -> tuple[str, str, str, list[str]]:
    lines = [f"cc -O2 -c src/obj_{i}.c -o build/obj_{i}.o" for i in range(500)]
    lines += ["src/parser.c:211:17: error: 'tok' undeclared (first use in this function)",
              "make: *** [Makefile:34: build/parser.o] Error 1"]  # fmt: skip
    return ("make", "make -j8", "\n".join(lines),
            ["src/parser.c:211:17: error: 'tok' undeclared", "make: *** [Makefile:34: build/parser.o] Error 1"])  # fmt: skip


def _search() -> tuple[str, str, str, list[str]]:
    lines = [f"src/pkg_{i // 10}/mod_{i}.py:{i % 300 + 1}:    value = legacy_lookup(key_{i})"
             for i in range(1500)]  # fmt: skip
    return ("search", "rg -n legacy_lookup", "\n".join(lines),
            ["src/pkg_0/mod_0.py:1:", "src/pkg_149/mod_1499.py:"])  # fmt: skip


def _lint() -> tuple[str, str, str, list[str]]:
    lines = [f"src/generated/module_{i}.py:1:1: I001 Import block is un-sorted or un-formatted"
             for i in range(850)]  # fmt: skip
    lines += [
        "src/api/routes.py:73:9: F821 Undefined name `current_user`",
        "src/db/session.py:118:5: B904 Within an `except` clause, raise exceptions with `raise ... from err`",
        "Found 852 errors.",
        "ruff check failed with exit code 1",
    ]
    return ("lint", "ruff check .", "\n".join(lines),
            ["src/api/routes.py:73:9: F821", "current_user", "src/db/session.py:118:5: B904",
             "Found 852 errors.", "exit code 1"])  # fmt: skip


def _stdout_stderr() -> tuple[str, str, str, list[str]]:
    stdout = [f"[worker-{i % 8}] processed batch {i:05d}: {i * 24} records"
              for i in range(1200)]  # fmt: skip
    stderr = [f"2026-09-24T11:{i // 60:02d}:{i % 60:02d}Z DEBUG retry loop {i}"
              for i in range(500)]  # fmt: skip
    stderr += [
        "2026-09-24T11:09:01Z ERROR worker failed: ConnectionResetError: peer reset",
        '  File "src/queue/consumer.py", line 207, in consume_batch',
        "    raise ConnectionResetError(104, 'Connection reset by peer')",
        "ConnectionResetError: [Errno 104] Connection reset by peer",
        "Process exited with status 23",
    ]
    output = "STDOUT:\n" + "\n".join(stdout) + "\nSTDERR:\n" + "\n".join(stderr)
    return ("stdout-stderr", "./scripts/consume --workers 8", output,
            ["STDOUT:", "STDERR:", "src/queue/consumer.py", "line 207",
             "ConnectionResetError: [Errno 104] Connection reset by peer",
             "status 23"])  # fmt: skip


CASES = (
    _pytest,
    _typecheck,
    _rust_build,
    _npm_install,
    _java_trace,
    _git_diff,
    _make_fail,
    _search,
    _lint,
    _stdout_stderr,
)


def cases() -> list[tuple[str, str, str, list[str]]]:
    return [make() for make in CASES]
