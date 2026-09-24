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


# ---------------------------------------------------------------- sized outputs (latency)


def pytest_verbose(n_tests: int, seed: int = 7) -> tuple[str, str, str, list[str]]:
    """`pytest -v`: many distinct passing test lines around one failure."""
    r = random.Random(seed)
    mods = ["test_models", "test_views", "test_api", "test_forms", "test_utils", "test_billing",
            "test_auth", "test_cache", "test_orders", "test_search"]  # fmt: skip
    verbs = ["creates", "rejects", "updates", "lists", "renders", "parses", "validates", "caches",
             "serializes", "handles"]  # fmt: skip
    nouns = ["user", "invoice", "order", "session", "token", "profile", "cart", "report",
             "address", "coupon", "webhook", "payload"]  # fmt: skip
    lines = ["============================= test session starts ==============================",
             "platform linux -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0 -- /usr/bin/python3",
             "cachedir: .pytest_cache", "rootdir: /home/dev/shop", "configfile: pyproject.toml",
             f"collected {n_tests + 1} items", ""]  # fmt: skip
    fail_at = n_tests * 2 // 3
    for i in range(n_tests):
        if i == fail_at:
            lines.append("tests/test_orders.py::test_refund_rounds_half_even FAILED"
                         f"{' ' * 18}[{i * 100 // n_tests:3d}%]")  # fmt: skip
        name = f"test_{r.choice(verbs)}_{r.choice(nouns)}_{r.choice(nouns)}"
        extra = r.choice(["", "[sqlite]", "[postgres]", "[en-US]", "[de-DE]", "[fast]"])
        lines.append(
            f"tests/{r.choice(mods)}.py::{name}{extra} PASSED{' ' * 20}[{i * 100 // n_tests:3d}%]"
        )
    lines += [
        "", "=================================== FAILURES ===================================",
        "________________________ test_refund_rounds_half_even _________________________",
        "", "    def test_refund_rounds_half_even():",
        "        order = make_order(total=Decimal('10.05'))",
        ">       assert refund(order, ratio=Decimal('0.5')) == Decimal('5.02')",
        "E       AssertionError: assert Decimal('5.03') == Decimal('5.02')",
        "", "tests/test_orders.py:118: AssertionError",
        "=========================== short test summary info ============================",
        ("FAILED tests/test_orders.py::test_refund_rounds_half_even - AssertionError: "
         "assert Decimal('5.03') == Decimal('5.02')"),
        f"======================== 1 failed, {n_tests} passed in 41.07s ========================",
    ]  # fmt: skip
    return (f"pytest-v-{n_tests}", "python -m pytest -v", "\n".join(lines),
            ["tests/test_orders.py::test_refund_rounds_half_even",
             "AssertionError: assert Decimal('5.03') == Decimal('5.02')",
             "tests/test_orders.py:118", f"1 failed, {n_tests} passed"])  # fmt: skip


def verbose_build(n_units: int, seed: int = 11) -> tuple[str, str, str, list[str]]:
    """`make V=1`: many distinct compiler command lines, one warning, one error."""
    r = random.Random(seed)
    dirs = ["src/core", "src/net", "src/io", "src/util", "lib/codec", "lib/crypto", "src/ui"]
    stems = ["buffer", "socket", "parser", "reader", "writer", "hash", "config", "queue",
             "event", "timer", "codec", "frame", "stream", "pool", "table", "string"]  # fmt: skip
    flags = ["-O2", "-g", "-Wall", "-Wextra", "-fPIC", "-DNDEBUG", "-std=c11", "-pthread",
             "-Iinclude", "-Ithird_party/zlib", "-MMD", "-MP"]  # fmt: skip
    lines = ["make[1]: Entering directory '/home/dev/proj/build'",
             "-- The C compiler identification is GNU 15.2.1",
             "-- Configuring done (0.4s)", "-- Generating done (0.1s)"]  # fmt: skip
    warn_at, err_at = n_units // 3, n_units * 5 // 6
    for i in range(n_units):
        d, s = r.choice(dirs), r.choice(stems) + r.choice(["", "_v2", "_impl", "_ops", "_test"])
        obj = f"obj/{d.replace('/', '_')}_{s}.o"
        lines.append(f"gcc {' '.join(r.sample(flags, 7))} -c {d}/{s}.c -o {obj}")
        if r.random() < 0.3:
            lines.append(f"ar rcs lib/lib{d.split('/')[-1]}.a {obj}")
        if i == warn_at:
            lines += [("src/net/socket_impl.c:214:17: warning: comparison of integer expressions "
                       "of different signedness: 'int' and 'size_t' [-Wsign-compare]"),
                      "  214 |     for (int k = 0; k < len; k++) {",
                      "      |                 ^"]  # fmt: skip
        if i == err_at:
            lines += [("lib/codec/frame_ops.c:87:5: error: implicit declaration of function "
                       "'frame_reset' [-Wimplicit-function-declaration]"),
                      "   87 |     frame_reset(f);", "      |     ^~~~~~~~~~~"]  # fmt: skip
    lines += ["make[1]: *** [Makefile:212: obj/lib_codec_frame_ops.o] Error 1",
              "make[1]: Leaving directory '/home/dev/proj/build'",
              "make: *** [Makefile:40: all] Error 2"]  # fmt: skip
    return (f"build-v-{n_units}", "make V=1", "\n".join(lines),
            ["src/net/socket_impl.c:214:17: warning", "lib/codec/frame_ops.c:87:5: error",
             "implicit declaration of function 'frame_reset'", "Makefile:212", "Error 2"])  # fmt: skip


def sized_cases() -> list[tuple[str, tuple[str, str, str, list[str]]]]:
    """(size, case): small ~17-20k, medium ~80-85k and large ~0.6-0.7M characters."""
    return [("small", pytest_verbose(180)), ("small", verbose_build(160)),
            ("medium", pytest_verbose(900)), ("medium", verbose_build(700)),
            ("large", pytest_verbose(7000)), ("large", verbose_build(6000))]  # fmt: skip


# ---------------------------------------------------------------- labelled blocks (judgement)


def judgement_cases() -> list[tuple[str, list[tuple[str, list[str]]]]]:
    """(command, [(label, block lines)]): what a local judge should keep or drop.

    `keep` blocks hold what the agent ran the command for (results, values,
    tables, requested data); `drop` blocks are routine success or progress noise.
    None of these lines is critical, so only the judge decides them.
    """
    passed = [f"tests/test_{m}.py::test_{v}_{n} PASSED{' ' * 12}[{p:3d}%]"
              for p, (m, v, n) in enumerate(zip(
                  ["api", "views", "orders", "cart", "auth", "billing", "forms", "cache"] * 3,
                  ["creates", "lists", "updates", "renders", "handles", "parses"] * 4,
                  ["user", "order", "coupon", "session", "invoice", "profile", "token"] * 4))]  # fmt: skip
    return [
        (
            "python -m pytest -v --cov=shop",
            [
                ("drop", passed[:8]),
                (
                    "keep",
                    [
                        "----------------------------- Captured stdout call -----------------------------",
                        "refund computed: amount=5.03 currency=EUR ratio=0.5",
                        "rounding mode: ROUND_HALF_UP (settings.MONEY_ROUNDING)",
                        "order id=8841 total=10.05 items=3 customer=acme-gmbh",
                    ],
                ),
                ("drop", passed[8:16]),
                (
                    "keep",
                    [
                        "---------- coverage: platform linux, python 3.14.7 -----------",
                        "Name                 Stmts   Miss  Cover",
                        "shop/orders.py         212     31    85%",
                        "shop/refunds.py         64     22    66%",
                        "TOTAL                 1840    203    89%",
                    ],
                ),
                ("drop", passed[16:24]),
            ],
        ),
        (
            "pip install -r requirements.txt",
            [
                (
                    "drop",
                    [
                        "Collecting requests>=2.31",
                        "  Downloading requests-2.32.3-py3-none-any.whl (64 kB)",
                        "Collecting urllib3<3,>=1.21.1",
                        "  Downloading urllib3-2.3.0-py3-none-any.whl (128 kB)",
                        "Collecting idna<4,>=2.5",
                        "  Using cached idna-3.10-py3-none-any.whl (70 kB)",
                    ],
                ),
                (
                    "drop",
                    [
                        "Requirement already satisfied: certifi>=2017.4.17 in ./.venv/lib/python3.14/site-packages",
                        "Requirement already satisfied: charset-normalizer<4,>=2 in ./.venv/lib/python3.14/site-packages",
                        "Requirement already satisfied: packaging>=23 in ./.venv/lib/python3.14/site-packages",
                    ],
                ),
                (
                    "keep",
                    [
                        "Installing collected packages: urllib3, idna, requests",
                        "Successfully installed idna-3.10 requests-2.32.3 urllib3-2.3.0",
                    ],
                ),
            ],
        ),
        (
            "npm run build",
            [
                (
                    "drop",
                    [
                        f"transforming ({n}) src/components/{c}.tsx"
                        for n, c in zip(
                            range(40, 400, 60), ["Button", "Modal", "Table", "Form", "Nav", "Card"]
                        )
                    ],
                ),
                (
                    "keep",
                    [
                        "dist/index.html                   0.46 kB │ gzip:   0.30 kB",
                        "dist/assets/index-4f1c2a.css     21.87 kB │ gzip:   4.91 kB",
                        "dist/assets/index-9be21d.js     512.31 kB │ gzip: 160.02 kB",
                        "✓ built in 7.42s",
                    ],
                ),
                (
                    "drop",
                    [
                        "rendering chunks...",
                        "computing gzip size...",
                        "copying public assets...",
                        "cleaning dist before build...",
                    ],
                ),
            ],
        ),
        (
            "git log --oneline -n 12",
            [
                (
                    "keep",
                    [
                        "9c631a7 Add local-first routing and a separate extractive prune operation",
                        "6e56a63 Keep the local model loaded and harden the Cline rule",
                        "6035572 Add experimental provider-neutral decision router",
                        "20ca085 Remove broken AddBos logic",
                    ],
                ),
                (
                    "keep",
                    [
                        "c14d3b5 Improve readme",
                        "ddf3840 Complete rewrite of godot bindings",
                        "9dc04ef chore: Fix CHANGELOG.md after previous release",
                        "9e36241 Update llama-cpp-2 to 0.1.156",
                    ],
                ),
            ],
        ),
        (
            "cargo build --release",
            [
                (
                    "drop",
                    [
                        f"   Compiling {c} v{v}"
                        for c, v in [
                            ("proc-macro2", "1.0.92"),
                            ("unicode-ident", "1.0.14"),
                            ("quote", "1.0.37"),
                            ("syn", "2.0.90"),
                            ("serde_derive", "1.0.215"),
                            ("serde", "1.0.215"),
                        ]
                    ],
                ),
                (
                    "drop",
                    [
                        f"   Compiling {c} v{v}"
                        for c, v in [
                            ("tokio", "1.42.0"),
                            ("mio", "1.0.3"),
                            ("bytes", "1.9.0"),
                            ("hyper", "1.5.1"),
                            ("tower", "0.5.1"),
                            ("axum", "0.7.9"),
                        ]
                    ],
                ),
                (
                    "keep",
                    [
                        "    Finished `release` profile [optimized] target(s) in 41.37s",
                        "     Running `target/release/shopd --config shop.toml`",
                        "listening on port 8080 (workers=16, database shop on host db)",
                    ],
                ),
            ],
        ),
        (
            "terraform plan",
            [
                (
                    "drop",
                    [
                        "aws_s3_bucket.assets: Refreshing state... [id=shop-assets-prod]",
                        "aws_iam_role.app: Refreshing state... [id=shop-app-role]",
                        "aws_security_group.web: Refreshing state... [id=sg-0a1b2c3d]",
                    ],
                ),
                (
                    "keep",
                    [
                        "  ~ aws_instance.web will be updated in-place",
                        '  ~ resource "aws_instance" "web" {',
                        '      ~ instance_type = "t3.small" -> "t3.medium"',
                        '        id            = "i-0123456789abcdef0"',
                        "    }",
                    ],
                ),
                ("keep", ["Plan: 0 to add, 1 to change, 0 to destroy."]),
            ],
        ),
    ]
