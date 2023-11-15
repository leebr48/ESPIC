"""Functions to be used by Nox."""

import nox

nox.options.sessions = ["lint", "test"]


@nox.session
def lint(session: nox.Session) -> None:
    """Run the linter."""
    session.install("pre-commit")
    session.run(
        "pre-commit",
        "run",
        "--all-files",
        "--show-diff-on-failure",
        *session.posargs,
    )


@nox.session
def test(session: nox.Session) -> None:
    """Run the test suite."""
    session.install("-e.[test]")
    session.run("pytest", *session.posargs)


@nox.session
def build(session: nox.Session) -> None:
    """Build an SDist and wheel."""
    session.install("build")
    session.run("python", "-m", "build")
