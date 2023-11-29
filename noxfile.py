"""Functions to be used by Nox."""

import nox

nox.options.sessions = ["lint", "type", "test"]


@nox.session
def lint(session: nox.Session):
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
def type(session: nox.Session):
    """Run the type checker."""
    session.install("-e.[test]")
    session.install("mypy")
    session.run("mypy")


@nox.session
def test(session: nox.Session):
    """Run the test suite."""
    session.install("-e.[test]")
    session.run("pytest", *session.posargs)


@nox.session
def build(session: nox.Session):
    """Build an SDist and wheel."""
    session.install("build")
    session.run("python", "-m", "build")
