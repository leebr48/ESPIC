"""Functions to be used by Nox."""

import nox
import argparse

nox.options.sessions = ["lint", "type", "test"]


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
def type(session: nox.Session) -> None:
    """Run the type checker."""
    session.install("-e.[test]")
    session.install("mypy")
    session.run("mypy")


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

@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass "--serve" to quickly preview the docs. Pass "-b linkcheck" to check links.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Preview docs after building")
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    args, posargs = parser.parse_known_args(session.posargs)

    if args.builder != "html" and args.serve:
        session.error("Must not specify non-HTML builder with --serve")

    extra_installs = ["sphinx-autobuild"] if args.serve else []

    session.install("-e.[docs]", *extra_installs)
    session.chdir("docs")
    session.run(
        "sphinx-apidoc",
        "-o",
        "source/",
        "--module-first",
        "--no-toc",
        "--force",
        "../src/espic/",
    )

    if args.builder == "linkcheck":
        session.run(
            "sphinx-build", "-b", "linkcheck", "source", "build/linkcheck", *posargs
        )
        return

    shared_args = (
        "-T",  # full tracebacks
        f"-b={args.builder}",
        "source",
        f"build/{args.builder}",
        *posargs,
    )

    if args.serve:
        session.run("sphinx-autobuild", *shared_args)
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)
