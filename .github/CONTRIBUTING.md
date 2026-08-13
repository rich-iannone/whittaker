# Contributing to Whittaker

Thanks for your interest in improving Whittaker! Contributions of code, tests, and docs are all
welcome.

## Development setup

```bash
git clone https://github.com/rich-iannone/whittaker.git
cd whittaker
python -m venv .venv && source .venv/bin/activate
make install          # pip install -e ".[dev]"
pre-commit install    # optional but recommended
```

## The check gate

Before opening a pull request, make sure the full gate is green:

```bash
make check            # ruff format + ruff check + pyright + pytest
```

Individual targets: `make lint`, `make type-check`, `make test`, `make test-rparity`.
Run `make help` for the full list.

## House conventions

- **Narwhals-native.** Never assume Pandas; write data handling against Narwhals and drop
  to NumPy only inside numeric kernels.
- **Typed & deterministic.** Full type hints (`py.typed`), `pyright` clean, byte-identical
  output for identical inputs.
- **Prose style.** Docstrings use Quarto/Markdown, not RST: single backticks for inline
  code, numpydoc section headers (`Parameters`, `Returns`).
- Implementation lives in underscore-prefixed private modules; the public surface is
  curated explicitly in `__init__.py`.

## Pull requests

- Keep PRs focused and add tests for new behavior (target >=90% coverage).
- For new/changed statistics, add or update R-parity fixtures.
- Update docs (docstrings, `user_guide/`) alongside code.

By contributing you agree that your contributions are licensed under the MIT License.
