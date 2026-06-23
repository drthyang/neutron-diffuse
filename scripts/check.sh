#!/usr/bin/env bash
# Run the same checks as GitHub CI (.github/workflows/ci.yml) before pushing:
#
#   pytest                                    -> tests
#   ruff  check src/ tests/                   -> lint
#   mypy  src/nebula3d --ignore-missing-imports  -> type check
#
# Usage:
#   bash scripts/check.sh
#   PY=/path/to/python bash scripts/check.sh      # choose the interpreter
#
# Install as an automatic pre-push guard (any clone):
#   ln -s ../../scripts/check.sh .git/hooks/pre-push
#
# Exits non-zero if any check fails. A check is skipped (not failed) when its
# tool is not installed, so a bare clone without dev extras still pushes.
set -uo pipefail

REPO="$(git rev-parse --show-toplevel 2>/dev/null)" \
    || REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
PY="${PY:-python3}"

fail=0
run() {  # run <label> <module> <args...>; skip if the tool is not importable
    local label="$1" mod="$2"; shift 2
    if ! "$PY" -c "import $mod" >/dev/null 2>&1; then
        echo "[check] $label: '$mod' not installed — skipped"
        return 0
    fi
    echo "[check] $label ..."
    "$PY" -m "$mod" "$@" || fail=1
}

# pytest: -o addopts= drops the pyproject --cov flags so it runs without
# pytest-cov (coverage has no fail-under, so this does not change pass/fail).
run pytest pytest -o addopts= -q
run ruff   ruff   check src/ tests/
run mypy   mypy   src/nebula3d --ignore-missing-imports

if [ "$fail" -ne 0 ]; then
    echo "" >&2
    echo "[check] FAILED — fix the issues above (bypass a push with: git push --no-verify)" >&2
    exit 1
fi
echo "[check] all checks passed."
exit 0
