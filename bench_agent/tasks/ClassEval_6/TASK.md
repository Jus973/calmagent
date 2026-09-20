# Task

Implement the class in `solution.py` so that the test suite passes.

Run the tests with:

    python -m pytest -q

Rules:

- Edit `solution.py` only. Do not edit anything under `tests/`.
- Keep the class name, the constructor, and every method signature exactly as given.
- Use the standard library only.
- You are done when `python -m pytest -q` reports no failures.

## Environment

- Every command runs in a fresh non-interactive subshell with no terminal attached.
- Never run an interactive tool. `nano`, `vi`, `vim`, `emacs`, `less` and `more` will hang until
  they are killed and you will lose the turn. Write files with `cat <<'EOF' > solution.py`, and
  read them with `cat`, `nl -ba` or `sed -n`.
- Rewriting the whole of `solution.py` in one `cat <<'EOF'` heredoc is usually faster and more
  reliable than patching it with `sed`.
