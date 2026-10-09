# Releasing asleep

Releases are published to PyPI by `.github/workflows/release.yml` when a GitHub
Release is published. Prepare releases from a clean checkout of `main`.

1. Confirm that the intended version does not already exist on PyPI or as a Git
   tag.
2. Update `version` in `pyproject.toml`.
3. Activate the project Conda environment and run the release checks:

   ```bash
   source "$(conda info --base)/etc/profile.d/conda.sh"
   conda activate ./venv
   ruff check .
   mypy
   pytest
   check-manifest
   python -m build
   python -m twine check dist/*
   ```

4. Commit and push the release preparation to `main`, then wait for the Test
   workflow to pass.
5. Create a draft GitHub Release whose tag and title exactly match the package
   version. Review the target commit and release notes, including contributor
   attribution, before publishing it.
6. Publish the GitHub Release. This triggers the Release workflow, which builds
   from the release tag and uploads the wheel and source distribution to PyPI.
7. Confirm that the Release workflow passed and that PyPI serves both files with
   the expected version metadata.

PyPI files and Git tags are immutable release records. If publishing fails,
repair the workflow or credentials and rerun the failed job; do not reuse a
version that has already been uploaded.
