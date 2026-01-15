# MKYZ Deployment Plan

This document outlines the steps to deploy MKYZ v0.2.0 to PyPI and update documentation on ReadTheDocs.

## 1. PyPI Deployment

### Prerequisites
- `twine` installed (`pip install twine`)
- `build` installed (`pip install build`)
- PyPI account credentials

### Steps

1. **Update Version**: Ensure `setup.py` and `mkyz/__init__.py` are set to `0.2.0`.
   - [x] `mkyz/__init__.py` updated
   - [ ] `setup.py` updated (Pending)

2. **Clean Build Artifacts**: Remove old build directories.
   ```powershell
   Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue
   ```

3. **Build Package**: Create source distribution and wheel.
   ```bash
   python -m build
   ```

4. **Check Package**: Verify the package integrity.
   ```bash
   twine check dist/*
   ```

5. **Upload to PyPI**:
   ```bash
   twine upload dist/*
   ```

---

## 2. ReadTheDocs (RTD) Deployment

We are switching to **MkDocs** because the new documentation is written in Markdown, which MkDocs handles natively and beautifully (using the Material theme).

### Configuration Changes

1. **New Configuration File**: Create `mkdocs.yml` in root.
   - Defines site name, theme (Material), and navigation structure.
   - Points to `docs/` folder.

2. **Update RTD Config**: Update `.readthedocs.yaml`.
   - Switch tool from `sphinx` to `mkdocs`.
   - Add `mkdocs-material` to requirements.

3. **Dependencies**: Update `requirements.txt` (or add a separate `docs/requirements.txt`).
   - Add `mkdocs` and `mkdocs-material`.

### Steps

1. Push the changes to GitHub (including `mkdocs.yml` and `.readthedocs.yaml`).
2. ReadTheDocs should automatically trigger a build.
3. If not, go to RTD dashboard and click "Build version".

---

## 3. Post-Deployment Verification

- **PyPI**: `pip install mkyz==0.2.0` in a fresh environment.
- **ReadTheDocs**: Visit https://mkyz.readthedocs.io/en/latest/ and check if new pages appear.
