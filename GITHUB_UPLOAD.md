# Updating the existing repository

The existing remote is `git@github.com:mzz235711/CMDA.git`.

Use this repository's root, not the isolated `github_release/` export, when updating the existing repository. The export remains ignored and available locally. Original training code remains in its existing location so local experiments are not disrupted.

The root ignore rules exclude environments, generated results, models, caches, local manuscript material and optional third-party checkouts. Tracked generated files are removed from the index only; their local copies remain available. This does not erase their historical commits.

Review before publishing:

```bash
git status --short
git diff
git diff --cached --stat
```

Do not use `git add .` until reviewing optional scripts for machine-specific paths and database configuration. Add intended source files explicitly. The previously built `github_release/` contains portable copies but has not been substituted over the working experiment code.

```bash
git add README.md .gitignore GITHUB_UPLOAD.md
git add mscn/*.py BaoForPostgreSQL/bao_server/*.py
git diff --cached --check
git commit -m "Organize CMDA artifact and exclude generated files"
git push origin HEAD
```

Before adding new scripts, replace private dataset roots and database roles with command-line arguments or environment configuration. Keep upstream license notices. Confirm workload redistribution permissions and the intended project license before public release.
