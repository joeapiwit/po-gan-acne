# Anonymization checklist (do this BEFORE citing this repo in a double-blind submission)

This copy is **identified** and is intended for internal/advisor review only.
Before the URL goes into a paper submitted to a double-blind venue (e.g. ACM HEALTH),
create a fresh repo under a NON-identifying account and apply every item below.

## Files that currently identify the authors

| File | Line | Content to change |
|------|------|-------------------|
| `README.md` | 6 | `Apiwit Puangsricharern and Suppawong Tuarob` |
| `README.md` | 7 | `Faculty of Information and Communication Technology, Mahidol University` |
| `README.md` | citation block | `@article{puangsricharern2026pogan, ... author={Puangsricharern, Apiwit and Tuarob, Suppawong}` |
| `LICENSE` | 3 | `Copyright (c) 2026 Apiwit Puangsricharern` |

Replace with: `Anonymous Authors (redacted for double-blind review)`.

## The account/URL itself

`github.com/joeapiwit/...` identifies the first author ("apiwit").
A new repo under a neutral account, or an anonymous host such as
`anonymous.4open.science`, is required. Renaming the repo alone is NOT enough.

## Also check before pushing

- [ ] `git log` author names and emails (use `git commit --author` or a fresh
      `git init` with a neutral `user.name` / `user.email`)
- [ ] No `.git/config` remote pointing at the identified account
- [ ] No dataset images committed (patient photographs must not be redistributed)
- [ ] No GAN weight `.pkl` files (host separately; see README)

## Safe to keep

- `/mnt/wd-ssd-4tb/...` default paths: generic server paths, no username or host
- `classifier/utils/metrics.py` header: upstream imbalanced-learn attribution
- `generator/` NVIDIA StyleGAN2-ADA license and attribution
