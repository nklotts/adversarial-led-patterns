# Project page — `docs/`

The NeurIPS 2026 project page for *Dynamic Physical Adversarial LED Patterns via Reinforcement
Learning*. It is a plain static site (no build step, no Jekyll, no npm) that reuses the visual
system of the conference deck: cream `#F5F4EF` paper, deep `#12161D` hero, `#7030A0` accent,
Space Grotesk + Inter.

```
docs/
├── index.html                  the whole page
├── .nojekyll                   tells GitHub Pages to serve the files as-is
└── assets/
    ├── css/style.css
    ├── js/main.js              mobile nav, scroll spy, reveal, BibTeX copy
    ├── img/                    figures exported from the deck + favicon
    ├── zyryanov-lotts-neurips2026-dynamic-led-adversarial.pdf
    └── zyryanov-lotts-neurips2026-slides.pptx
```

All links in the page point at `github.com/nklotts/adversarial-led-patterns`, and the canonical
and social-card URLs at `nklotts.github.io/adversarial-led-patterns`.

---

## 1. Deploying to GitHub Pages

The site lives in `docs/` on a branch, and GitHub Pages serves that folder directly. There is
nothing to build and no GitHub Action to configure.

### Step 1 — commit the site

```bash
git add docs WEBSITE.md
```

```bash
git commit -m "Add NeurIPS 2026 project page"
```

### Step 2 — put `docs/` on `main`

`main` is the default branch, so that is the natural place to serve from. This copies the site
across without merging unrelated work in progress:

```bash
git push origin lotts-dev
```

```bash
git checkout main && git pull && git checkout lotts-dev -- docs WEBSITE.md && git commit -m "Add NeurIPS 2026 project page" && git push origin main && git checkout lotts-dev
```

> Serving straight from `lotts-dev` works too — pick that branch instead in step 3. The catch is
> that the site disappears if the branch is deleted after merging.

### Step 3 — turn Pages on (once)

1. Open `https://github.com/nklotts/adversarial-led-patterns/settings/pages`
2. **Build and deployment → Source**: `Deploy from a branch`
3. **Branch**: `main`  ·  **Folder**: `/docs`
4. **Save**

The first build takes one to two minutes. Progress is visible under the repository's **Actions**
tab, as a `pages build and deployment` run.

### Step 4 — open it

**<https://nklotts.github.io/adversarial-led-patterns/>**

Every later push to `main` republishes the site automatically.

### Requirements and gotchas

- **The repository must be public** for Pages on a free account. Private repositories need GitHub
  Pro / Team / Enterprise. Settings → General → Danger Zone → Change visibility.
- `docs/.nojekyll` must stay. Without it GitHub runs the files through Jekyll, which can drop
  files and folders whose names begin with `_`.
- The `assets/` paths in `index.html` are all relative, so the site works at
  `/adversarial-led-patterns/` as well as at a custom domain — do not change them to start
  with `/`.
- **If the repository is ever renamed, the Pages URL changes with it.** Update the
  `<link rel="canonical">` and the four `og:`/`twitter:` URLs at the top of `docs/index.html`.
- Pages is a static host. It cannot run Python, so nothing in `train.py` or `server.py` is
  involved.

---

## 2. Optional: a custom domain

1. Add a file `docs/CNAME` containing nothing but the hostname, e.g. `led-attack.example.com`
2. At your DNS provider create a `CNAME` record pointing that host at `nklotts.github.io`
   (for an apex domain such as `example.com`, create four `A` records to `185.199.108.153`,
   `185.199.109.153`, `185.199.110.153`, `185.199.111.153` instead)
3. Settings → Pages → Custom domain → enter the host → Save → tick **Enforce HTTPS** once the
   certificate is issued, which can take up to an hour
4. Update the canonical and social-card URLs in `docs/index.html`

---

## 3. Previewing locally

Relative paths need a real HTTP server — opening `index.html` over `file://` will not load the
CSS. From the repository root:

```bash
python -m http.server 4173 --directory docs
```

Then open <http://localhost:4173>.

---

## 4. Editing the page

Everything lives in three files.

| What you want to change | Where |
| --- | --- |
| Any text, number or table row | `docs/index.html` — sections are commented, e.g. `<!-- ===== MAIN RESULT ===== -->` |
| Colours, type, spacing | `docs/assets/css/style.css` — the `:root` block at the top holds every token |
| Nav behaviour, BibTeX copy | `docs/assets/js/main.js` |

A few specifics:

- **New figure** — drop the file in `docs/assets/img/`, then copy an existing `<figure class="fig">`
  block. Keep the `width`/`height` attributes accurate so the layout does not jump while loading,
  and write a real `alt` description.
- **Transferability heatmap** — each cell's tint comes from its inline `--t` custom property,
  where `t = (0.56 − value) / 0.16`, clamped to `0…1`. Recompute it if the numbers change.
- **Latent-action bars** — each `.bar` carries `style="--v:71"`, the bar width in percent.
- **Fade-in on scroll** — add `class="reveal"` to any element. It is disabled automatically for
  visitors with `prefers-reduced-motion`.
- **Camera-ready update** — replace the PDF in `docs/assets/` under the same filename; nothing
  else needs touching.

---

## 5. Where the content came from

| Source | Used for |
| --- | --- |
| The conference deck (`.pptx`) | Visual system, section structure, all figures |
| The NeurIPS camera-ready PDF | Abstract, Tables 1–3 and 8, limitations, broader impact |

Figures were exported from the deck's embedded media and re-encoded: photographs as progressive
JPEG, plots and the pipeline diagram as optimized PNG, about 1.5 MB of images in total.
