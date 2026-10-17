# Deploy the static frontend (Netlify, free)

This folder is a **static site** — no Python server needed. The trained Ridge
regression model was exported to `model.json` as plain numbers, and `index.html`
does the prediction math in JavaScript.

## Option A — drag & drop (fastest, no GitHub needed)
1. Go to **app.netlify.com/drop** (log in with GitHub/Email).
2. Drag the **`netlify/`** folder onto the page.
3. Done — you get a live URL like `https://california-house-predictor.netlify.app`.

## Option B — via GitHub (auto-redeploy on push)
1. Push the **`netlify/`** folder to a GitHub repo.
2. In Netlify: **Add new site → Import an existing project** → pick the repo.
3. Build settings: leave empty (no build step). Publish directory: `netlify`
   (or the repo root if you pushed only this folder's contents).
4. Deploy.

## Files
- `index.html` — the app (sliders + live prediction + model info)
- `model.json` — exported model: feature names/labels, scaler means & scales,
  Ridge coefficients, intercept, metrics, slider ranges/defaults
- `netlify.toml` — tells Netlify this folder is the site root

## To retrain with different settings
Re-run the export step (see `build_notebook.py` / the training snippet in the
ML teaching guide), overwrite `model.json`, redeploy. The page reads everything
— labels, ranges, coefficients — from `model.json`, so no HTML edits are needed.
