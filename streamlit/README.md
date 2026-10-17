# Deploy the Streamlit app (Streamlit Cloud, free, via GitHub)

## One-time setup
1. Create a GitHub repo (e.g. `california-housing-app`) and push the **`streamlit/`**
   folder contents (`app.py` + `requirements.txt`) — or push the whole project
   folder and point at the sub-path below.
2. Go to **share.streamlit.io** and sign in with GitHub.
3. Click **New app** → choose your repo, branch (`main`), and set
   **Main file path** to `streamlit/app.py` (or just `app.py` if you pushed only
   that folder).
4. Click **Deploy**. First launch trains the models (~30–60 seconds); later
   visits reuse the cache.

## Notes
- `requirements.txt` pins the Python packages Streamlit Cloud installs.
- The app downloads the California Housing dataset on first run (no data files needed).
- To update the app later, just `git push` — Streamlit Cloud redeploys automatically.
