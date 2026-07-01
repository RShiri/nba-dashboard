# 🚀 Deployment Guide: Streamlit Community Cloud

Follow these steps to get your **Deni Avdija Analytics Dashboard** online!

## Step 1: Prepare Your GitHub Repository
1.  Log in to your **GitHub** account.
2.  Click the **+** icon (top right) -> **New repository**.
3.  Name it (e.g., `deni-analytics-dashboard`).
4.  Select **Public**.
5.  Click **Create repository**.

## Step 2: Upload Your Project Files
You need to upload the following files to your new repository:
*   `app.py` (Your main application code)
*   `fetch_data.py` (The data fetcher)
*   `nba_data.pkl` (Your data file - **Crucial!**)
*   `requirements.txt` (List of libraries)
*   `.streamlit/config.toml` (Dark theme — **needed for the correct look on Streamlit Cloud**)
*   `profile_pic.png` (About Me photo)
*   `.gitignore` (Optional config)

**How to upload via Browser:**
1.  In your new repository, click the link **"uploading an existing file"**.
2.  Drag and drop the files listed above from your computer folder.
3.  Add a commit message (e.g., "Initial commit") and click **Commit changes**.

## Step 3: Deploy on Streamlit Cloud
1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Log in with your **GitHub** account.
3.  Click **New app** (top right).
4.  **Repository**: Select `your-username/deni-analytics-dashboard`.
5.  **Branch**: Usually `main` or `master`.
6.  **Main file path**: `app.py`.
7.  Click **Deploy!** 🎈

## ⏳ What happens next?
*   Streamlit will spin up a server.
*   It will read `requirements.txt` and install the libraries.
*   It will run `app.py`.
*   Since `nba_data.pkl` is in the repo, the app will load instanly without needing to fetch new data!

## 💡 Updating Data in the Future
To update the stats on the live site:
1.  Run the fetcher locally to get fresh data for the **current season** (auto-detected — no season string to edit):
    ```bash
    python fetch_data.py      # Windows: py fetch_data.py
    ```
2.  This updates your local `nba_data.pkl` (and, if Git is configured, `auto_update.py` commits & pushes it for you).
3.  Otherwise, commit/upload the NEW `nba_data.pkl` to GitHub (replace the old one).
4.  Streamlit Cloud detects the push and **auto-redeploys** in ~1–3 minutes with fresh data — the URL stays the same.

> 🗓️ **Season rollover is automatic.** When the 2026-27 season (or any later one) begins, the same command fetches the new season's game logs, shot charts, All-Star race, and league tables — and the dashboard labels roll forward on their own. No code changes required.

> 🔐 **Push auth note:** on Windows the first `git push` may open a **GitHub sign-in window** (Git Credential Manager). Complete it once and the credentials are cached for future pushes.
