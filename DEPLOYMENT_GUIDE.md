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
1.  Run `fetch_data.py` locally on your computer to get fresh 2025-26 data.
2.  This updates your local `nba_data.pkl`.
3.  Upload the NEW `nba_data.pkl` to your GitHub repository (replace the old one).
4.  Streamlit Cloud will detect the change and restart the app automatically with fresh data!
