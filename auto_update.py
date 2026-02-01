"""
Automatic Git Update Script
Runs after data updates to commit and push changes to GitHub.
This ensures the deployed dashboard on Streamlit Cloud stays current.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_git_command(cmd: list, cwd: str = ".") -> tuple[bool, str]:
    """Execute a git command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def auto_commit_and_push(message: str = None) -> bool:
    """
    Automatically commit and push changes to Git.
    Returns True if successful, False otherwise.
    """
    if message is None:
        message = f"Auto-update: Data refreshed on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    print("=" * 60)
    print("🔄 Starting Git Auto-Update")
    print("=" * 60)
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # 1. Check if we're in a git repository
    success, output = run_git_command(["git", "status"], cwd=script_dir)
    if not success:
        print("❌ Not a git repository or git not installed")
        return False
    
    # 2. Add the data file
    print("\n📦 Adding nba_data.pkl to staging...")
    success, output = run_git_command(["git", "add", "nba_data.pkl"], cwd=script_dir)
    if not success:
        print(f"❌ Failed to add file: {output}")
        return False
    print("✅ File staged")
    
    # 3. Check if there are changes to commit
    success, output = run_git_command(["git", "diff", "--cached", "--quiet"], cwd=script_dir)
    if success:  # No changes (exit code 0 means no diff)
        print("ℹ️  No changes to commit")
        return True
    
    # 4. Commit changes
    print(f"\n💾 Committing with message: '{message}'")
    success, output = run_git_command(
        ["git", "commit", "-m", message],
        cwd=script_dir
    )
    if not success:
        print(f"❌ Commit failed: {output}")
        return False
    print("✅ Changes committed")
    
    # 5. Push to remote
    print("\n🚀 Pushing to GitHub...")
    success, output = run_git_command(["git", "push"], cwd=script_dir)
    if not success:
        print(f"⚠️  Push failed: {output}")
        print("   This might be due to authentication or network issues.")
        print("   The commit was successful locally.")
        return False
    
    print("✅ Successfully pushed to GitHub!")
    print("\n" + "=" * 60)
    print("✅ Git Auto-Update Complete")
    print("=" * 60)
    return True


if __name__ == "__main__":
    # Can be called directly or with a custom message
    custom_message = sys.argv[1] if len(sys.argv) > 1 else None
    success = auto_commit_and_push(custom_message)
    sys.exit(0 if success else 1)
