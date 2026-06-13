import os
import subprocess
import sys
from database.db_manager import init_db

def main():
    print("Initializing FinSentiment database...")
    init_db()
    
    # Launch Streamlit App
    app_path = os.path.join(os.path.dirname(__file__), "app", "dashboard.py")
    print(f"Launching Streamlit application from: {app_path}")
    
    # Ensure root workspace is in PYTHONPATH so submodules can be imported
    env = os.environ.copy()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{root_dir}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = root_dir
        
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], env=env, check=True)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error starting Streamlit app: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
