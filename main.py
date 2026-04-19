import os
import subprocess
import sys

def run_script(script_path):
    print(f"\n--- Running {script_path} ---")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"Error: {script_path} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    print("Starting Employee Retention Intelligence Pipeline...")
    
    # Run pipeline sequentially
    run_script(os.path.join("data", "generate_data.py"))
    run_script(os.path.join("eda", "analysis.py"))
    run_script(os.path.join("model", "train_model.py"))

    print("\n✓ All pipeline steps complete. Run `python dashboard/dashboard.py` to view the comprehensive B2B Employee Retention dashboard.")

if __name__ == "__main__":
    main()
