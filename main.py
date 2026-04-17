import os
import sys

# Add subdirectories to sys.path if they are run from root
sys.path.append(os.path.dirname(os.path.abspath(__line__ if '__line__' in locals() else __file__)))

try:
    from data import generate_data
    from eda import analysis
    from model import train_model
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you are running main.py from the churn_platform directory.")
    sys.exit(1)

def main():
    """
    Orchestrates the entire Customer Churn Intelligence Platform pipeline.
    """
    # Step 1: Create outputs directory
    try:
        os.makedirs('outputs', exist_ok=True)
        print("✓ Outputs directory is ready.")
    except Exception as e:
        print(f"Failed to create outputs directory: {e}")
        sys.exit(1)

    # Step 2: Generate Data
    try:
        print("\n--- Running generate_data.py ---")
        generate_data.main()
    except Exception as e:
        print(f"Step failed: generate_data.py encoutered an error: {e}")
        sys.exit(1)

    # Step 3: Run Analysis
    try:
        print("\n--- Running analysis.py ---")
        analysis.main()
    except Exception as e:
        print(f"Step failed: analysis.py encoutered an error: {e}")
        sys.exit(1)

    # Step 4: Train Model
    try:
        print("\n--- Running train_model.py ---")
        train_model.main()
    except Exception as e:
        print(f"Step failed: train_model.py encoutered an error: {e}")
        sys.exit(1)

    # Final summary
    print("\n✓ All pipeline steps complete. Run `python dashboard/dashboard.py` to view the dashboard.")

if __name__ == "__main__":
    main()
