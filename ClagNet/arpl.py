"""Run ARPL experiments (low or moderate-high openness)."""
import sys
from scripts.run_experiment import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--method", "arpl", *sys.argv[1:]]
    main()
