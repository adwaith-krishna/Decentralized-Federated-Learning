"""
Main training entrypoint located inside implementation/ folder.
Executes the full decentralized federated learning experiment.
"""

import os
import sys

# Ensure root directory is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import main

if __name__ == "__main__":
    main()
