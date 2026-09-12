import sys
from pathlib import Path

# Add project root and src directory to Python path for pytest session
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))