#!/usr/bin/env python3
"""vasp-symmetry 启动脚本"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vasp_symmetry.cli import main
if __name__ == "__main__":
    sys.exit(main())
