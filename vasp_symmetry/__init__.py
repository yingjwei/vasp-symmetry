"""vasp_symmetry - VASP 结构对称性分析工具包

功能：
  - 读取 POSCAR 文件
  - 检测并分类所有对称操作（旋转轴、镜面、反演、滑移面、螺旋轴）
  - 矩阵表示所有对称操作
  - 分析指定 k 点的小群（little group）对称性
"""

from .poscar_reader import read_poscar
from .symmetry import SymmetryAnalyzer
from .little_group import LittleGroupAnalyzer
from .cli import main

__all__ = ["read_poscar", "SymmetryAnalyzer", "LittleGroupAnalyzer", "main"]
