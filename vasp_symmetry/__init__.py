"""vasp_symmetry - VASP 结构对称性分析工具包

功能：
  - 读取 POSCAR 文件
  - 检测并分类所有对称操作（旋转轴、镜面、反演、滑移面、螺旋轴）
  - 矩阵表示所有对称操作
  - 分析指定 k 点的小群（little group）对称性
  - 对称操作对 (k, σ) 的作用分析
  - k·p 模型推导（对称性允许项展开）
  - 压电系数张量和自旋霍尔电导张量（点群约束）
  - 磁空间群 (MSG) 与交错磁 (Altermagnetism) 分析
"""

from .poscar_reader import read_poscar
from .symmetry import SymmetryAnalyzer
from .little_group import LittleGroupAnalyzer
from .symmetry_tensors import (
    KpModel, PiezoelectricTensor, SpinHallTensor,
    analyze_operation_action, report_symmetry_actions, full_tensor_report,
)
from .magnetic_symmetry import MagneticSymmetryAnalyzer
from .cli import main

__all__ = [
    "read_poscar", "SymmetryAnalyzer", "LittleGroupAnalyzer",
    "KpModel", "PiezoelectricTensor", "SpinHallTensor",
    "analyze_operation_action", "report_symmetry_actions", "full_tensor_report",
    "MagneticSymmetryAnalyzer",
    "main",
]
