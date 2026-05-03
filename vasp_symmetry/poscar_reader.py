"""POSCAR 文件解析模块"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Structure:
    """晶体结构数据"""

    scale: float = 1.0
    lattice: np.ndarray = field(default_factory=lambda: np.eye(3))
    elements: list[str] = field(default_factory=list)
    num_atoms: list[int] = field(default_factory=list)
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    selective_dynamics: np.ndarray | None = None
    comment: str = ""
    is_cartesian: bool = False

    @property
    def total_atoms(self) -> int:
        return sum(self.num_atoms)

    @property
    def frac_positions(self) -> np.ndarray:
        """返回分数坐标"""
        if self.is_cartesian:
            lat_inv = np.linalg.inv(self.lattice * self.scale)
            return self.positions @ lat_inv
        return self.positions

    def to_spglib_cell(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """转换为 spglib 需要的 (lattice, positions, numbers) 元组"""
        lat = self.lattice * self.scale
        pos = self.frac_positions

        numbers = []
        for elem, n in zip(self.elements, self.num_atoms):
            # 用原子序数代替元素符号
            atomic_numbers = {
                "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
                "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
                "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
                "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
                "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
                "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
                "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
                "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56,
                "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63,
                "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
                "Lu": 71,
                "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
                "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
                "Rn": 86,
                "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92, "Np": 93,
                "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
            }
            numbers.extend([atomic_numbers.get(elem, 0)] * n)
        return lat, pos, np.array(numbers, dtype=int)


def read_poscar(filename: str | Path) -> Structure:
    """读取 VASP POSCAR 文件

    参数:
        filename: POSCAR 文件路径

    返回:
        Structure 对象
    """
    lines = _read_clean_lines(filename)
    if len(lines) < 6:
        raise ValueError(f"POSCAR 文件格式错误: 行数不足 ({len(lines)} 行)")

    struct = Structure()
    idx = 0

    # 第1行: 注释
    struct.comment = lines[idx].strip()
    idx += 1

    # 第2行: 缩放因子
    try:
        scale_str = lines[idx].strip()
        struct.scale = float(scale_str.split()[0])
    except ValueError:
        raise ValueError(f"无法解析缩放因子: {lines[idx]}")
    idx += 1

    # 第3-5行: 晶格矢量
    lattice = []
    for i in range(3):
        try:
            parts = lines[idx + i].strip().split()
            lattice.append([float(x) for x in parts[:3]])
        except ValueError:
            raise ValueError(f"无法解析晶格矢量 {i+1}: {lines[idx + i]}")
    struct.lattice = np.array(lattice)

    if struct.scale < 0:
        # 负缩放因子表示体积
        vol = abs(struct.scale)
        current_vol = abs(np.linalg.det(struct.lattice))
        struct.scale = (vol / current_vol) ** (1.0 / 3.0)

    idx += 3

    # 第6行: 元素符号 (可能不存在)
    element_part = lines[idx].strip()

    # 判断是否为元素符号（字母开头）
    if element_part[0].isalpha():
        struct.elements = element_part.split()
        idx += 1
    else:
        # 没有元素符号，用虚拟名称
        struct.elements = []

    # 原子数目
    num_line = lines[idx].strip()
    try:
        struct.num_atoms = [int(x) for x in num_line.split()]
    except ValueError:
        raise ValueError(f"无法解析原子数目: {num_line}")
    idx += 1
    total_n = struct.total_atoms

    # 可选: Selective dynamics
    if idx < len(lines) and lines[idx].strip().upper().startswith("S"):
        # 以 S 开头就忽略整个单词
        has_selective = "SELECTIVE" in lines[idx].strip().upper() or "S" == lines[idx].strip().upper()[0]
        if has_selective:
            idx += 1
        else:
            has_selective = False
    else:
        has_selective = False

    # 坐标模式: Direct/Cartesian
    if idx < len(lines):
        coord_mode = lines[idx].strip().upper()
        if coord_mode.startswith("D"):
            struct.is_cartesian = False
        elif coord_mode.startswith("C") or coord_mode.startswith("K"):
            struct.is_cartesian = True
        else:
            raise ValueError(f"无法识别坐标模式: {lines[idx]}")
        idx += 1
    else:
        raise ValueError("缺少坐标模式标识 (Direct/Cartesian)")

    # 读取坐标
    positions = []
    sd_flags = [] if has_selective else None

    for i in range(total_n):
        if idx + i >= len(lines):
            raise ValueError(f"坐标行数不足: 需要 {total_n} 行，只有 {len(lines) - idx} 行")
        parts = lines[idx + i].strip().split()
        if len(parts) < 3:
            raise ValueError(f"坐标格式错误 (第 {idx + i + 1} 行): {lines[idx + i]}")
        try:
            positions.append([float(x) for x in parts[:3]])
        except ValueError:
            raise ValueError(f"无法解析坐标 (第 {idx + i + 1} 行): {lines[idx + i]}")

        if has_selective:
            flags = []
            for j in range(3):
                f = parts[3 + j].upper()
                if f in ("T", "TRUE", "F", "FALSE"):
                    flags.append(f[0] == "T")
                else:
                    # 跳过，可能没有 selective dynamics
                    flags = None
                    break
            if flags:
                sd_flags.append(flags)
            elif sd_flags is not None and len(sd_flags) > 0:
                # 前面有但这里没有，说明格式不一致
                sd_flags = None

    struct.positions = np.array(positions)
    if sd_flags:
        struct.selective_dynamics = np.array(sd_flags, dtype=bool)

    return struct


def _read_clean_lines(filename: str | Path) -> list[str]:
    """读取文件并去除空行和首尾空格"""
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {filename}")
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = []
        for line in f:
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    return lines
