"""小群（Little Group / 波矢群）分析模块

给定 k 点，找到空间群中保持该 k 点不变（模倒格矢）的所有对称操作，
并确定对应的点群。
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .poscar_reader import Structure
from .symmetry import SymmetryAnalyzer, SymmetryOp, OpType


# ============================================================
# 点群名称映射
# ============================================================

# Schönflies 符号 → 中文名
POINTGROUP_CN = {
    "C1":  "C1 (无对称性)",
    "Ci":  "Ci (反演)",
    "Cs":  "Cs (镜面)",
    "C2":  "C2 (二次轴)",
    "C3":  "C3 (三次轴)",
    "C4":  "C4 (四次轴)",
    "C6":  "C6 (六次轴)",
    "D2":  "D2 (三个二次轴)",
    "D3":  "D3 (一个三次轴+三个二次轴)",
    "D4":  "D4 (一个四次轴+四个二次轴)",
    "D6":  "D6 (一个六次轴+六个二次轴)",
    "C2v": "C2v (二次轴+两个镜面)",
    "C3v": "C3v (三次轴+三个镜面)",
    "C4v": "C4v (四次轴+四个镜面)",
    "C6v": "C6v (六次轴+六个镜面)",
    "C2h": "C2h (二次轴+镜面+反演)",
    "C3h": "C3h (三次轴+水平镜面)",
    "C4h": "C4h (四次轴+镜面+反演)",
    "C6h": "C6h (六次轴+镜面+反演)",
    "D2h": "D2h",
    "D3h": "D3h",
    "D4h": "D4h",
    "D6h": "D6h",
    "D2d": "D2d",
    "D3d": "D3d",
    "S4":  "S4 (四次非真旋转)",
    "Td":  "Td (四面体群)",
    "Oh":  "Oh (八面体群)",
    "O":   "O (八面体旋转群)",
    "Th":  "Th",
    "T":   "T (四面体旋转群)",
}


@dataclass
class LittleGroupResult:
    """小群分析结果"""
    kpoint: np.ndarray  # k 点分数坐标
    kpoint_label: str  # k 点标记 (如 Γ, X, M, K 等)
    operations: list[SymmetryOp] = field(default_factory=list)
    pointgroup: str = ""  # 小群的点群符号
    num_operations: int = 0

    def summary(self) -> str:
        lines = [
            f"  k 点:   {self.kpoint_label} ({', '.join(f'{x:.4f}' for x in self.kpoint)})",
            f"  小群点群: {self.pointgroup}",
            f"  对称操作数: {self.num_operations}",
        ]
        if self.operations:
            lines.append("  对称操作:")
            for op in self.operations:
                lines.append(f"    {op.symbol:6s}  {op.axis_info}")
        return "\n".join(lines)


# ============================================================
# 常见高对称 k 点
# ============================================================

# 常见晶格类型的高对称 k 点（分数坐标）
# 参考 Setyawan & Curtarolo, Comp. Mat. Sci. 49, 299 (2010)

HIGH_SYMM_KPOINTS = {
    "cubic": {  # 立方晶系
        "Γ":  [0.0, 0.0, 0.0],
        "X":  [0.5, 0.0, 0.5],
        "R":  [0.5, 0.5, 0.5],
        "M":  [0.5, 0.5, 0.0],
    },
    "fcc": {  # 面心立方
        "Γ":  [0.0, 0.0, 0.0],
        "X":  [0.5, 0.0, 0.5],
        "L":  [0.5, 0.5, 0.5],
        "W":  [0.5, 0.25, 0.75],
        "K":  [0.375, 0.375, 0.75],
        "U":  [0.625, 0.25, 0.625],
    },
    "bcc": {  # 体心立方
        "Γ":  [0.0, 0.0, 0.0],
        "H":  [0.5, -0.5, 0.5],
        "N":  [0.0, 0.0, 0.5],
        "P":  [0.25, 0.25, 0.25],
    },
    "hexagonal": {  # 六角
        "Γ":  [0.0, 0.0, 0.0],
        "A":  [0.0, 0.0, 0.5],
        "K":  [1/3, 1/3, 0.0],
        "H":  [1/3, 1/3, 0.5],
        "M":  [0.0, 0.5, 0.0],
        "L":  [0.0, 0.5, 0.5],
    },
    "tetragonal": {  # 四方
        "Γ":  [0.0, 0.0, 0.0],
        "X":  [0.5, 0.0, 0.0],
        "M":  [0.5, 0.5, 0.0],
        "Z":  [0.0, 0.0, 0.5],
        "R":  [0.5, 0.5, 0.5],
        "A":  [0.5, 0.0, 0.5],
    },
    "orthorhombic": {  # 正交
        "Γ":  [0.0, 0.0, 0.0],
        "X":  [0.5, 0.0, 0.0],
        "Y":  [0.0, 0.5, 0.0],
        "Z":  [0.0, 0.0, 0.5],
        "U":  [0.5, 0.5, 0.0],
        "T":  [0.5, 0.0, 0.5],
        "S":  [0.0, 0.5, 0.5],
        "R":  [0.5, 0.5, 0.5],
    },
    "monoclinic": {  # 单斜
        "Γ":  [0.0, 0.0, 0.0],
        "A":  [0.5, 0.0, 0.0],
        "C":  [0.0, 0.5, 0.0],
        "D":  [0.5, 0.5, 0.0],
        "Z":  [0.0, 0.0, 0.5],
        "E":  [0.0, 0.5, 0.5],
    },
    "triclinic": {  # 三斜
        "Γ":  [0.0, 0.0, 0.0],
    },
}


# ============================================================
# 小群分析器
# ============================================================

class LittleGroupAnalyzer:
    """小群分析器：找出保持 k 点不变的对称子群"""

    def __init__(self, analyzer: SymmetryAnalyzer):
        """
        参数:
            analyzer: 已完成分析的 SymmetryAnalyzer 对象
        """
        self.analyzer = analyzer
        result = analyzer._result
        if result is None:
            raise ValueError("请先调用 analyzer.analyze() 再进行小群分析")
        self._result = result

    def analyze_kpoint(
        self,
        kpoint: list[float] | np.ndarray,
        label: str = "",
        tol: float = 1e-6,
    ) -> LittleGroupResult:
        """分析指定 k 点的小群

        参数:
            kpoint: k 点的分数坐标 [kx, ky, kz]
            label: k 点标签（如 "Γ", "X", "M"）
            tol: 判断 k 点是否等价的容差

        返回:
            LittleGroupResult
        """
        k = np.array(kpoint, dtype=float)
        lat = self._result.lattice
        # 倒格矢
        if lat is not None:
            recip_lat = np.linalg.inv(lat).T * 2 * np.pi
        else:
            recip_lat = np.eye(3) * 2 * np.pi

        little_ops = []
        for op in self._result.operations:
            R = op.rotation
            # 将 R 作用在 k 上（分数坐标下）
            k_rotated = R @ k
            # 检查 k_rotated 与 k 是否相差一个倒格矢（整数）
            diff = k_rotated - k
            diff_frac = diff - np.round(diff)
            if np.all(np.abs(diff_frac) < tol):
                little_ops.append(op)

        # 确定小群的抽象点群类型
        pg = self._identify_little_pointgroup(little_ops)

        label_str = label if label else f"({', '.join(f'{x:.4f}' for x in k)})"

        return LittleGroupResult(
            kpoint=k,
            kpoint_label=label_str,
            operations=little_ops,
            pointgroup=pg,
            num_operations=len(little_ops),
        )

    def analyze_path(
        self,
        kpoints: list[tuple[str, list[float]]],
        npoints: int = 11,
        tol: float = 1e-6,
    ) -> list[LittleGroupResult]:
        """分析 k 路径上所有 k 点的小群

        参数:
            kpoints: 路径节点列表 [(label, [kx, ky, kz]), ...]
            npoints: 每段路径的插值点数
            tol: 容差

        返回:
            LittleGroupResult 列表
        """
        results = []
        for i in range(len(kpoints) - 1):
            label_a, ka = kpoints[i]
            label_b, kb = kpoints[i + 1]
            for j in range(npoints):
                frac = j / (npoints - 1) if i == len(kpoints) - 2 else j / (npoints - 1)
                # 只对最后一个节点插值到端点
                if j == npoints - 1:
                    break
                k = np.array(ka, dtype=float) + frac * (np.array(kb, dtype=float) - np.array(ka, dtype=float))
                # 给中间点一个简单的标签
                if j == 0:
                    label = label_a
                else:
                    label = f"{label_a}→{label_b}#{j}"
                results.append(self.analyze_kpoint(k, label=label, tol=tol))
            # 最后一小段的终点就是下一个节点
            if i == len(kpoints) - 2:
                results.append(self.analyze_kpoint(kb, label=label_b, tol=tol))

        return results

    def list_high_symmetry_kpoints(self) -> dict[str, list[float]]:
        """根据当前结构晶系，列出常见高对称 k 点"""
        spg_num = self._result.spacegroup_number
        crystal_system = self._crystal_system(spg_num)

        key_map = {
            "cubic": "cubic",
            "fcc": "fcc",
            "bcc": "bcc",
            "hexagonal": "hexagonal",
            "trigonal": "hexagonal",  # 三角用六角坐标
            "tetragonal": "tetragonal",
            "orthorhombic": "orthorhombic",
            "monoclinic": "monoclinic",
            "triclinic": "triclinic",
        }

        key = key_map.get(crystal_system, "triclinic")
        return HIGH_SYMM_KPOINTS.get(key, HIGH_SYMM_KPOINTS["triclinic"])

    # ---- 辅助方法 ----

    def _identify_little_pointgroup(self, ops: list[SymmetryOp]) -> str:
        """从小群的操作集合推断点群类型（基于旋转矩阵集合）"""
        if not ops:
            return "C1"

        # 收集所有唯一的旋转矩阵（取整后去重）
        seen = set()
        unique_rots = []
        for op in ops:
            key = tuple(op.rotation.astype(int).flatten())
            if key not in seen:
                seen.add(key)
                unique_rots.append(op.rotation)

        n_total = len(unique_rots)

        # 检查反演
        has_inv = any(np.allclose(R, -np.eye(3), atol=1e-6) for R in unique_rots)

        # 分别统计真旋转 (det=+1) 和非真旋转 (det=-1)
        proper_rots = [R for R in unique_rots if np.linalg.det(R) > 0]
        improper_rots = [R for R in unique_rots if np.linalg.det(R) < 0]
        n_proper = len(proper_rots)

        # 统计各阶旋转轴数目（从真旋转推导，identity不算）
        n_C2, n_C3, n_C4, n_C6 = self._count_rotation_axes(proper_rots)

        # 统计镜面数量（从非真旋转中找 trace=1 的反射）
        n_mirrors = sum(1 for R in improper_rots
                        if abs(np.trace(R) - 1) < 0.5)

        # 构建特征签名 → 查表
        return self._lookup_pointgroup(
            n_total, n_proper, n_C2, n_C3, n_C4, n_C6, has_inv, n_mirrors)

    @staticmethod
    def _count_rotation_axes(proper_rots: list[np.ndarray]) -> tuple[int, int, int, int]:
        """统计真旋转中 C2/C3/C4/C6 轴的数量（不重复轴方向）"""
        axes_set = {"C2": set(), "C3": set(), "C4": set(), "C6": set()}

        for R in proper_rots:
            tr = np.round(np.trace(R))
            if abs(tr - 3) < 0.5:  # identity
                continue

            # 确定阶数
            if abs(tr - (-1)) < 0.5:
                order = 2
            elif abs(tr - 0) < 0.5:
                order = 3
            elif abs(tr - 1) < 0.5:
                order = 4
            elif abs(tr - 2) < 0.5:
                order = 6
            else:
                continue

            # 找旋转轴（特征值=1的特征向量）
            eigvals, eigvecs = np.linalg.eig(R)
            idx = np.argmin(np.abs(eigvals - 1.0))
            axis = np.real(eigvecs[:, idx]).flatten()
            # 归一化并取正方向
            axis = axis / (np.linalg.norm(axis) + 1e-15)
            # 确保第一个非零分量为正（规范方向）
            for i in range(3):
                if abs(axis[i]) > 1e-6:
                    if axis[i] < 0:
                        axis = -axis
                    break

            # 四舍五入到小数点后4位作为 key
            key = tuple(np.round(axis, 4))
            axes_set[f"C{order}"].add(key)

        return (
            len(axes_set["C2"]),
            len(axes_set["C3"]),
            len(axes_set["C4"]),
            len(axes_set["C6"]),
        )

    @staticmethod
    def _lookup_pointgroup(
        n_total: int, n_proper: int,
        n_C2: int, n_C3: int, n_C4: int, n_C6: int,
        has_inv: bool,
        n_mirrors: int = 0,
    ) -> str:
        """根据特征签名查表确定点群

        签名: (n_total, n_proper, n_C2, n_C3, n_C4, n_C6, has_inv)
        """
        # --- 三斜晶系 ---
        if n_total == 1:
            return "C1"
        if n_total == 2:
            if has_inv:
                return "Ci"
            # C2 或 Cs 需要进一步区分
            if n_C2 == 1 and n_proper == 2:
                return "C2"
            return "Cs"

        # --- 单斜/正交晶系 ---
        if n_total == 4:
            if n_proper == 2:  # 含非真旋转
                if n_C2 == 1:
                    return "C2h"
                return "Cs"  # fallback
            # n_proper == 4
            if n_C2 == 3:
                if has_inv:
                    return "D2h"
                # D2 或 C2v
                return "C2v" if has_inv is False else "D2"  # 这里 has_inv 已经检查过
            if n_C2 == 1:
                if has_inv:
                    return "C2h"
                return "C2v"
            return "C2v"

        # --- 三角晶系 ---
        if n_total == 3:
            return "C3"
        if n_total == 6:
            if n_C3 == 1 and n_proper == 3:
                if n_mirrors == 3:
                    return "C3v"
                return "C3h" if not has_inv else "C3"
            if n_C3 == 1 and n_proper == 6:
                return "D3"
            if n_C3 == 1 and n_proper == 3 and has_inv:
                return "S6"  # = C3i
            if n_C3 == 1:
                return "C3v"
            return "D3d" if has_inv else "D3h"
        if n_total == 12 and n_C3 == 4:
            if n_proper == 12:
                return "T"
            if has_inv:
                return "Th"
            return "Td"

        # D3d (含反演) / D3h (无反演): n_total=12, n_C3=1, n_C2=3
        if n_total == 12 and n_C3 == 1 and n_C2 == 3 and n_proper == 6:
            return "D3d" if has_inv else "D3h"

        # D6 / C6v / C6h 在 n_total=12 但 n_C6=1 时已在六角部分处理

        # --- 四方晶系 ---
        if n_total == 4 and n_C4 == 1:
            return "S4" if n_proper == 2 else "C4"
        if n_total == 8:
            if n_C4 == 1:
                if n_proper == 8:
                    return "D4"
                if n_proper == 4:
                    if n_C2 == 2:
                        return "D2d"
                    if has_inv:
                        return "C4h"
                    return "C4v"
                if n_proper == 2:
                    return "S4"  # fallback
            if n_C4 == 0:
                # n_total=8, n_C4=0: D2h (n_C2=3, has_inv) or D2d/D2/C2h
                if n_C2 == 3:
                    return "D2h" if has_inv else "D2"
                if n_C2 == 2 and n_proper == 4:
                    return "C2v"
                if n_C2 == 1:
                    return "C2h" if has_inv else "C2v"
            return "?"

        # --- 六角晶系 ---
        if n_total == 6 and n_C6 == 1:
            return "C3h" if n_proper == 3 else "C6"
        if n_total == 12:
            if n_C6 == 1 and n_proper == 12:
                return "D6"
            if n_C6 == 1 and n_proper == 6:
                if has_inv:
                    return "C6h"
                return "C6v"
            if n_C6 == 1:
                return "D6h" if has_inv else "D3h"
            if n_C3 == 1:
                return "D3d" if has_inv else "D3h"
            if n_C3 == 4 and n_C2 == 3:
                return "T"
            return "?"

        # --- 立方晶系 ---
        if n_total == 24 and n_C3 == 4:
            if n_proper == 24:
                return "O"
            if has_inv:
                return "Th"
            return "Td"
        if n_total == 48:
            return "Oh"

        # --- 特殊情况 ---
        # D4h 在四方中 = 16
        if n_total == 16:
            return "D4h"
        if n_total == 24 and n_C6 == 1:
            return "D6h"

        return "?"

    @staticmethod
    def _crystal_system(spacegroup_number: int) -> str:
        """根据空间群号判断晶系"""
        if 1 <= spacegroup_number <= 2:
            return "triclinic"
        elif 3 <= spacegroup_number <= 15:
            return "monoclinic"
        elif 16 <= spacegroup_number <= 74:
            return "orthorhombic"
        elif 75 <= spacegroup_number <= 142:
            return "tetragonal"
        elif 143 <= spacegroup_number <= 167:
            return "trigonal"
        elif 168 <= spacegroup_number <= 194:
            return "hexagonal"
        else:
            return "cubic"
