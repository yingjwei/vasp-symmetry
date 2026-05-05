"""对称性检测与分类核心模块

使用 spglib 检测对称操作，并按类型分类：
  - 旋转轴（C2, C3, C4, C6）
  - 镜面（σ）
  - 反演中心（i）
  - 滑移面（a, b, c, n, d）
  - 螺旋轴（2₁, 3₁, 4₁, 6₁ 等）
"""

from __future__ import annotations

import numpy as np
import spglib
from dataclasses import dataclass, field
from typing import Literal

from .poscar_reader import Structure


# ============================================================
# 对称操作分类枚举
# ============================================================

class OpType:
    IDENTITY = "identity"       # 恒等操作 (E)
    INVERSION = "inversion"     # 反演 (i)
    ROTATION = "rotation"       # 旋转 (C_n)
    MIRROR = "mirror"           # 镜面 (σ)
    IMPROPER = "improper"       # 非真旋转 (S_n)
    SCREW = "screw"             # 螺旋轴
    GLIDE = "glide"             # 滑移面


# ============================================================
# 数据类型
# ============================================================

@dataclass
class SymmetryOp:
    """单个对称操作"""
    rotation: np.ndarray  # 3x3 旋转/变换矩阵
    translation: np.ndarray  # 3 元平移向量（分数坐标）
    op_type: str  # 操作类型
    symbol: str  # 符号表示，如 "C2", "σ", "i", "2₁"
    axis_info: str = ""  # 轴或面方向描述
    order: int = 0  # 阶数

    def matrix_str(self) -> str:
        """返回旋转矩阵的格式化字符串"""
        lines = []
        for row in self.rotation:
            lines.append("  " + " ".join(f"{v:6.0f}" for v in row))
        return "\n".join(lines)

    def __repr__(self) -> str:
        line = f"[{self.symbol:6s}] 类型={self.op_type:10s}  阶数={self.order}"
        if self.axis_info:
            line += f"  {self.axis_info}"
        t = self.translation
        if np.any(np.abs(t) > 1e-10):
            line += f"  平移=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})"
        return line


@dataclass
class SymmetryResult:
    """对称性分析结果"""
    spacegroup_number: int
    spacegroup_symbol: str  # 国际符号 (Hermann-Mauguin)
    pointgroup_symbol: str  # 点群符号 (Schoenflies)
    operations: list[SymmetryOp] = field(default_factory=list)
    lattice: np.ndarray | None = None
    positions: np.ndarray | None = None  # 分数坐标

    # 按类型分组
    rotations: dict[str, list[SymmetryOp]] = field(default_factory=dict)
    mirrors: list[SymmetryOp] = field(default_factory=list)
    glides: list[SymmetryOp] = field(default_factory=list)
    screws: list[SymmetryOp] = field(default_factory=list)
    inversion: SymmetryOp | None = None
    identity: SymmetryOp | None = None

    def summary(self) -> str:
        """生成分析摘要"""
        lines = [
            "=" * 60,
            f"  空间群 (国际号):   #{self.spacegroup_number}",
            f"  空间群 (HM 符号):   {self.spacegroup_symbol}",
            f"  点群:                {self.pointgroup_symbol}",
            f"  对称操作总数:        {len(self.operations)}",
            "=" * 60,
        ]

        if self.identity:
            lines.append(f"  恒等操作:  {self.identity.symbol}")
        if self.inversion:
            lines.append(f"  反演中心:  {self.inversion.symbol}")

        for axis_label, ops in sorted(self.rotations.items()):
            lines.append(f"  {axis_label} 旋转轴: {len(ops)} 个")
            for op in ops:
                lines.append(f"    {op}")

        if self.mirrors:
            lines.append(f"  镜面: {len(self.mirrors)} 个")
            for op in self.mirrors:
                lines.append(f"    {op}")

        if self.glides:
            lines.append(f"  滑移面: {len(self.glides)} 个")
            for op in self.glides:
                lines.append(f"    {op}")
                lines.append(f"      旋转矩阵:\n{op.matrix_str()}")

        if self.screws:
            lines.append(f"  螺旋轴: {len(self.screws)} 个")
            for op in self.screws:
                lines.append(f"    {op}")

        return "\n".join(lines)

    def full_report(self) -> str:
        """生成详细报告，包含所有操作的矩阵"""
        lines = [self.summary(), "", "─" * 60, "  所有对称操作的矩阵表示:", "─" * 60]

        for i, op in enumerate(self.operations):
            lines.append(f"\n  操作 #{i + 1}: {op.symbol}  ({op.op_type})")
            lines.append(f"  旋转矩阵 R:")
            for row in op.rotation:
                lines.append("    " + " ".join(f"{v:6.0f}" for v in row))
            t = op.translation
            if np.any(np.abs(t) > 1e-10):
                lines.append(f"  平移向量 τ = ({t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f})")
            if op.axis_info:
                lines.append(f"  方向描述: {op.axis_info}")

        return "\n".join(lines)


# ============================================================
# 对称性分析器
# ============================================================

class SymmetryAnalyzer:
    """对称性分析器：检测晶体的所有对称操作并分类"""

    # 允许的数值误差
    _TOL = 1e-8

    def __init__(self, structure: Structure, symprec: float = 1e-5):
        """
        参数:
            structure: Structure 对象
            symprec: 对称性检测精度
        """
        self.structure = structure
        self.symprec = symprec
        self._cell = structure.to_spglib_cell()
        self._analyzed = False
        self._result: SymmetryResult | None = None

    def analyze(self) -> SymmetryResult:
        """执行对称性分析"""
        if self._analyzed and self._result:
            return self._result

        lat, pos, numbers = self._cell

        # 获取空间群信息（兼容 spglib 2.6 dict 和 2.7+ attribute 接口）
        dataset = spglib.get_symmetry_dataset(self._cell, symprec=self.symprec)
        if dataset is None:
            raise RuntimeError("spglib 对称性分析失败")

        if isinstance(dataset, dict):
            spacegroup_number = dataset["number"]
            spacegroup_symbol = dataset["international"]
            pointgroup_symbol = dataset["pointgroup"]
        else:
            spacegroup_number = dataset.number
            spacegroup_symbol = dataset.international
            pointgroup_symbol = dataset.pointgroup

        # 获取所有对称操作
        sym = spglib.get_symmetry(self._cell, symprec=self.symprec)
        if sym is None:
            raise RuntimeError("spglib 无法获取对称操作")

        rotations = sym["rotations"]
        translations = sym["translations"]

        result = SymmetryResult(
            spacegroup_number=spacegroup_number,
            spacegroup_symbol=spacegroup_symbol,
            pointgroup_symbol=pointgroup_symbol,
            lattice=lat,
            positions=pos,
        )

        for R, t in zip(rotations, translations):
            op = self._classify_op(R, t)
            result.operations.append(op)

            # 按类型分组
            if op.op_type == OpType.IDENTITY:
                result.identity = op
            elif op.op_type == OpType.INVERSION:
                result.inversion = op
            elif op.op_type == OpType.ROTATION:
                key = f"C{op.order}"
                if key not in result.rotations:
                    result.rotations[key] = []
                result.rotations[key].append(op)
            elif op.op_type == OpType.MIRROR:
                result.mirrors.append(op)
            elif op.op_type == OpType.GLIDE:
                result.glides.append(op)
            elif op.op_type == OpType.SCREW:
                result.screws.append(op)

        self._result = result
        self._analyzed = True
        return result

    def _classify_op(self, R: np.ndarray, t: np.ndarray) -> SymmetryOp:
        """将 spglib 的 (R, t) 分类为具体对称操作类型"""
        R = np.array(R, dtype=float)
        t = np.array(t, dtype=float)
        # 将 t 规约到 [0, 1)
        t = t - np.floor(t + self._TOL)

        det = np.round(np.linalg.det(R))
        trace = np.round(np.trace(R))
        is_identity = np.allclose(R, np.eye(3), atol=self._TOL)
        is_inversion = np.allclose(R, -np.eye(3), atol=self._TOL)

        # 使用稍宽松的容差判断纯旋转（spglib 可能返回 ~1e-8 的数值噪声）
        is_pure = np.all(np.abs(t) < 1e-7)

        # --- 恒等操作 ---
        if is_identity:
            return SymmetryOp(
                rotation=R, translation=t,
                op_type=OpType.IDENTITY, symbol="E", order=1
            )

        # --- 反演 ---
        if is_inversion:
            return SymmetryOp(
                rotation=R, translation=t,
                op_type=OpType.INVERSION, symbol="i", order=2
            )

        # --- 确定变换类型 ---
        if det == 1:  # 真旋转
            if is_pure:
                order, axis, angle = self._rotation_info(R)
                symbol = f"C{order}"
                axis_str = self._axis_to_str(axis)
                return SymmetryOp(
                    rotation=R, translation=t,
                    op_type=OpType.ROTATION, symbol=symbol,
                    order=order, axis_info=axis_str
                )
            else:
                # 螺旋轴：旋转 + 平移
                order, axis, angle = self._rotation_info(R)
                # 计算沿轴方向的平移分量
                t_parallel = self._parallel_translation(t, axis)
                screw_order = self._screw_order(t_parallel, order)
                symbol = f"{order}{{{screw_order}}}"
                if screw_order > 0 and screw_order < order:
                    sym_str = f"{order}_{screw_order}"
                else:
                    sym_str = f"C{order}"
                axis_str = self._axis_to_str(axis)
                return SymmetryOp(
                    rotation=R, translation=t,
                    op_type=OpType.SCREW, symbol=sym_str,
                    order=order,
                    axis_info=f"螺旋轴 沿 {axis_str}, 滑移分量={t_parallel:.4f}"
                )

        else:  # det == -1: 包含反射/非真旋转
            # 先检查镜面/滑移面（特征值：1, 1, -1）
            if self._is_mirror(R):
                normal = self._mirror_normal(R)
                normal_str = self._axis_to_str(normal)
                # 计算面内平移分量（平行于镜面）
                n_unit = normal / (np.linalg.norm(normal) + 1e-15)
                t_parallel = t - np.dot(t, n_unit) * n_unit  # 面内分量
                t_parallel_norm = np.linalg.norm(t_parallel)

                if t_parallel_norm < self._TOL:
                    # 面内平移 ≈ 0 → 纯镜面（可能有原点偏移，但偏移垂直于镜面方向）
                    return SymmetryOp(
                        rotation=R, translation=t,
                        op_type=OpType.MIRROR, symbol="σ",
                        order=2, axis_info=f"法向: {normal_str}"
                    )
                else:
                    # 有面内平移 → 滑移面
                    glide_type = self._glide_type(t_parallel, normal)
                    return SymmetryOp(
                        rotation=R, translation=t,
                        op_type=OpType.GLIDE, symbol=glide_type,
                        order=2,
                        axis_info=f"滑移面 法向: {normal_str}, 滑移方向: {t_parallel}"
                    )

            # 非真旋转 (improper rotation)
            order = self._improper_order(R)
            symbol = f"S{order}"
            return SymmetryOp(
                rotation=R, translation=t,
                op_type=OpType.IMPROPER, symbol=symbol,
                order=order
            )

    # ---- 辅助方法 ----

    @staticmethod
    def _rotation_info(R: np.ndarray) -> tuple[int, np.ndarray, float]:
        """从旋转矩阵提取阶数、轴方向和旋转角度（度）"""
        # 特征值1对应的特征向量即为旋转轴
        eigvals, eigvecs = np.linalg.eig(R)
        idx = np.argmin(np.abs(eigvals - 1.0))
        axis = np.real(eigvecs[:, idx]).flatten()
        axis = axis / (np.linalg.norm(axis) + 1e-15)

        trace = np.trace(R)
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.degrees(np.arccos(cos_theta))

        # 根据角度确定阶数
        angle_map = {360: 1, 180: 2, 120: 3, 90: 4, 60: 6}
        order = 1
        min_diff = 999
        for ang, ord_ in angle_map.items():
            diff = abs(theta - ang)
            if diff < min_diff:
                min_diff = diff
                order = ord_

        return order, axis, theta

    @staticmethod
    def _is_mirror(R: np.ndarray) -> bool:
        """判断是否为纯镜面操作（特征值 1, 1, -1）"""
        eigvals = np.linalg.eigvals(R)
        rounded = np.sort(np.round(np.real(eigvals)))
        return np.allclose(rounded, [-1, 1, 1], atol=1e-6)

    @staticmethod
    def _mirror_normal(R: np.ndarray) -> np.ndarray:
        """计算镜面的法向量（特征值 -1 对应的特征向量）"""
        eigvals, eigvecs = np.linalg.eig(R)
        idx = np.argmin(np.abs(eigvals + 1.0))
        normal = np.real(eigvecs[:, idx]).flatten()
        return normal / (np.linalg.norm(normal) + 1e-15)

    @staticmethod
    def _axis_to_str(axis: np.ndarray) -> str:
        """将轴方向向量转换为可读字符串"""
        # 找到最大的分量并取整
        a = np.array(axis)
        # 尝试化简为整数比
        max_idx = np.argmax(np.abs(a))
        if abs(a[max_idx]) < 1e-6:
            return "(0,0,0)"
        scaled = a / abs(a[max_idx])
        # 四舍五入到最近的 0 或 ±1
        rounded = np.round(scaled * 2) / 2
        rounded = np.where(np.abs(rounded) < 1e-6, 0, rounded)
        return f"[{rounded[0]:.0f} {rounded[1]:.0f} {rounded[2]:.0f}]"

    @staticmethod
    def _parallel_translation(t: np.ndarray, axis: np.ndarray) -> float:
        """计算平移沿轴方向的投影分量（的模）"""
        axis_u = axis / (np.linalg.norm(axis) + 1e-15)
        return float(np.dot(t, axis_u))

    @staticmethod
    def _screw_order(t_parallel: float, rotation_order: int) -> int:
        """确定螺旋轴阶数（如 2₁ 中的 1）"""
        t_parallel = t_parallel % 1.0
        if t_parallel < 1e-10:
            return 0
        # t_parallel ≈ n/m 其中 m = rotation_order
        best_n = 0
        min_diff = 999.0
        for n in range(1, rotation_order):
            expected = n / rotation_order
            diff = abs(t_parallel - expected)
            if diff < min_diff:
                min_diff = diff
                best_n = n
        return best_n

    @staticmethod
    def _glide_type(t_plane: np.ndarray, normal: np.ndarray) -> str:
        """判断滑移面类型 (a, b, c, n, d)"""
        t = t_plane - np.floor(t_plane + 1e-8)
        t_frac = np.round(t * 4) / 4  # 四舍五入到 0.25 的倍数

        n = normal / (np.linalg.norm(normal) + 1e-15)

        # 判断滑移方向
        abs_comp = np.abs(t_frac)

        # a轴滑移：沿 [100] 方向平移 a/2
        # b轴滑移：沿 [010] 方向平移 b/2
        # c轴滑移：沿 [001] 方向平移 c/2
        # n滑移：(a+b)/2, (a+c)/2, (b+c)/2 等对角滑移
        # d滑移：(a±b)/4 等金刚石滑移

        if any(abs(c - 0.5) < 0.1 for c in abs_comp):
            # 半整数平移
            nonzero = np.where(abs_comp > 0.1)[0]
            if len(nonzero) == 1:
                # a/b/c 滑移
                return ["a", "b", "c"][nonzero[0]]
            else:
                return "n"  # 对角滑移
        elif any(abs(c - 0.25) < 0.1 for c in abs_comp):
            return "d"  # 金刚石滑移
        return "glide"

    @staticmethod
    def _improper_order(R: np.ndarray) -> int:
        """确定非真旋转的阶数"""
        # 非真旋转 R 满足 R^k = E，找到最小的 k
        current = np.eye(3, dtype=float)
        for k in range(1, 12):
            current = current @ R
            if np.allclose(current, np.eye(3), atol=1e-6):
                return k
        return 0
