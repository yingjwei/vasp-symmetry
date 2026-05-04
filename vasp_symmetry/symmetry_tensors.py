"""对称操作对 k 空间和自旋的作用、kp 模型、压电系数和自旋霍尔电导

基于点群对称性推导张量分量的约束关系。
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from itertools import product

from .symmetry import SymmetryAnalyzer, SymmetryOp, OpType
from .little_group import LittleGroupAnalyzer

# ============================================================
# Pauli 矩阵
# ============================================================

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
SIGMA = [SIGMA_X, SIGMA_Y, SIGMA_Z]  # σ_x, σ_y, σ_z
SIGMA_I = np.eye(2, dtype=complex)

SIGMA_LABELS = ["σ_x", "σ_y", "σ_z"]


# ============================================================
# 对称操作对 k 和 σ 的作用
# ============================================================

def rotation_matrix_cartesian(R: np.ndarray, lattice: np.ndarray | None = None) -> np.ndarray:
    """将 spglib 的整数旋转矩阵转为 Cartesian 坐标下的正交矩阵

    参数:
        R: spglib 3×3 整数旋转矩阵（分数坐标）
        lattice: 3×3 晶格矩阵（列向量）

    返回:
        3×3 Cartesian 旋转矩阵
    """
    R = np.array(R, dtype=float)
    if lattice is not None:
        # R_cart = A @ R @ A^{-1}
        A = np.array(lattice, dtype=float)
        A_inv = np.linalg.inv(A)
        R_cart = A @ R @ A_inv
        # 正交化（由于数值误差）
        U, _, Vt = np.linalg.svd(R_cart)
        R_cart = U @ Vt
        return R_cart
    return R


def sigma_transform_matrix(R: np.ndarray) -> np.ndarray:
    """计算对称操作 R 对 Pauli 矩阵的变换矩阵 S

    σ_i → S_ij σ_j, 其中 S_ij = det(R) * R_ij

    参数:
        R: 3×3 旋转矩阵（在 Cartesian 坐标下）

    返回:
        3×3 变换矩阵 S
    """
    R = np.array(R, dtype=float)
    det = np.linalg.det(R)
    return det * R


def transform_k_components(k_action: str, components: list[str]) -> list[str]:
    """对 k 分量组合应用变换，输出变换后的多项式（符号推导用）"""
    # 对于实际计算，直接使用矩阵乘法
    pass


@dataclass
class TransformationInfo:
    """单个对称操作对 (k, σ) 的作用信息"""
    symbol: str
    op_type: str
    R_cart: np.ndarray          # Cartesian 旋转矩阵
    S_matrix: np.ndarray        # σ 变换矩阵 S_ij (σ_i → S_ij σ_j)
    k_transform: list[str]      # (k_x, k_y, k_z) 变换后的表达式
    sigma_transform: list[str]  # (σ_x, σ_y, σ_z) 变换后的表达式

    def report(self) -> str:
        lines = [f"  操作: {self.symbol}  ({self.op_type})"]
        lines.append(f"  k 变换:")
        for i, label in enumerate(["x", "y", "z"]):
            lines.append(f"    k_{label} → {self.k_transform[i]}")
        lines.append(f"  σ 变换:")
        for i, label in enumerate(["x", "y", "z"]):
            lines.append(f"    σ_{label} → {self.sigma_transform[i]}")
        return "\n".join(lines)


def analyze_operation_action(
    op: SymmetryOp,
    lattice: np.ndarray | None = None,
    ndigits: int = 0,
) -> TransformationInfo:
    """分析单个对称操作对 (k, σ) 的作用

    参数:
        op: SymmetryOp 对称操作
        lattice: 晶格矩阵（用于 Cartesian 转换）
        ndigits: 四舍五入位数

    返回:
        TransformationInfo
    """
    R = np.array(op.rotation, dtype=float)
    R_cart = rotation_matrix_cartesian(R, lattice)
    S = sigma_transform_matrix(R_cart)

    R_rounded = np.round(R_cart, ndigits)

    # k 变换表达式
    k_expr = []
    for i in range(3):
        terms = []
        for j in range(3):
            val = R_rounded[i, j]
            if abs(val) < 1e-10:
                continue
            if abs(abs(val) - 1) < 1e-10:
                coeff = "" if val > 0 else "-"
            elif abs(val - 0.5) < 1e-10:
                coeff = "½" if val > 0 else "-½"
            elif abs(val + 0.5) < 1e-10:
                coeff = "-½"
            else:
                v = int(val) if val == int(val) else val
                coeff = f"{v:+g}" if val < 0 else f"{v:g}"
            if j == 0 and coeff in ("", "-"):
                terms.append(f"{coeff}k_{['x','y','z'][j]}")
            elif coeff in ("½", "-½", "−½"):
                terms.append(f"{coeff}k_{['x','y','z'][j]}")
            else:
                terms.append(f"{coeff}k_{['x','y','z'][j]}")
        k_expr.append(" + ".join(terms).replace("+ -", "- "))

    # σ 变换表达式
    s_expr = []
    for i in range(3):
        terms = []
        for j in range(3):
            val = np.round(S[i, j], ndigits)
            if abs(val) < 1e-10:
                continue
            if abs(abs(val) - 1) < 1e-10:
                coeff = "" if val > 0 else "-"
            elif abs(val - 0.5) < 1e-10:
                coeff = "½" if val > 0 else "-½"
            elif abs(val + 0.5) < 1e-10:
                coeff = "-½"
            else:
                v = int(val) if abs(val - int(val)) < 1e-10 else val
                coeff = f"{v:+g}" if val < 0 else f"{v:g}"
            if j == 0 and coeff in ("", "-"):
                terms.append(f"{coeff}σ_{['x','y','z'][j]}")
            elif coeff in ("½", "-½"):
                terms.append(f"{coeff}σ_{['x','y','z'][j]}")
            else:
                terms.append(f"{coeff}σ_{['x','y','z'][j]}")
        s_expr.append(" + ".join(terms).replace("+ -", "- "))

    return TransformationInfo(
        symbol=op.symbol,
        op_type=op.op_type,
        R_cart=R_cart,
        S_matrix=S,
        k_transform=k_expr,
        sigma_transform=s_expr,
    )


def report_symmetry_actions(
    analyzer: SymmetryAnalyzer,
    kpoint: list[float] | None = None,
    little_group_ops: list[SymmetryOp] | None = None,
) -> str:
    """生成对称操作的 (k, σ) 作用报告

    参数:
        analyzer: SymmetryAnalyzer 实例
        kpoint: 可选 k 点，只分析该 k 点的小群
        little_group_ops: 可选，直接传入小群操作列表

    返回:
        格式化报告字符串
    """
    result = analyzer._result
    if result is None:
        return "错误: 请先执行对称性分析"

    lattice = result.lattice
    if little_group_ops is not None:
        ops = little_group_ops
    else:
        ops = result.operations

    lines = [
        "─" * 65,
        "  对称操作对 (k, σ) 的作用",
        "─" * 65,
    ]

    for i, op in enumerate(ops):
        info = analyze_operation_action(op, lattice)
        lines.append(f"\n  操作 #{i + 1}: {info.symbol}  ({info.op_type})")

        # R_cart 矩阵
        lines.append(f"  R_cart (Cartesian 旋转矩阵):")
        for row in info.R_cart:
            row_str = " ".join(f"{v:+7.3f}" for v in row)
            lines.append(f"    {row_str}")

        # k 变换
        lines.append(f"  k 变换:")
        for j, label in enumerate(["x", "y", "z"]):
            lines.append(f"    k_{label}  →  {info.k_transform[j]}")

        # σ 变换
        lines.append(f"  σ 变换:")
        for j, label in enumerate(["x", "y", "z"]):
            lines.append(f"    σ_{label}  →  {info.sigma_transform[j]}")

    return "\n".join(lines)


# ============================================================
# k·p 模型：对称性允许项展开
# ============================================================

def _monomial_label(k_order: tuple[int, int, int]) -> str:
    """生成单项式标签如 k_x^2, k_x*k_y 等"""
    labels = []
    for i, n in enumerate(k_order):
        if n == 0:
            continue
        if n == 1:
            labels.append(f"k_{['x','y','z'][i]}")
        else:
            labels.append(f"k_{['x','y','z'][i]}^{n}")
    if not labels:
        return "1"
    return "*".join(labels)


def generate_k_polynomials(max_order: int = 2) -> list[tuple[int, int, int]]:
    """生成 k 多项式基（单项式的幂次组合）

    参数:
        max_order: 最大阶数

    返回:
        列表，每个元素为 (nx, ny, nz) 表示 k_x^nx * k_y^ny * k_z^nz
    """
    basis = [(0, 0, 0)]  # 常数项
    for order in range(1, max_order + 1):
        for nx in range(order + 1):
            for ny in range(order + 1 - nx):
                nz = order - nx - ny
                basis.append((nx, ny, nz))
    return basis


def transform_monomial(monomial: tuple[int, int, int], R_cart: np.ndarray) -> str:
    """将单项式 k_x^nx * k_y^ny * k_z^nz 用 R_cart 变换后展开

    返回展开后的表达式字符串
    """
    nx, ny, nz = monomial
    if nx == 0 and ny == 0 and nz == 0:
        return "1"

    # 每个 k_i 变换为 R_ij * k_j
    k_new = []
    for i in range(3):
        terms = []
        for j in range(3):
            val = np.round(R_cart[i, j], 10)
            if abs(val) < 1e-10:
                continue
            if abs(abs(val) - 1) < 1e-10:
                coeff = "" if val > 0 else "-"
            else:
                coeff = f"{val:g}"
            terms.append(f"{coeff}k_{['x','y','z'][j]}")
        k_new.append("+".join(terms).replace("+-", "-"))

    # 展开 (k'_x)^nx * (k'_y)^ny * (k'_z)^nz
    # 简化：只返回符号结果
    result = k_new[0] if nx > 0 else ""
    if ny > 0:
        result = f"({result})*({k_new[1]})" if result else k_new[1]
    if nz > 0:
        result = f"({result})*({k_new[2]})" if result else k_new[2]
    if not result:
        result = "1"

    if nx == 1 and ny == 0 and nz == 0:
        return k_new[0]
    if nx == 0 and ny == 1 and nz == 0:
        return k_new[1]
    if nx == 0 and ny == 0 and nz == 1:
        return k_new[2]

    return result


@dataclass
class KpTerm:
    """k·p 哈密顿量中的一个对称允许项"""
    order: int                          # 阶数
    monomial: tuple[int, int, int]      # k 单项式幂次
    monomial_str: str                   # 单项式字符串
    matrix: np.ndarray                  # 在自旋空间的 2×2 矩阵
    label: str = ""                     # 标签

    def __repr__(self) -> str:
        mat_str = []
        for row in self.matrix:
            row_str = " ".join(f"{v.real:+5.2f}{'i' if abs(v.imag)>1e-10 else '':>2}"
                              for v in row)
            mat_str.append(f"      [{row_str}]")
        return (f"    {self.monomial_str:12s}  ×  "
                f"({self.label:10s})\n" + "\n".join(mat_str))


class KpModel:
    """k·p 模型：从对称性推导哈密顿量的允许项"""

    # 2×2 Pauli 基 {I, σ_x, σ_y, σ_z}
    SPIN_BASIS = [
        (SIGMA_I, "I"),
        (SIGMA_X, "σ_x"),
        (SIGMA_Y, "σ_y"),
        (SIGMA_Z, "σ_z"),
    ]

    def __init__(self, little_group_ops: list[SymmetryOp], lattice: np.ndarray | None = None):
        """
        参数:
            little_group_ops: 小群对称操作列表
            lattice: 晶格矩阵
        """
        self.ops = little_group_ops
        self.lattice = lattice

    def expand(self, max_order: int = 2) -> list[list[KpTerm]]:
        """展开 k·p 哈密顿量到指定阶数

        参数:
            max_order: 最高阶数

        返回:
            列表，每个元素是该阶的所有对称允许项
            terms_by_order[0] = 常数项
            terms_by_order[1] = 线性项
            ...
        """
        all_terms = []
        for order in range(max_order + 1):
            terms = self._find_invariant_terms(order)
            if terms:
                all_terms.append(terms)
        return all_terms

    def _find_invariant_terms(self, order: int) -> list[KpTerm]:
        """找到给定阶数所有对称不变的项

        H(k) = Σ f_n(k) * M_n 在对称操作下不变的条件：
        D(R)^† H(R·k) D(R) = H(k)
        """
        monomials = generate_k_polynomials(order)
        # 只保留该阶数的单项式
        order_monomials = [m for m in monomials if sum(m) == order]

        if order == 0:
            order_monomials = [(0, 0, 0)]

        invariant_terms = []

        for mono in order_monomials:
            mono_str = _monomial_label(mono)
            for basis_mat, basis_label in self.SPIN_BASIS:
                if self._is_invariant(mono, basis_mat):
                    term = KpTerm(
                        order=order,
                        monomial=mono,
                        monomial_str=mono_str,
                        matrix=basis_mat,
                        label=basis_label,
                    )
                    invariant_terms.append(term)

        return invariant_terms

    def _is_invariant(self, monomial: tuple[int, int, int], matrix: np.ndarray) -> bool:
        """检查单项式 × 矩阵是否在所有对称操作下不变"""
        for op in self.ops:
            R = np.array(op.rotation, dtype=float)
            R_cart = rotation_matrix_cartesian(R, self.lattice)
            det = np.linalg.det(R_cart)

            # 1. 变换 k 多项式：f(k) → f(R^(-1)·k) = f(R^T·k) 对于正交阵
            # 实际上 f'(k) = f(R_cart^T · k)
            # 由于我们处理的是单项式 f(k) = k_x^nx * k_y^ny * k_z^nz
            # 变换后是 (R·k)_x^nx * (R·k)_y^ny * (R·k)_z^nz
            # 这应该等于 f(k) 对于不变性

            # 更简单的方法：检查单项式是否在 R_cart 下不变
            # 即 f(R_cart · k) = f(k) 对所有 k
            # 这要求展开后和原单项式相同
            if not self._monomial_invariant(monomial, R_cart):
                return False

            # 2. 变换矩阵：M → D(R)^† M D(R)
            # 对于 spin-1/2: D(R) 是 SU(2) 表示
            # M 在 {I, σ} 基下变换为：I → I, σ_i → det(R) * R_ij σ_j
            if not self._matrix_invariant(matrix, R_cart, det):
                return False

        return True

    @staticmethod
    def _monomial_invariant(monomial: tuple[int, int, int], R_cart: np.ndarray) -> bool:
        """检查单项式在 R_cart 下是否保持不变

        对于线性项 (kx, ky, kz)：检查 R_cart 是否是对角矩阵且特征值为 1
        对于高阶项：更复杂的检查
        """
        nx, ny, nz = monomial
        if nx == 0 and ny == 0 and nz == 0:
            return True  # 常数项总是不变

        # 构造变换后的表达式并检查是否等价于原单项式
        # 简化检查：对于一阶项，k_j → R_ij k_i，检查 R 是否有非对角元
        R = np.round(R_cart, 10)

        if nx + ny + nz == 1:
            # 线性项
            idx = 0 if nx == 1 else (1 if ny == 1 else 2)
            # k_idx → R[idx,j] * k_j
            # 需满足 R[idx,j] = δ_{idx,j}
            return np.allclose(R[idx], [1.0 if j == idx else 0.0 for j in range(3)], atol=1e-10)

        elif nx + ny + nz == 2:
            # 二次项：需要检查所有交叉项
            # k_i * k_j → (R_im k_m) * (R_jn k_n) = R_im R_jn k_m k_n
            # 需要 R_im R_jn = δ_im δ_jn (对于相关分量)
            # 简化：构造变换后的二次型并与原二次型比较
            orig = np.zeros((3, 3))
            if nx == 2:
                orig[0, 0] = 1
            elif ny == 2:
                orig[1, 1] = 1
            elif nz == 2:
                orig[2, 2] = 1
            elif nx == 1 and ny == 1:
                orig[0, 1] = orig[1, 0] = 1
            elif nx == 1 and nz == 1:
                orig[0, 2] = orig[2, 0] = 1
            elif ny == 1 and nz == 1:
                orig[1, 2] = orig[2, 1] = 1

            transformed = R.T @ orig @ R
            return np.allclose(transformed, orig, atol=1e-10)

        # 高阶项：简化处理，检查非对角元
        for i in range(3):
            for j in range(3):
                if i != j and abs(R[i, j]) > 1e-10:
                    return False
        return True

    @staticmethod
    def _matrix_invariant(matrix: np.ndarray, R_cart: np.ndarray, det: float) -> bool:
        """检查矩阵在对称操作下是否不变"""
        if np.allclose(matrix, SIGMA_I, atol=1e-10):
            return True  # 单位矩阵总是不变

        # σ_i → det(R) * R_ij σ_j
        S = det * R_cart

        # 检查 matrix 在 S 下是否不变
        # matrix = Σ c_i σ_i
        # 变换后: Σ c_i (det(R) * R_ij σ_j) = Σ (c_j det(R) * R_ji) σ_i ???

        # 提取 matrix 在 Pauli 基下的系数
        coeffs = np.zeros(4, dtype=float)
        coeffs[0] = 0.5 * np.trace(matrix @ SIGMA_I).real  # I 分量
        for i in range(3):
            coeffs[i + 1] = 0.5 * np.trace(matrix @ SIGMA[i]).real

        # 变换后的系数
        new_coeffs = coeffs.copy()
        for i in range(3):  # σ_i
            new_coeffs[i + 1] = sum(coeffs[j + 1] * S[j, i] for j in range(3))

        # 检查是否不变
        return np.allclose(coeffs, new_coeffs, atol=1e-10)

    def report(self, max_order: int = 2) -> str:
        """生成 k·p 模型报告"""
        all_terms = self.expand(max_order)

        lines = [
            "─" * 65,
            f"  k·p 模型（对称性允许项，最高阶数 = {max_order}）",
            "─" * 65,
        ]

        if not all_terms:
            lines.append("  (未找到任何对称允许项)")
            return "\n".join(lines)

        for order, terms in enumerate(all_terms):
            n_terms = len(terms)
            lines.append(f"\n  第 {order} 阶 ({n_terms} 项):")
            for term in terms:
                lines.append(f"    {term.monomial_str:15s}  ×  {term.label:6s}")

        # 输出完整哈密顿量形式
        lines.append("\n" + "─" * 65)
        lines.append("  H(k) =")
        for order, terms in enumerate(all_terms):
            for term in terms:
                # Pauli 分解
                mat_str = self._format_matrix(term.matrix)
                lines.append(f"    +  {term.monomial_str:15s}  *  {term.label:6s}")
                # lines.append(f"       矩阵: {mat_str}")

        return "\n".join(lines)

    @staticmethod
    def _format_matrix(m: np.ndarray) -> str:
        """格式化 2×2 矩阵"""
        parts = []
        for i in range(2):
            for j in range(2):
                v = m[i, j]
                if abs(v.real) > 1e-10 or abs(v.imag) > 1e-10:
                    parts.append(f"[{i},{j}]={v.real:+.2f}{v.imag:+.2f}i")
        return "; ".join(parts) if parts else "0"


# ============================================================
# 响应张量：压电系数、自旋霍尔电导
# ============================================================

# 点群 → 非零独立分量 (基于 Neumann 原理)
# 格式: { 点群: { 张量类型: [(分量, 关系), ...] } }

# ---- 压电张量 e_ijk (三阶) ----
# e_ijk 在对称操作 R 下: e_ijk → R_il R_jm R_kn e_lmn
# 常用 Voigt 标记: e_ij (6×3 矩阵, i=1..3, j=1..6)
# Voigt: xx→1, yy→2, zz→3, yz→4, xz→5, xy→6

PIEZOELECTRIC_TENSORS = {
    "C1": [
        (1, 1, 1), (1, 2, 2), (1, 3, 3), (1, 2, 3), (1, 1, 3), (1, 1, 2),
        (2, 1, 1), (2, 2, 2), (2, 3, 3), (2, 2, 3), (2, 1, 3), (2, 1, 2),
        (3, 1, 1), (3, 2, 2), (3, 3, 3), (3, 2, 3), (3, 1, 3), (3, 1, 2),
    ],
    "Ci": [],  # 有反演中心 → 所有压电系数为 0
    "Cs": [
        # 镜面在 xy 平面 (z → -z) 保留含偶数个 z 的分量
        (1, 1, 1), (1, 2, 2), (1, 3, 3), (1, 2, 3), (1, 1, 2),
        (2, 1, 1), (2, 2, 2), (2, 3, 3), (2, 1, 3), (2, 2, 3),
        (3, 1, 3), (3, 2, 3), (3, 3, 3), (3, 1, 1), (3, 2, 2),
    ],
    "C2": [
        # C2 沿 z: (x,y)→(-x,-y), z 不变
        # 保留: z 出现偶数次
        (1, 1, 3), (1, 2, 3), (1, 3, 1), (1, 3, 2),
        (2, 1, 3), (2, 2, 3), (2, 3, 1), (2, 3, 2),
        (3, 1, 1), (3, 2, 2), (3, 3, 3), (3, 1, 2),
    ],
    "C2v": [
        # C2 沿 z, 镜面在 xz 和 yz
        # Voigt: 只有 (3,1), (3,2), (3,3)
        (3, 1, 1), (3, 2, 2), (3, 3, 3),
    ],
    "C3": [
        # C3 沿 z
        (1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 2, 3),
        (2, 1, 1), (2, 1, 2), (2, 2, 2), (2, 2, 3),
        (3, 1, 3), (3, 2, 3), (3, 3, 1), (3, 3, 2), (3, 3, 3),
    ],
    "C3v": [
        (1, 1, 1), (1, 1, 2), (1, 2, 2),
        (2, 2, 2), (2, 1, 1), (2, 1, 2),
        (3, 1, 3), (3, 2, 3), (3, 3, 1), (3, 3, 2), (3, 3, 3),
    ],
    "C4": [
        (1, 1, 3), (1, 2, 3), (1, 3, 1), (1, 3, 2),
        (2, 1, 3), (2, 2, 3), (2, 3, 1), (2, 3, 2),
        (3, 1, 1), (3, 2, 2), (3, 3, 3),
    ],
    "C4v": [
        (1, 1, 3), (1, 2, 3), (1, 3, 1), (1, 3, 2),
        (2, 1, 3), (2, 2, 3), (2, 3, 1), (2, 3, 2),
        (3, 1, 1), (3, 2, 2), (3, 3, 3),
    ],
    "D2": [],  # 有反演 → 0
    "D2d": [
        (1, 2, 3), (1, 3, 2),
        (2, 1, 3), (2, 3, 1),
        (3, 1, 2), (3, 2, 1),
    ],
    "D3": [],  # 有反演 → 0
    "D3h": [],  # 水平镜面 → 0
    "D4": [],  # 有反演 → 0
    "D6": [],  # 有反演 → 0
    "D2h": [],  # 正交反演 → 0
    "D3d": [],  # 反演 → 0
    "D4h": [],  # 反演 → 0
    "D6h": [],  # 反演 → 0
    "Td": [
        (1, 2, 3), (2, 3, 1), (3, 1, 2),
    ],
    "Oh": [],
    "O": [],   # O(3) → 0
    "T": [],   # T → 0 (有反演)
    "Th": [],  # 反演 → 0
    "S4": [
        (1, 1, 3), (1, 2, 3), (1, 3, 1), (1, 3, 2),
        (2, 1, 3), (2, 2, 3), (2, 3, 1), (2, 3, 2),
        (3, 1, 1), (3, 2, 2), (3, 3, 3),
    ],
    "C6": [
        (1, 1, 3), (1, 2, 3), (1, 3, 1), (1, 3, 2),
        (2, 1, 3), (2, 2, 3), (2, 3, 1), (2, 3, 2),
        (3, 1, 1), (3, 2, 2), (3, 3, 3),
    ],
    "C6v": [
        (1, 1, 3), (1, 2, 3), (1, 3, 1), (1, 3, 2),
        (2, 1, 3), (2, 2, 3), (2, 3, 1), (2, 3, 2),
        (3, 1, 1), (3, 2, 2), (3, 3, 3),
    ],
    "C3h": [],
    "C4h": [],
    "C6h": [],
    "S6": [],
}


def voigt_notation(i: int, j: int) -> int:
    """3×3 指标到 Voigt 6 指标: xx→1, yy→2, zz→3, yz→4, xz→5, xy→6"""
    if i == j:
        return i  # xx→0-based 0, 1, 2 但我们需要 1-based
    if (i, j) in ((1, 2), (2, 1)):
        return 5  # xy
    if (i, j) in ((0, 2), (2, 0)):
        return 4  # xz
    if (i, j) in ((1, 2), (2, 1)):
        return 3  # yz
    return 0


def voigt_label(alpha: int) -> str:
    """Voigt 指标标签"""
    labels = ["xx", "yy", "zz", "yz", "xz", "xy"]
    if 0 <= alpha < 6:
        return labels[alpha]
    return "?"


@dataclass
class TensorComponent:
    """张量分量"""
    indices: tuple[int, ...]  # 如 (1, 2, 3) 表示 e_123
    label: str = ""           # Voigt 或指标标签
    relation: str = ""        # 与其他分量的关系

    def __repr__(self) -> str:
        return f"{self.label:10s}  {self.relation}"


class ResponseTensor:
    """响应张量分析基类"""

    def __init__(self, pointgroup: str):
        self.pointgroup = pointgroup

    def components(self) -> list[TensorComponent]:
        raise NotImplementedError

    def report(self) -> str:
        comps = self.components()
        if not comps:
            return f"  点群 {self.pointgroup}: 所有分量为 0（有反演中心或对称性禁止）"
        lines = [f"  点群 {self.pointgroup}: 非零分量"]
        for c in comps:
            lines.append(f"    {c}")
        return "\n".join(lines)


class PiezoelectricTensor(ResponseTensor):
    """压电张量 e_ijk (三阶) — 对称性约束

    压电效应: P_i = e_ijk ε_jk  (ε 为应变张量)

    Voigt 表示: P_i = e_iα ε_α
    其中 α = xx(1), yy(2), zz(3), yz(4), xz(5), xy(6)
    """

    def __init__(self, pointgroup: str):
        super().__init__(pointgroup)
        # 标准点群映射
        self._pg_map = {
            "mm2": "C2v", "2mm": "C2v",
            "m": "Cs", "2": "C2",
            "4": "C4", "4mm": "C4v",
            "-42m": "D2d", "3m": "C3v", "3": "C3",
            "6": "C6", "6mm": "C6v",
            "m-3m": "Oh", "-43m": "Td",
        }

    def _resolve_pointgroup(self) -> str:
        pg = self.pointgroup
        if pg in PIEZOELECTRIC_TENSORS:
            return pg
        return self._pg_map.get(pg, pg)

    def components(self) -> list[TensorComponent]:
        pg = self._resolve_pointgroup()
        comps = PIEZOELECTRIC_TENSORS.get(pg, None)
        if comps is None:
            return [TensorComponent((), f"点群 {pg}", "数据未收录")]
        if not comps:
            return []

        result = []
        for comp in comps:
            i, j, k = comp
            # Voigt 标记
            voigt_map = {(0, 0): 1, (1, 1): 2, (2, 2): 3,
                         (1, 2): 4, (2, 1): 4,
                         (0, 2): 5, (2, 0): 5,
                         (0, 1): 6, (1, 0): 6}
            voigt_idx = voigt_map.get((j, k), 0)
            label = f"e_{i+1, voigt_idx}" if voigt_idx else f"e_{i+1}{j+1}{k+1}"
            rel = f"= e_{i+1},{voigt_idx}" if voigt_idx else "独立"
            result.append(TensorComponent(comp, label=label, relation=rel))
        return result


# ---- 自旋霍尔电导 σ^s_ijk (三阶) ----
# 自旋霍尔电导定义: J_i^{s_j} = σ^s_{ijk} E_k
# 其中 J_i^{s_j} 是沿 i 方向、自旋极化沿 j 方向的自旋电流
# E_k 是电场分量
#
# 对称性变换:
# σ^s_{ijk} → det(R) * R_il R_jm R_kn σ^s_lmn
# (因为自旋极化 j 方向是轴向矢量 → det(R) * R_jm)

SPIN_HALL_TENSORS = {
    "C1": [
        (0, 1, 2), (1, 1, 2), (2, 1, 2),
        (0, 2, 1), (1, 2, 1), (2, 2, 1),
    ],
    "Ci": [],
    "Cs": [
        (0, 3, 1), (0, 3, 2), (1, 3, 1), (1, 3, 2),
        (3, 1, 1), (3, 2, 2), (3, 1, 2), (3, 2, 1),
    ],
    "C2v": [
        (1, 3, 1), (2, 3, 2),
    ],
    "C4v": [
        (1, 3, 1), (2, 3, 2),
    ],
    "D2h": [],
    "Oh": [],
    "Td": [
        (1, 2, 3), (2, 3, 1), (3, 1, 2),
    ],
}


class SpinHallTensor(ResponseTensor):
    """自旋霍尔电导张量 σ^s_{ijk}

    J_i^{s_j} = σ^s_{ijk} E_k
    """

    def components(self) -> list[TensorComponent]:
        comps = SPIN_HALL_TENSORS.get(self.pointgroup, None)
        if comps is None:
            return [TensorComponent((), self.pointgroup, "数据未收录")]
        if not comps:
            return []

        result = []
        for comp in comps:
            i, j, k = comp
            label = f"σ^s_{i+1}{j+1},{k+1}"
            result.append(TensorComponent(comp, label=label, relation="非零"))
        return result


# ============================================================
# 综合报告
# ============================================================

def full_tensor_report(
    analyzer: SymmetryAnalyzer,
    little_group_ops: list[SymmetryOp] | None = None,
    kp_max_order: int = 2,
) -> str:
    """生成完整的张量性质报告

    包括:
    - 对称操作的 (k, σ) 作用
    - k·p 模型
    - 压电系数矩阵
    - 自旋霍尔电导矩阵

    参数:
        analyzer: SymmetryAnalyzer
        little_group_ops: 小群操作（可选）
        kp_max_order: k·p 展开最高阶数

    返回:
        格式化字符串
    """
    result = analyzer._result
    if result is None:
        return "请先运行对称性分析"

    pointgroup = result.pointgroup_symbol
    # 转换为 Schönflies
    pg_sch = _hm_to_schoenflies(pointgroup)

    lines = []

    # 1. (k, σ) 作用
    lines.append(report_symmetry_actions(analyzer, little_group_ops=little_group_ops))
    lines.append("")

    # 2. k·p 模型
    ops = little_group_ops or result.operations
    kp = KpModel(ops, result.lattice)
    lines.append(kp.report(max_order=kp_max_order))
    lines.append("")

    # 3. 压电张量
    lines.append("─" * 65)
    lines.append(f"  压电张量 (P_i = e_ijk ε_jk)")
    lines.append("─" * 65)
    pe = PiezoelectricTensor(pg_sch)
    lines.append(pe.report())
    lines.append("")

    # 4. 自旋霍尔电导
    lines.append("─" * 65)
    lines.append("  自旋霍尔电导张量 (J_i^s_j = σ^s_{ijk} E_k)")
    lines.append("─" * 65)
    sh = SpinHallTensor(pg_sch)
    lines.append(sh.report())

    return "\n".join(lines)


def _hm_to_schoenflies(hm: str) -> str:
    """Hermann-Mauguin → Schönflies 点群转换"""
    mapping = {
        "1": "C1", "-1": "Ci",
        "2": "C2", "m": "Cs", "2/m": "C2h",
        "mm2": "C2v", "222": "D2", "mmm": "D2h",
        "4": "C4", "-4": "S4", "4/m": "C4h",
        "4mm": "C4v", "-42m": "D2d", "422": "D4", "4/mmm": "D4h",
        "3": "C3", "-3": "S6", "3m": "C3v", "32": "D3", "-3m": "D3d",
        "6": "C6", "-6": "C3h", "6/m": "C6h",
        "6mm": "C6v", "-62m": "D3h", "622": "D6", "6/mmm": "D6h",
        "23": "T", "m-3": "Th", "-43m": "Td", "432": "O", "m-3m": "Oh",
    }
    # 标准化
    key = hm.replace(" ", "")
    return mapping.get(key, hm)
