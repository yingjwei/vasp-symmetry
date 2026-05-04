"""对称操作对 k 空间和自旋的作用、kp 模型、压电系数和自旋霍尔电导

基于 POSCAR 实际对称操作推导张量分量的约束关系（而非仅查表）。
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
SIGMA = [SIGMA_X, SIGMA_Y, SIGMA_Z]
SIGMA_I = np.eye(2, dtype=complex)
SIGMA_LABELS = ["σ_x", "σ_y", "σ_z"]


# ============================================================
# 对称操作对 k 和 sigma 的作用
# ============================================================

def rotation_matrix_cartesian(R: np.ndarray, lattice: np.ndarray | None = None) -> np.ndarray:
    """将 spglib 的整数旋转矩阵转为 Cartesian 坐标下的正交矩阵"""
    R = np.array(R, dtype=float)
    if lattice is not None:
        A = np.array(lattice, dtype=float)
        A_inv = np.linalg.inv(A)
        R_cart = A @ R @ A_inv
        U, _, Vt = np.linalg.svd(R_cart)
        R_cart = U @ Vt
        return R_cart
    return R


def sigma_transform_matrix(R: np.ndarray) -> np.ndarray:
    """sigma_i -> S_ij sigma_j, 其中 S_ij = det(R) * R_ij"""
    R = np.array(R, dtype=float)
    det = np.linalg.det(R)
    return det * R


@dataclass
class TransformationInfo:
    """单个对称操作对 (k, sigma) 的作用信息"""
    symbol: str
    op_type: str
    R_cart: np.ndarray
    S_matrix: np.ndarray
    k_transform: list[str]
    sigma_transform: list[str]

    def report(self) -> str:
        lines = [f"  操作: {self.symbol}  ({self.op_type})"]
        lines.append("  k 变换:")
        for i, label in enumerate(["x", "y", "z"]):
            lines.append(f"    k_{label} -> {self.k_transform[i]}")
        lines.append("  σ 变换:")
        for i, label in enumerate(["x", "y", "z"]):
            lines.append(f"    σ_{label} -> {self.sigma_transform[i]}")
        return "\n".join(lines)


def analyze_operation_action(
    op: SymmetryOp,
    lattice: np.ndarray | None = None,
    ndigits: int = 0,
) -> TransformationInfo:
    """分析单个对称操作对 (k, sigma) 的作用"""
    R = np.array(op.rotation, dtype=float)
    R_cart = rotation_matrix_cartesian(R, lattice)
    S = sigma_transform_matrix(R_cart)
    R_rounded = np.round(R_cart, ndigits)

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
                coeff = "1/2" if val > 0 else "-1/2"
            elif abs(val + 0.5) < 1e-10:
                coeff = "-1/2"
            else:
                v = int(val) if val == int(val) else val
                coeff = f"{v:+g}" if val < 0 else f"{v:g}"
            terms.append(f"{coeff}k_{['x','y','z'][j]}")
        expr = " + ".join(terms).replace("+ -", "- ")
        k_expr.append(expr)

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
                coeff = "1/2" if val > 0 else "-1/2"
            elif abs(val + 0.5) < 1e-10:
                coeff = "-1/2"
            else:
                v = int(val) if abs(val - int(val)) < 1e-10 else val
                coeff = f"{v:+g}" if val < 0 else f"{v:g}"
            terms.append(f"{coeff}σ_{['x','y','z'][j]}")
        expr = " + ".join(terms).replace("+ -", "- ")
        s_expr.append(expr)

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
    """生成对称操作的 (k, sigma) 作用报告"""
    result = analyzer._result
    if result is None:
        return "错误: 请先执行对称性分析"

    lattice = result.lattice
    if little_group_ops is not None:
        ops = little_group_ops
    else:
        ops = result.operations

    lines = ["-" * 65, "  对称操作对 (k, σ) 的作用", "-" * 65]

    for i, op in enumerate(ops):
        info = analyze_operation_action(op, lattice)
        lines.append(f"\n  操作 #{i + 1}: {info.symbol}  ({info.op_type})")
        lines.append("  R_cart (Cartesian 旋转矩阵):")
        for row in info.R_cart:
            lines.append("    " + " ".join(f"{v:+7.3f}" for v in row))
        lines.append("  k 变换:")
        for j, label in enumerate(["x", "y", "z"]):
            lines.append(f"    k_{label}  ->  {info.k_transform[j]}")
        lines.append("  σ 变换:")
        for j, label in enumerate(["x", "y", "z"]):
            lines.append(f"    σ_{label}  ->  {info.sigma_transform[j]}")

    return "\n".join(lines)


# ============================================================
# k.p 模型：对称性允许项展开
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
    """生成 k 多项式基（单项式的幂次组合）"""
    basis = [(0, 0, 0)]
    for order in range(1, max_order + 1):
        for nx in range(order + 1):
            for ny in range(order + 1 - nx):
                nz = order - nx - ny
                basis.append((nx, ny, nz))
    return basis


@dataclass
class KpTerm:
    """k.p 哈密顿量中的一个对称允许项"""
    order: int
    monomial: tuple[int, int, int]
    monomial_str: str
    matrix: np.ndarray
    label: str = ""

    def __repr__(self) -> str:
        return f"    {self.monomial_str:15s}  x  ({self.label:6s})"


class KpModel:
    """k.p 模型：从对称性推导哈密顿量的允许项

    对于每个单项式 f(k) 和 Pauli 矩阵 M，检查乘积 f(k)*M 在
    所有小群操作下是否满足不变性条件:
      D(R)^dagger * f(R*k) * M * D(R) = f(k) * M
    """

    SPIN_BASIS = [
        (SIGMA_I, "I"),
        (SIGMA_X, "σ_x"),
        (SIGMA_Y, "σ_y"),
        (SIGMA_Z, "σ_z"),
    ]

    def __init__(self, little_group_ops: list[SymmetryOp], lattice: np.ndarray | None = None):
        self.ops = little_group_ops
        self.lattice = lattice

    def expand(self, max_order: int = 2) -> list[list[KpTerm]]:
        """展开 k.p 哈密顿量到指定阶数"""
        all_terms = []
        for order in range(max_order + 1):
            terms = self._find_invariant_terms(order)
            if terms:
                all_terms.append(terms)
        return all_terms

    def _find_invariant_terms(self, order: int) -> list[KpTerm]:
        """找到给定阶数所有对称不变的项"""
        monomials = generate_k_polynomials(order)
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
        """检查乘积 f(k) * M 是否在所有对称操作下不变

        条件: D(R)^dagger * f(R*k) * M * D(R) = f(k) * M
        正确方法：f(k) 和 M 的变换可以互相抵消，乘积只需整体不变。
        用随机 k 点数值采样确认多项式恒等。
        """
        nx, ny, nz = monomial

        for op in self.ops:
            R = np.array(op.rotation, dtype=float)
            R_cart = rotation_matrix_cartesian(R, self.lattice)
            det = np.linalg.det(R_cart)
            S = det * R_cart  # sigma 变换矩阵

            # 数值采样：检查多项等式是否成立
            # f(R*k) * M' = f(k) * M  对于所有 k
            found = False
            for _ in range(7):
                k_test = np.random.randn(3) * 0.5
                k_rot = R_cart @ k_test

                f_orig = (k_test[0] ** nx) * (k_test[1] ** ny) * (k_test[2] ** nz)
                f_rot = (k_rot[0] ** nx) * (k_rot[1] ** ny) * (k_rot[2] ** nz)

                h_orig = f_orig * matrix
                m_trans = self._transform_matrix(matrix, S)
                h_trans = f_rot * m_trans

                if np.allclose(h_orig, h_trans, atol=1e-10):
                    found = True
                    break

            if not found:
                return False

        return True

    @staticmethod
    def _transform_matrix(matrix: np.ndarray, S: np.ndarray) -> np.ndarray:
        """D(R)^dagger M D(R) 在 Pauli 基下的变换：
           I -> I, sigma_i -> S_ij sigma_j
        """
        coeffs = np.zeros(4, dtype=float)
        for i, pauli in enumerate([SIGMA_I, SIGMA_X, SIGMA_Y, SIGMA_Z]):
            coeffs[i] = 0.5 * np.trace(matrix @ pauli).real

        result = coeffs[0] * SIGMA_I
        for i in range(3):
            new_c = sum(coeffs[j + 1] * S[j, i] for j in range(3))
            result += new_c * SIGMA[i]
        return result

    def report(self, max_order: int = 2) -> str:
        """生成 k.p 模型报告"""
        all_terms = self.expand(max_order)

        lines = [
            "-" * 65,
            f"  k.p 模型（对称性允许项，最高阶数 = {max_order})",
            "-" * 65,
        ]

        if not all_terms:
            lines.append("  (未找到任何对称允许项)")
            return "\n".join(lines)

        for order, terms in enumerate(all_terms):
            n_terms = len(terms)
            lines.append(f"\n  第 {order} 阶 ({n_terms} 项):")
            for term in terms:
                lines.append(f"    {term.monomial_str:15s}  x  {term.label:6s}")

        lines.append("\n" + "-" * 65)
        lines.append("  H(k) =")
        for order, terms in enumerate(all_terms):
            for term in terms:
                lines.append(f"    +  {term.monomial_str:15s}  *  {term.label:6s}")

        return "\n".join(lines)


# ============================================================
# 张量约束引擎：基于实际对称操作推导非零分量
# ============================================================

# Voigt 指标: xx=0, yy=1, zz=2, yz=3, xz=4, xy=5
VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
VOIGT_LABELS = ["xx", "yy", "zz", "yz", "xz", "xy"]
IDX_TO_XYZ = ["x", "y", "z"]


def _voigt_idx(j: int, k: int) -> int:
    """3x3 对称指标对 (j,k) -> Voigt 索引 (0-based)"""
    if j == k:
        return j
    if {j, k} == {1, 2}:
        return 3  # yz
    if {j, k} == {0, 2}:
        return 4  # xz
    if {j, k} == {0, 1}:
        return 5  # xy
    return -1


def _constrain_rank3_tensor(
    ops_rcart: list[np.ndarray],
    is_axial: bool = False,
    symmetrize_jk: bool = False,
    symmetrize_ij: bool = False,
    tol: float = 1e-8,
) -> list[tuple[int, int, int]]:
    """对三阶张量施加对称性约束，返回非零 (i,j,k) 列表

    使用群平均投影方法（SVD 求 (P−I) 零空间）正确找出所有对称允许的分量。
    等价于计算 Neumann 原理下不变的张量子空间。

    参数:
        ops_rcart: Cartesian 旋转矩阵列表
        is_axial: True=轴矢张量（多一个 det(R) 因子）
        symmetrize_jk: True=对 j,k 对称（如压电张量的应变指标）
        symmetrize_ij: True=对 i,j 对称（如自旋霍尔电导的自旋流对称 σ_{ijk}=σ_{jik}）

    返回:
        [(i,j,k), ...] 对称允许的非零分量列表（0-based 指标）
    """
    if not ops_rcart:
        all_idx = list(product(range(3), repeat=3))
        if symmetrize_jk:
            all_idx = [(i, j, k) for (i, j, k) in all_idx if j <= k]
        if symmetrize_ij:
            all_idx = [(i, j, k) for (i, j, k) in all_idx if i <= j]
        return all_idx

    dim = 27
    # 构建群平均投影算子 P (27×27)
    P = np.zeros((dim, dim))
    for R_cart in ops_rcart:
        det = np.linalg.det(R_cart)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    out = a * 9 + b * 3 + c
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                val = R_cart[a, i] * R_cart[b, j] * R_cart[c, k]
                                if is_axial:
                                    val *= det
                                P[out, i * 9 + j * 3 + k] += val
    P /= len(ops_rcart)

    # 求 (P - I) 的零空间 → 所有满足 P[T] = T 的不变张量
    A = P - np.eye(dim)
    u, s, vh = np.linalg.svd(A)
    null_mask = np.abs(s) < max(tol, 1e-10)
    nullspace = vh[null_mask]

    if nullspace.shape[0] == 0:
        return []

    allowed = set()
    for vec in nullspace:
        v3 = vec.reshape(3, 3, 3)
        if symmetrize_jk:
            v3_sym = v3.copy()
            for j in range(3):
                for k in range(j + 1, 3):
                    avg = (v3_sym[:, j, k] + v3_sym[:, k, j]) / 2.0
                    v3_sym[:, j, k] = avg
                    v3_sym[:, k, j] = avg
            v3 = v3_sym
        if symmetrize_ij:
            v3_sym = v3.copy()
            for i in range(3):
                for j in range(i + 1, 3):
                    avg = (v3_sym[i, j, :] + v3_sym[j, i, :]) / 2.0
                    v3_sym[i, j, :] = avg
                    v3_sym[j, i, :] = avg
            v3 = v3_sym
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if abs(v3[i, j, k]) > tol:
                        if symmetrize_jk and j > k:
                            continue
                        if symmetrize_ij and i > j:
                            continue
                        allowed.add((i, j, k))

    return sorted(allowed)


def _symmetry_ops_to_rcart(
    ops: list[SymmetryOp],
    lattice: np.ndarray | None = None,
) -> list[np.ndarray]:
    """提取对称操作的 Cartesian 旋转矩阵列表（只保留唯一矩阵）"""
    seen = set()
    unique = []
    for op in ops:
        R_cart = rotation_matrix_cartesian(op.rotation, lattice)
        key = tuple(np.round(R_cart, 8).flatten())
        if key not in seen:
            seen.add(key)
            unique.append(R_cart)
    return unique


# ============================================================
# 压电张量
# ============================================================

class PiezoelectricTensor:
    """压电张量 e_ijk

    P_i = e_ijk * epsilon_jk  (epsilon 为应变张量)
    Voigt 表示: P_i = e_ialpha * epsilon_alpha, alpha=xx,yy,zz,yz,xz,xy

    支持两种模式:
      1. 基于实际对称操作的推导
      2. 基于点群名称的查表
    """

    def __init__(self, pointgroup: str | None = None, ops: list[SymmetryOp] | None = None,
                 lattice: np.ndarray | None = None):
        self.pointgroup = pointgroup or ""
        self.ops = ops
        self.lattice = lattice
        self._derived = ops is not None

    def components(self) -> list[tuple[int, int, int]]:
        """返回 0-based (i,j,k) 列表"""
        if self._derived and self.ops:
            ops_rcart = _symmetry_ops_to_rcart(self.ops, self.lattice)
            return _constrain_rank3_tensor(ops_rcart, is_axial=False, symmetrize_jk=True)
        return self._lookup_components()

    def _lookup_components(self) -> list[tuple[int, int, int]]:
        TABLE = {
            "C1": [(1,1,1),(1,2,2),(1,3,3),(1,2,3),(1,1,3),(1,1,2),
                   (2,1,1),(2,2,2),(2,3,3),(2,2,3),(2,1,3),(2,1,2),
                   (3,1,1),(3,2,2),(3,3,3),(3,2,3),(3,1,3),(3,1,2)],
            "Ci": [],
            "Cs": [(1,1,1),(1,2,2),(1,3,3),(1,2,3),(1,1,2),
                   (2,1,1),(2,2,2),(2,3,3),(2,1,3),(2,2,3),
                   (3,1,3),(3,2,3),(3,3,3),(3,1,1),(3,2,2)],
            "C2": [(1,1,3),(1,2,3),(1,3,1),(1,3,2),
                   (2,1,3),(2,2,3),(2,3,1),(2,3,2),
                   (3,1,1),(3,2,2),(3,3,3),(3,1,2)],
            "C2v": [(1,1,3),(2,2,3),(3,1,1),(3,2,2),(3,3,3)],
            "C3": [(1,1,1),(1,1,2),(1,2,2),(1,2,3),
                   (2,1,1),(2,1,2),(2,2,2),(2,2,3),
                   (3,1,3),(3,2,3),(3,3,1),(3,3,2),(3,3,3)],
            "C3v": [(1,1,1),(1,1,2),(1,2,2),
                    (2,2,2),(2,1,1),(2,1,2),
                    (3,1,3),(3,2,3),(3,3,1),(3,3,2),(3,3,3)],
            "C4": [(1,1,3),(1,2,3),(1,3,1),(1,3,2),
                   (2,1,3),(2,2,3),(2,3,1),(2,3,2),
                   (3,1,1),(3,2,2),(3,3,3)],
            "C4v": [(1,1,3),(1,2,3),(1,3,1),(1,3,2),
                    (2,1,3),(2,2,3),(2,3,1),(2,3,2),
                    (3,1,1),(3,2,2),(3,3,3)],
            "D2d": [(1,2,3),(1,3,2),(2,1,3),(2,3,1),(3,1,2),(3,2,1)],
            "S4": [(1,1,3),(1,2,3),(1,3,1),(1,3,2),
                   (2,1,3),(2,2,3),(2,3,1),(2,3,2),
                   (3,1,1),(3,2,2),(3,3,3)],
            "C6": [(1,1,3),(1,2,3),(1,3,1),(1,3,2),
                   (2,1,3),(2,2,3),(2,3,1),(2,3,2),
                   (3,1,1),(3,2,2),(3,3,3)],
            "C6v": [(1,1,3),(1,2,3),(1,3,1),(1,3,2),
                    (2,1,3),(2,2,3),(2,3,1),(2,3,2),
                    (3,1,1),(3,2,2),(3,3,3)],
            "Td": [(1,2,3),(2,3,1),(3,1,2)],
            "D2": [], "D3": [], "D4": [], "D6": [],
            "D2h": [], "D3d": [], "D4h": [], "D6h": [],
            "D3h": [], "C3h": [], "C4h": [], "C6h": [], "S6": [],
            "Oh": [], "O": [], "T": [], "Th": [],
        }
        pg = self.pointgroup
        if pg in TABLE:
            comps = TABLE[pg]
            return [(i - 1, j - 1, k - 1) for i, j, k in comps]
        return []

    def voigt_matrix(self) -> np.ndarray:
        """返回 3x6 Voigt 矩阵（1 表示非零）"""
        m = np.zeros((3, 6), dtype=int)
        comps = self.components()
        for i, j, k in comps:
            alpha = _voigt_idx(j, k)
            if alpha >= 0:
                m[i, alpha] = 1
        return m

    def report(self) -> str:
        lines = []
        if self._derived:
            lines.append(f"  压电张量（基于 {len(self.ops)} 个对称操作推导）")
        else:
            lines.append(f"  压电张量（点群 {self.pointgroup}）")

        comps = self.components()
        if not comps:
            lines.append("    所有分量为 0")
            return "\n".join(lines)

        vmat = self.voigt_matrix()
        lines.append("  Voigt 3x6 矩阵 (i=1..3, alpha=xx,yy,zz,yz,xz,xy):")
        header = "       " + "  ".join(f"{l:>5s}" for l in VOIGT_LABELS)
        lines.append(header)
        for i in range(3):
            row = "  ".join(f"{'  1  ' if vmat[i, a] else '  .  '}" for a in range(6))
            lines.append(f"    i={i+1}  {row}")

        lines.append(f"\n  非零张量分量 ({len(comps)} 个):")
        xyz = IDX_TO_XYZ
        for i, j, k in sorted(comps):
            alpha = _voigt_idx(j, k)
            if alpha >= 0:
                vlabel = VOIGT_LABELS[alpha]
                lines.append(f"    e_{{{xyz[i]},{vlabel}}} = e_{{{xyz[i]}{xyz[j]}{xyz[k]}}}")
            else:
                lines.append(f"    e_{{{xyz[i]}{xyz[j]}{xyz[k]}}}")

        return "\n".join(lines)


# ============================================================
# 自旋霍尔电导张量
# ============================================================

class SpinHallTensor:
    """自旋霍尔电导张量 sigma^s_ijk

    J_i^{s_j} = sigma^s_ijk * E_k
    轴矢张量: 变换规则多一个 det(R) 因子

    支持推导模式和查表模式
    """

    def __init__(self, pointgroup: str | None = None, ops: list[SymmetryOp] | None = None,
                 lattice: np.ndarray | None = None):
        self.pointgroup = pointgroup or ""
        self.ops = ops
        self.lattice = lattice
        self._derived = ops is not None

    def components(self) -> list[tuple[int, int, int]]:
        """返回 0-based (i,j,k) 列表"""
        if self._derived and self.ops:
            ops_rcart = _symmetry_ops_to_rcart(self.ops, self.lattice)
            return _constrain_rank3_tensor(ops_rcart, is_axial=True,
                                           symmetrize_jk=False, symmetrize_ij=False)
        return self._lookup_components()

    def _lookup_components(self) -> list[tuple[int, int, int]]:
        TABLE = {
            "C1": [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)],
            "Ci": [],
            "Cs": [(0,2,0),(0,2,1),(1,2,0),(1,2,1),
                   (2,0,0),(2,1,1),(2,0,1),(2,1,0)],
            "C2v": [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)],
            "C4v": [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)],
            "D2h": [],
            "Oh": [],
            "Td": [(0,1,2),(1,2,0),(2,0,1)],
        }
        return TABLE.get(self.pointgroup, [])

    def report(self) -> str:
        lines = []
        if self._derived:
            lines.append(f"  自旋霍尔电导（基于 {len(self.ops)} 个对称操作推导）")
        else:
            lines.append(f"  自旋霍尔电导（点群 {self.pointgroup}）")

        comps = self.components()
        if not comps:
            lines.append("    所有分量为 0")
            return "\n".join(lines)

        lines.append(f"  非零分量 ({len(comps)} 个):")
        xyz = IDX_TO_XYZ
        for i, j, k in sorted(comps):
            lines.append(f"    σ^{{{xyz[j]}}}_{{{xyz[i]},{xyz[k]}}}")

        return "\n".join(lines)


# ============================================================
# 综合报告
# ============================================================

def _hm_to_schoenflies(hm: str) -> str:
    """Hermann-Mauguin -> Schoenflies 点群转换"""
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
    key = hm.replace(" ", "")
    return mapping.get(key, hm)


def full_tensor_report(
    analyzer: SymmetryAnalyzer,
    little_group_ops: list[SymmetryOp] | None = None,
    kp_max_order: int = 2,
) -> str:
    """生成完整的张量性质报告

    基于 POSCAR 的实际对称操作推导一切，而非仅查表。
    """
    result = analyzer._result
    if result is None:
        return "请先运行对称性分析"

    pointgroup = result.pointgroup_symbol
    pg_sch = _hm_to_schoenflies(pointgroup)

    lines = []

    # 1. (k, sigma) 作用
    lines.append(report_symmetry_actions(analyzer, little_group_ops=little_group_ops))
    lines.append("")

    # 2. k.p 模型
    ops = little_group_ops or result.operations
    kp = KpModel(ops, result.lattice)
    lines.append(kp.report(max_order=kp_max_order))
    lines.append("")

    # 3. 压电张量（基于实际对称操作推导）
    lines.append("-" * 65)
    lines.append("  压电张量 (P_i = e_ijk * epsilon_jk)")
    lines.append("-" * 65)
    pe = PiezoelectricTensor(ops=ops, lattice=result.lattice)
    lines.append(pe.report())
    lines.append("")

    # 4. 自旋霍尔电导（基于实际对称操作推导）
    lines.append("-" * 65)
    lines.append("  自旋霍尔电导张量 (J_i^{s_j} = σ^{j}_{ik} E_k)")
    lines.append("-" * 65)
    sh = SpinHallTensor(ops=ops, lattice=result.lattice)
    lines.append(sh.report())

    return "\n".join(lines)
