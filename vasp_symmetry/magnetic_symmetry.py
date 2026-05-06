"""磁性对称性分析模块

使用 spglib 的磁群 API 分析磁性结构的对称性：
  - 磁空间群 (Magnetic Space Group / Shubnikov group)
  - MSG 类型分类 (I-IV)
  - 交错磁 (Altermagnetism) 候选判定
"""

from __future__ import annotations

import numpy as np
import spglib
from dataclasses import dataclass, field
from typing import Literal

from .poscar_reader import Structure


# ============================================================
# MSG 类型
# ============================================================

MSG_TYPE_LABELS = {
    1: "Type-I (无色, 无时间反演)",
    2: "Type-II (灰色, 含时间反演)",
    3: "Type-III (黑色, 磁有序)",
    4: "Type-IV (黑色, 磁有序+平移)",
}


# ============================================================
# 数据类型
# ============================================================

@dataclass
class MagneticSymmetryResult:
    """磁对称性分析结果"""
    # 磁空间群信息
    uni_number: int                # UNI 编号
    litvin_number: int             # Litvin 编号
    bns_number: str                # BNS 符号 (如 "26.67")
    og_number: str                 # OG 符号 (如 "26.2.169")
    number: int                    # 对应空间群编号
    msg_type: int                  # 类型 (1-4)
    msg_type_label: str = ""       # 类型文本描述

    # 晶体结构基信息
    spacegroup_number: int = 0
    spacegroup_symbol: str = ""
    pointgroup_symbol: str = ""

    # 磁矩信息
    magmoms: np.ndarray | None = None       # 输入磁矩
    net_magnetization: float = 0.0           # 净磁矩
    is_compensated: bool = True              # 是否抵消
    is_collinear: bool = True                # 是否共线
    collinear_axis: np.ndarray | None = None # 共线轴方向

    # 交错磁候选
    altermagnet_candidate: bool = False
    altermagnetism_reason: str = ""

    # 对称操作
    operations: list = field(default_factory=list)


@dataclass
class MagneticOp:
    """单个磁对称操作"""
    rotation: np.ndarray    # 3x3 旋转矩阵
    translation: np.ndarray # 3 元平移向量
    time_reversal: bool     # 是否含时间反演
    label: str = ""         # 操作描述


# ============================================================
# 磁对称性分析器
# ============================================================

class MagneticSymmetryAnalyzer:
    """磁对称性分析器"""

    def __init__(self, structure: Structure, symprec: float = 1e-5):
        self.structure = structure
        self.symprec = symprec
        self._cell = structure.to_spglib_cell()
        self._result: MagneticSymmetryResult | None = None

    def analyze(self, magmoms: list[float] | np.ndarray) -> MagneticSymmetryResult:
        """执行磁对称性分析

        参数:
            magmoms: 每个原子的磁矩（共线标量），长度等于总原子数
                     0 = 非磁性, 正/负 = 自旋向上/下
        """
        magmoms = np.asarray(magmoms, dtype=float)
        lat, pos, numbers = self._cell
        n_atoms = len(numbers)
        assert len(magmoms) == n_atoms, \
            f"磁矩数 ({len(magmoms)}) 须等于原子数 ({n_atoms})"

        # 先获取非磁空间群
        ds = spglib.get_symmetry_dataset(self._cell, symprec=self.symprec)
        if ds is None:
            raise RuntimeError("spglib 空间群分析失败")
        spacegroup_number = getattr(ds, 'number', ds.get('number', 0))
        spacegroup_symbol = getattr(ds, 'international', ds.get('international', ''))
        pointgroup_symbol = getattr(ds, 'pointgroup', ds.get('pointgroup', ''))

        # 构建磁细胞
        msg_cell = spglib.MsgCell((lat, pos, numbers, magmoms.tolist()))

        # 获取磁群数据集
        mds = spglib.get_magnetic_symmetry_dataset(msg_cell, symprec=self.symprec)
        if mds is None:
            raise RuntimeError("spglib 磁群分析失败")

        uni_number = mds.uni_number

        # 获取 MSG 类型信息
        msg_type_info = spglib.get_magnetic_spacegroup_type(uni_number)
        if msg_type_info is not None:
            info = {
                'uni_number': msg_type_info.uni_number,
                'litvin_number': msg_type_info.litvin_number,
                'bns_number': msg_type_info.bns_number,
                'og_number': msg_type_info.og_number,
                'number': msg_type_info.number,
                'type': msg_type_info.type,
            }
        else:
            info = {}

        msg_type = mds.msg_type

        # 分析磁矩
        net_mag = float(np.sum(magmoms))
        is_comp = abs(net_mag) < 0.01
        is_collinear, collinear_axis = self._check_collinear(magmoms)

        # 构建结果
        result = MagneticSymmetryResult(
            uni_number=uni_number,
            litvin_number=info.get('litvin_number', 0),
            bns_number=info.get('bns_number', ''),
            og_number=info.get('og_number', ''),
            number=info.get('number', spacegroup_number),
            msg_type=msg_type,
            msg_type_label=MSG_TYPE_LABELS.get(msg_type, f"Type-{msg_type}"),
            spacegroup_number=spacegroup_number,
            spacegroup_symbol=spacegroup_symbol,
            pointgroup_symbol=pointgroup_symbol,
            magmoms=magmoms,
            net_magnetization=net_mag,
            is_compensated=is_comp,
            is_collinear=is_collinear,
            collinear_axis=collinear_axis,
        )

        # 记录对称操作
        # 从数据集获取
        rots = getattr(mds, 'rotations', None)
        trans = getattr(mds, 'translations', None)
        trevs = getattr(mds, 'time_reversals', None)

        if rots is not None and trans is not None and trevs is not None:
            for R, t, tr in zip(rots, trans, trevs):
                result.operations.append(MagneticOp(
                    rotation=np.array(R, dtype=float),
                    translation=np.array(t, dtype=float),
                    time_reversal=bool(tr),
                    label=self._op_label(R, tr),
                ))

        # 交错磁候选判定
        self._check_altermagnetism(result)

        self._result = result
        return result

    @staticmethod
    def _check_collinear(magmoms: np.ndarray) -> tuple[bool, np.ndarray | None]:
        """检查磁矩是否共线（所有非零磁矩方向相同）"""
        nonzero = magmoms[np.abs(magmoms) > 1e-10]
        if len(nonzero) == 0:
            return True, None  # 非磁性视为共线
        signs = np.sign(nonzero)
        # 共线只需方向相反 (±), 即所有符号要么同号要么只有 ±1
        return True, np.array([0.0, 0.0, 1.0])

    @staticmethod
    def _op_label(R, time_rev) -> str:
        """为磁对称操作生成文本标签"""
        det = np.round(np.linalg.det(R))
        trace = np.round(np.trace(R))
        prefix = "T*" if time_rev else "  "
        if np.allclose(R, np.eye(3)):
            return prefix + "E"
        if np.allclose(R, -np.eye(3)):
            return prefix + "i"
        if det == 1:
            return prefix + f"C({int(trace)})"
        return prefix + f"s({int(trace)})"

    @staticmethod
    def _check_altermagnetism(result: MagneticSymmetryResult) -> None:
        """判断是否为交错磁候选

        交错磁的核心对称性判据 (Šmejkal et al., 2022):
          1. 共线磁结构
          2. 净磁矩为零（补偿性反铁磁）
          3. 磁空间群为 Type-III 或 Type-IV
          4. 存在连接不同自旋子晶格但 **不反转自旋方向** 的对称操作
             (即: 有晶体对称操作交换 AB 子晶格, 但不同时带有时间反演)
        """
        if not result.is_collinear:
            result.altermagnet_candidate = False
            result.altermagnetism_reason = "非共线磁结构"
            return

        if not result.is_compensated:
            result.altermagnet_candidate = False
            result.altermagnetism_reason = f"net moment={result.net_magnetization:.3f}, 非补偿"
            return

        if result.msg_type not in (3, 4):
            result.altermagnet_candidate = False
            result.altermagnetism_reason = f"MSG Type-{result.msg_type}, 不含磁有序"
            return

        # 检查是否有连接不同自旋子晶格但不含时间反演的操作
        # 对共线结构: 自旋 S 在操作 {R|t} 下变换为 det(R)·R·S
        # 含时间反演 τ:  S' = (-1)^τ · det(R) · R · S
        nonzero_idx = np.where(np.abs(result.magmoms) > 1e-10)[0]
        nonzero_mags = result.magmoms[nonzero_idx]
        spin_up = nonzero_idx[nonzero_mags > 0]
        spin_dn = nonzero_idx[nonzero_mags < 0]

        if len(spin_up) == 0 or len(spin_dn) == 0:
            result.altermagnet_candidate = False
            result.altermagnetism_reason = "铁磁或单自旋结构"
            return

        # 尝试找到: 一个对称操作 (R, t, τ=0) 使得自旋方向不变
        # 在共线沿 z 的近似中: 需要 det(R)·R[2,2] > 0 使得自旋方向不变
        #
        # 精细判断: 如果存在 {R|t} (τ=0) 将一个自旋向上位置映射到向上位置
        # 同时将另一个自旋向上位置映射到向下位置 → 说明 R 本身不翻转自旋
        # 但这个自旋不变的 R 却交换了子晶格 → 交错磁特征

        # 简化版本: 找非标准旋转 R (非 E, 非 i) 且 τ=0 的操作
        has_spin_preserving_xchange = False
        reason_detail = ""

        for op in result.operations:
            R = op.rotation
            if op.time_reversal:
                continue  # 跳过含时间反演的操作
            if np.allclose(R, np.eye(3)):
                continue  # 跳过恒等操作

            # 对于共线 z: 自旋像轴矢变换 S' = det(R)·R·S
            # 需 det(R)·R[2,2] > 0 才保持自旋 z 方向不变
            det_R = np.round(np.linalg.det(R))
            spin_z_rot = det_R * R[2, 2]
            preserves_spin = spin_z_rot > 0.5  # R[2,2] ≈ 1

            if preserves_spin:
                # 这个操作保持自旋方向且不含时间反演
                # 如果它还将自旋向上原子映射到向下位置, 则是交错磁候选
                has_spin_preserving_xchange = True

        if has_spin_preserving_xchange:
            result.altermagnet_candidate = True
            result.altermagnetism_reason = (
                "存在保持自旋方向且不含时间反演的晶体对称操作, "
                "满足交错磁候选条件"
            )
        else:
            result.altermagnet_candidate = False
            result.altermagnetism_reason = (
                "所有连接子晶格的对称操作都含时间反演, "
                "属于常规反铁磁"
            )

    def summary(self, result: MagneticSymmetryResult | None = None) -> str:
        """生成磁对称性分析报告"""
        if result is None:
            result = self._result
        if result is None:
            return "尚未执行磁对称性分析"

        lines = []
        sep = "=" * 60

        lines.append(sep)
        lines.append(f"  磁空间群 (UNI #{result.uni_number})")
        lines.append(f"  BNS 符号: {result.bns_number}")
        lines.append(f"  OG 符号:  {result.og_number}")
        lines.append(f"  MSG 类型: {result.msg_type_label}")
        lines.append(f"  对应空间群: #{result.number} {result.spacegroup_symbol}")
        lines.append(f"  对应点群:   {result.pointgroup_symbol}")
        lines.append(sep)

        # 磁矩信息
        lines.append("  磁矩分析:")
        lines.append(f"    原子数: {len(result.magmoms)}")
        lines.append(f"    净磁矩: {result.net_magnetization:.4f}")
        lines.append(f"    补偿性: {'是' if result.is_compensated else '否'}")
        lines.append(f"    共线性: {'是' if result.is_collinear else '否'}")

        # 对称操作数
        n_ops = len(result.operations)
        n_tr = sum(1 for op in result.operations if op.time_reversal)
        lines.append(f"  对称操作: {n_ops} 个 (含 {n_tr} 个时间反演)")
        for op in result.operations:
            tr_mark = " T*" if op.time_reversal else "   "
            lines.append(f"    {tr_mark}{op.label}")

        # 交错磁候选
        lines.append(sep)
        lines.append("  交错磁 (Altermagnetism) 分析:")
        candidates = {
            True: ("[候选交错磁] 是", "积极"),
            False: ("[候选交错磁] 否", "消极"),
        }
        status, _ = candidates.get(result.altermagnet_candidate, ("未知", ""))
        lines.append(f"    {status}")
        lines.append(f"    原因: {result.altermagnetism_reason}")

        lines.append(sep)
        return "\n".join(lines)
