"""命令行接口"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .poscar_reader import read_poscar
from .symmetry import SymmetryAnalyzer
from .little_group import LittleGroupAnalyzer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="VASP 结构对称性分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  vasp-symmetry POSCAR                           # 分析对称性
  vasp-symmetry POSCAR -k 0 0 0 --label Γ        # 分析 Γ 点小群
  vasp-symmetry POSCAR -k 0 0 0 -k 0.5 0 0.5     # 分析多个 k 点
  vasp-symmetry POSCAR --high-sym                # 列出高对称 k 点
  vasp-symmetry POSCAR --path Γ,0,0,0 X,0.5,0,0.5  # 分析 k 路径
  vasp-symmetry POSCAR --detail                   # 详细输出（含矩阵）
  vasp-symmetry POSCAR --symprec 1e-3             # 设置对称性精度
        """,
    )

    parser.add_argument("poscar", type=str, help="POSCAR 文件路径")
    parser.add_argument("-k", "--kpoint", action="append", nargs=3,
                        type=float, metavar=("kx", "ky", "kz"),
                        help="指定 k 点分数坐标")
    parser.add_argument("--label", type=str, action="append",
                        help="k 点标签 (与 -k 一一对应)")
    parser.add_argument("--high-sym", action="store_true",
                        help="显示常见高对称 k 点")
    parser.add_argument("--path", type=str,
                        help="k 路径分析, 格式: label1,x,y,z label2,x,y,z ...")
    parser.add_argument("--detail", action="store_true",
                        help="详细输出（含所有操作矩阵）")
    parser.add_argument("--symprec", type=float, default=1e-5,
                        help="对称性检测精度 (默认: 1e-5)")

    args = parser.parse_args(argv)

    # ------------------------------------------------
    # 1. 读取 POSCAR
    # ------------------------------------------------
    poscar_path = Path(args.poscar)
    if not poscar_path.exists():
        print(f"错误: 找不到文件 '{args.poscar}'")
        return 1

    try:
        structure = read_poscar(poscar_path)
    except Exception as e:
        print(f"错误: 读取 POSCAR 失败: {e}")
        return 1

    _print_header(f"POSCAR: {poscar_path.name}")
    print(f"  晶格常数: {structure.scale:.6f}")
    print(f"  元素: {structure.elements}")
    print(f"  原子数: {structure.num_atoms} (总计 {structure.total_atoms})")
    print(f"  晶格矢量:")
    for i, vec in enumerate(structure.lattice * structure.scale):
        print(f"    a{i + 1} = ({vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f})")
    print()

    # ------------------------------------------------
    # 2. 对称性分析
    # ------------------------------------------------
    print("正在检测对称性...")
    try:
        analyzer = SymmetryAnalyzer(structure, symprec=args.symprec)
        result = analyzer.analyze()
    except Exception as e:
        print(f"错误: 对称性分析失败: {e}")
        return 1

    if args.detail:
        print(result.full_report())
    else:
        print(result.summary())
    print()

    # ------------------------------------------------
    # 3. 高对称 k 点
    # ------------------------------------------------
    if args.high_sym:
        lg = LittleGroupAnalyzer(analyzer)
        kpoints = lg.list_high_symmetry_kpoints()

        _print_header("高对称 k 点（该晶系常见）")
        for label, k in kpoints.items():
            lg_result = lg.analyze_kpoint(k, label=label)
            print(f"  {label:4s}  ({k[0]:6.4f}, {k[1]:6.4f}, {k[2]:6.4f})  "
                  f"→ 小群: {lg_result.pointgroup:6s}  "
                  f"对称操作: {lg_result.num_operations}")
        print()

    # ------------------------------------------------
    # 4. 指定 k 点的小群分析
    # ------------------------------------------------
    if args.kpoint:
        lg = LittleGroupAnalyzer(analyzer)
        labels = args.label or []

        _print_header("小群（Little Group）分析")
        for i, k in enumerate(args.kpoint):
            label = labels[i] if i < len(labels) else ""
            k_arr = np.array(k)
            lg_result = lg.analyze_kpoint(k_arr, label=label)
            print(lg_result.summary())

            if args.detail:
                for op in lg_result.operations:
                    print(f"    旋转矩阵 R:")
                    for row in op.rotation:
                        print("      " + " ".join(f"{v:6.0f}" for v in row))
                    if np.any(np.abs(op.translation) > 1e-10):
                        print(f"    平移 τ = ({op.translation[0]:.6f}, "
                              f"{op.translation[1]:.6f}, {op.translation[2]:.6f})")
            print()

    # ------------------------------------------------
    # 5. k 路径分析
    # ------------------------------------------------
    if args.path:
        _print_header("k 路径小群分析")
        try:
            nodes = _parse_path(args.path)
            lg = LittleGroupAnalyzer(analyzer)
            results = lg.analyze_path(nodes, npoints=11)

            # 汇总不同段
            current_pg = None
            for r in results:
                if r.pointgroup != current_pg:
                    print(f"  {r.kpoint_label:15s}  →  {r.pointgroup:6s}  ({r.num_operations} ops)")
                    current_pg = r.pointgroup
        except Exception as e:
            print(f"错误: 路径分析失败: {e}")

    return 0


# ============================================================
# 辅助函数
# ============================================================

def _print_header(title: str):
    width = 60
    print("─" * width)
    print(f"  {title}")
    print("─" * width)


def _parse_path(path_str: str) -> list[tuple[str, list[float]]]:
    """解析路径参数 'Γ,0,0,0 X,0.5,0,0.5' """
    nodes = []
    segments = path_str.split()
    for seg in segments:
        parts = seg.split(",")
        if len(parts) != 4:
            raise ValueError(f"路径段格式错误: '{seg}'，需要 label,kx,ky,kz")
        label = parts[0]
        coords = [float(x) for x in parts[1:4]]
        nodes.append((label, coords))
    if len(nodes) < 2:
        raise ValueError("路径至少需要两个节点")
    return nodes


if __name__ == "__main__":
    sys.exit(main())
