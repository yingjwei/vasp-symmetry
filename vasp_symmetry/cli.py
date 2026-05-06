"""命令行接口 — 交互菜单 (VASPKIT 风格) + 批处理模式"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .poscar_reader import read_poscar, Structure
from .symmetry import SymmetryAnalyzer
from .little_group import LittleGroupAnalyzer
from .symmetry_tensors import (
    report_symmetry_actions, KpModel, PiezoelectricTensor,
    SpinHallTensor, full_tensor_report,
)


# ============================================================
# 交互菜单
# ============================================================

MENU_ITEMS = [
    ("对称性分析", "空间群 + 对称操作分类"),
    ("k 路径分析", "沿高对称路径追踪 k 点小群变化"),
    ("指定 k 点小群", "分析单个/多个 k 点的小群"),
    ("高对称 k 点列表", "显示该晶系常见高对称 k 点"),
    ("k·p 模型", "对称性允许项展开"),
    ("压电 & 自旋霍尔张量", "基于实际对称操作推导"),
    ("磁空间群 + 交错磁", "MSG 类型分类, Altermagnetism 候选判定"),
    ("对称操作对 (k,σ) 作用", "每个对称操作对波矢和自旋的变换"),
    ("全部功能依次执行", "运行以上所有分析"),
]


def run_interactive(structure: Structure, symprec: float) -> int:
    """交互菜单主循环"""
    # 预检测晶系
    try:
        analyzer = SymmetryAnalyzer(structure, symprec=symprec)
        _ = analyzer.analyze()
        crystal_sys = LittleGroupAnalyzer._crystal_system(_.spacegroup_number)
    except Exception:
        crystal_sys = "未知"

    while True:
        _print_header(f"POSCAR: {structure.comment or '未命名'}  |  {crystal_sys}晶系")
        print(f"  元素: {structure.elements}  |  原子: {structure.total_atoms}")
        print()

        print("=" * 58)
        print("  vasp-symmetry 任务选择")
        print("=" * 58)
        for i, (title, desc) in enumerate(MENU_ITEMS, 1):
            print(f"  {i:2d}) {title}")
            print(f"     {desc}")
        print(f"  0) 退出")
        print("=" * 58)

        try:
            choice = input("  请选择 [0-9]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见。")
            break

        if choice == "0":
            print("  再见。")
            break
        elif choice == "1":
            _task_symmetry(structure, symprec)
        elif choice == "2":
            _task_kpath(structure, symprec)
        elif choice == "3":
            _task_kpoints(structure, symprec)
        elif choice == "4":
            _task_highsym(structure, symprec)
        elif choice == "5":
            _task_kp(structure, symprec)
        elif choice == "6":
            _task_tensor(structure, symprec)
        elif choice == "7":
            _task_magnetic(structure, symprec)
        elif choice == "8":
            _task_pauli(structure, symprec)
        elif choice == "9":
            _task_all(structure, symprec)
        else:
            print("  无效选择，请重试。")

        input("\n  按 Enter 继续...")
        print()

    return 0


# ============================================================
# 各任务实现
# ============================================================

def _get_analyzer(structure, symprec):
    analyzer = SymmetryAnalyzer(structure, symprec=symprec)
    result = analyzer.analyze()
    return analyzer, result


def _task_symmetry(structure, symprec):
    _print_header("对称性分析")
    analyzer, result = _get_analyzer(structure, symprec)
    print(result.summary())
    print()

    # 询问是否查看详细矩阵
    try:
        detail = input("  是否查看详细矩阵? (y/n, 默认 n): ").strip().lower()
        if detail in ("y", "yes", "是"):
            print(result.full_report())
    except (EOFError, KeyboardInterrupt):
        pass


def _task_kpath(structure, symprec):
    _print_header("k 路径小群分析")

    # 获取晶系和预定义路径
    analyzer, _ = _get_analyzer(structure, symprec)
    lg = LittleGroupAnalyzer(analyzer)

    try:
        paths = lg.list_standard_paths()
        print(f"  检测到晶系, 可用预定义路径:\n")
        for i, (name, path_str) in enumerate(paths, 1):
            print(f"  {i:2d}) {name}")
            print(f"     {path_str}")
        print(f"  0) 自定义输入")
        print()

        choice = input("  请选择路径 [0-{}]: ".format(len(paths))).strip()
        if not choice:
            print("  已取消。")
            return
        choice_num = int(choice)
        if choice_num == 0:
            path_str = input("  请输入自定义路径: ").strip()
            if not path_str:
                print("  已取消。")
                return
        elif 1 <= choice_num <= len(paths):
            path_str = paths[choice_num - 1][1]
            print(f"  已选择: {paths[choice_num - 1][0]}")
        else:
            print("  无效选择。")
            return
    except (EOFError, KeyboardInterrupt):
        return
    except ValueError:
        print("  无效输入。")
        return

    try:
        nodes = _parse_path(path_str)
        results = lg.analyze_path(nodes, npoints=11)

        print(f"\n  {'k 路径':20s}  {'点群':8s}  {'操作数':8s}")
        print("  " + "-" * 40)
        current_pg = None
        for r in results:
            if r.pointgroup != current_pg:
                print(f"  {r.kpoint_label:20s}  →  {r.pointgroup:8s}  ({r.num_operations:3d} ops)")
                current_pg = r.pointgroup
    except Exception as e:
        print(f"  错误: {e}")


def _task_kpoints(structure, symprec):
    _print_header("k 点小群分析")
    print("  输入分数坐标, 每行一个 k 点, 空行结束")
    print("  可选: 可在坐标前加标签, 如 G 0 0 0")
    print()

    kpoints = []
    labels = []
    try:
        while True:
            line = input(f"  k 点 #{len(kpoints) + 1} (或空行结束): ").strip()
            if not line:
                break
            parts = line.split()
            if len(parts) == 4:
                label, *coords = parts
                labels.append(label)
                kpoints.append([float(x) for x in coords])
            elif len(parts) == 3:
                labels.append("")
                kpoints.append([float(x) for x in parts])
            else:
                print("  格式错误: 需要 3 个坐标 或 标签+3个坐标")
    except (EOFError, KeyboardInterrupt):
        pass

    if not kpoints:
        print("  已取消。")
        return

    analyzer, _ = _get_analyzer(structure, symprec)
    lg = LittleGroupAnalyzer(analyzer)

    _print_header("小群分析结果")
    for i, (k, label) in enumerate(zip(kpoints, labels)):
        lg_result = lg.analyze_kpoint(np.array(k), label=label)
        print(lg_result.summary())
        print()


def _task_highsym(structure, symprec):
    _print_header("高对称 k 点")
    analyzer, _ = _get_analyzer(structure, symprec)
    lg = LittleGroupAnalyzer(analyzer)
    kpoints = lg.list_high_symmetry_kpoints()

    print(f"  {'标签':6s}  {'坐标':24s}  {'小群':10s}  {'操作数':8s}")
    print("  " + "-" * 54)
    for label, k in kpoints.items():
        lg_result = lg.analyze_kpoint(k, label=label)
        coord = f"({k[0]:.4f}, {k[1]:.4f}, {k[2]:.4f})"
        print(f"  {label:6s}  {coord:24s}  {lg_result.pointgroup:10s}  ({lg_result.num_operations:3d} ops)")


def _task_kp(structure, symprec):
    _print_header("k·p 模型")
    try:
        order_str = input("  展开最高阶数 (默认 2): ").strip()
        max_order = int(order_str) if order_str else 2
    except (EOFError, KeyboardInterrupt):
        return
    except ValueError:
        max_order = 2

    analyzer, result = _get_analyzer(structure, symprec)
    lg = LittleGroupAnalyzer(analyzer)
    lg_result = lg.analyze_kpoint([0, 0, 0], label="Γ")

    kp = KpModel(lg_result.operations, result.lattice)
    print(kp.report(max_order=max_order))


def _task_tensor(structure, symprec):
    _print_header("压电 & 自旋霍尔张量")
    analyzer, _ = _get_analyzer(structure, symprec)
    print(full_tensor_report(analyzer))


def _task_magnetic(structure, symprec):
    _print_header("磁空间群 & 交错磁分析")
    n_atoms = structure.total_atoms

    print(f"  总原子数: {n_atoms}")
    print(f"  元素分布: {dict(zip(structure.elements, structure.num_atoms))}")
    print()
    print("  逐原子输入磁矩, 回车=0")
    print("  正数=自旋向上, 负数=自旋向下, 0=非磁性")
    print()

    try:
        import readline
    except ImportError:
        pass  # Windows: readline unavailable, fine

    magmoms = []
    while True:
        magmoms = []
        error = False
        try:
            for elem, cnt in zip(structure.elements, structure.num_atoms):
                for i in range(1, cnt + 1):
                    label = f"{elem}{i}"
                    prompt = f"    {label:<5} [{0}]: "
                    val = input(prompt).strip()
                    if val == "":
                        magmoms.append(0.0)
                    else:
                        magmoms.append(float(val))
        except (EOFError, KeyboardInterrupt):
            print()
            return
        except ValueError:
            print("  格式错误，只能输入数字。重试该原子。")
            error = True

        if error:
            print("  (重新输入该原子的值)\n")
            continue

        print()
        print(f"  磁矩: {' '.join(f'{m:>6.2f}' for m in magmoms)}")
        confirm = input("  确认? (y/n, 默认 y): ").strip().lower()
        if confirm != 'n':
            break
        print("  (重新输入)\n")

    try:
        from .magnetic_symmetry import MagneticSymmetryAnalyzer
        mag_analyzer = MagneticSymmetryAnalyzer(structure, symprec=symprec)
        mag_result = mag_analyzer.analyze(magmoms)
        print(mag_analyzer.summary(mag_result))
    except Exception as e:
        print(f"  分析失败: {e}")


def _task_pauli(structure, symprec):
    _print_header("对称操作对 (k, σ) 的作用")
    analyzer, _ = _get_analyzer(structure, symprec)
    print(report_symmetry_actions(analyzer))


def _task_all(structure, symprec):
    _print_header("全部功能依次执行")
    input("  即将依次运行所有分析, 按 Enter 继续...")
    print()

    _task_symmetry(structure, symprec)
    print("─" * 60)
    _task_highsym(structure, symprec)
    print("─" * 60)
    _task_kp(structure, symprec)
    print("─" * 60)
    _task_tensor(structure, symprec)
    print("─" * 60)

    # 磁分析需要交互输入, 跳过
    print("  磁空间群 & 交错磁分析:")
    print("    跳过 (需交互输入磁矩)")
    print()


# ============================================================
# 批处理模式 (原 CLI 接口)
# ============================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="VASP 结构对称性分析工具 — 交互菜单 + 批处理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""批处理模式使用示例:
  vasp-symmetry POSCAR                            交互菜单
  vasp-symmetry POSCAR --detail                  对称性 + 详细矩阵
  vasp-symmetry POSCAR --path "G,0,0,0 X,0.5,0,0.5"   k 路径
  vasp-symmetry POSCAR -k 0 0 0 --label Γ        k 点小群
  vasp-symmetry POSCAR --high-sym                高对称 k 点
  vasp-symmetry POSCAR --kp                       k·p 模型
  vasp-symmetry POSCAR --tensor                  压电/自旋霍尔张量
  vasp-symmetry POSCAR --magmom "1 -1 0..."      磁空间群
  vasp-symmetry POSCAR --symprec 1e-3            调整精度

不指定任务参数时进入交互菜单模式。
        """,
    )

    parser.add_argument("poscar", type=str, nargs="?", default=None,
                        help="POSCAR 文件路径 (省略则进入菜单)")

    # 功能标志
    parser.add_argument("--detail", action="store_true",
                        help="详细输出（含所有操作矩阵）")
    parser.add_argument("--high-sym", action="store_true",
                        help="显示常见高对称 k 点")
    parser.add_argument("--pauli-action", action="store_true",
                        help="分析对称操作对 (k, σ) 的变换作用")
    parser.add_argument("--kp", action="store_true",
                        help="对 Γ 点进行 k·p 模型分析")
    parser.add_argument("--tensor", action="store_true",
                        help="输出压电系数和自旋霍尔电导张量")
    parser.add_argument("--magnetic", action="store_true",
                        help="执行磁空间群和交错磁分析")

    # 参数
    parser.add_argument("-k", "--kpoint", action="append", nargs=3,
                        type=float, metavar=("kx", "ky", "kz"),
                        help="指定 k 点分数坐标")
    parser.add_argument("--label", type=str, action="append",
                        help="k 点标签 (与 -k 一一对应)")
    parser.add_argument("--path", type=str,
                        help="k 路径分析, 格式: label1,x,y,z label2,x,y,z ...")
    parser.add_argument("--symprec", type=float, default=1e-5,
                        help="对称性检测精度 (默认: 1e-5)")
    parser.add_argument("--kp-order", type=int, default=2,
                        help="k·p 展开最高阶数 (默认: 2)")
    parser.add_argument("--magmom", type=str, default=None,
                        help="磁矩列表, 如 '0 0 0 1 -1 0.5 -0.5 0.5 -0.5'")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="强制进入交互菜单模式")

    args = parser.parse_args(argv)

    # -------------------------------------------------------
    # 决定模式: 交互菜单 vs 批处理
    # -------------------------------------------------------
    # 批处理模式: 给了具体任务参数 或 --detail (修饰默认对称性分析)
    has_task_flags = any([
        args.kpoint, args.high_sym, args.path,
        args.pauli_action, args.kp, args.tensor,
        args.magmom is not None, args.magnetic,
        args.detail,
    ])

    use_interactive = args.interactive or (args.poscar and not has_task_flags)

    if args.poscar is None and not has_task_flags:
        # 没给 POSCAR 也没给任务: 询问文件路径后进入菜单
        return _interactive_ask_file(symprec=args.symprec)

    # 读取 POSCAR
    poscar_path = Path(args.poscar)
    if not poscar_path.exists() and not use_interactive:
        print(f"错误: 找不到文件 '{args.poscar}'")
        return 1

    try:
        structure = read_poscar(poscar_path)
    except Exception as e:
        if use_interactive:
            print(f"错误: 读取 POSCAR 失败: {e}")
            return 1
        print(f"错误: 读取 POSCAR 失败: {e}")
        return 1

    if use_interactive:
        return run_interactive(structure, symprec=args.symprec)

    # -------------------------------------------------------
    # 批处理模式
    # -------------------------------------------------------
    _print_header(f"POSCAR: {poscar_path.name}")
    print(f"  晶格常数: {structure.scale:.6f}")
    print(f"  元素: {structure.elements}")
    print(f"  原子数: {structure.num_atoms} (总计 {structure.total_atoms})")
    print(f"  晶格矢量:")
    for i, vec in enumerate(structure.lattice * structure.scale):
        print(f"    a{i + 1} = ({vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f})")
    print()

    # 2. 对称性分析 (基本输出)
    print("正在检测对称性...")
    analyzer = SymmetryAnalyzer(structure, symprec=args.symprec)
    result = analyzer.analyze()

    if args.detail:
        print(result.full_report())
    else:
        print(result.summary())
    print()

    # 3. 高对称 k 点
    if args.high_sym:
        _print_header("高对称 k 点")
        lg = LittleGroupAnalyzer(analyzer)
        kpoints = lg.list_high_symmetry_kpoints()
        for label, k in kpoints.items():
            lg_result = lg.analyze_kpoint(k, label=label)
            print(f"  {label:4s}  ({k[0]:6.4f}, {k[1]:6.4f}, {k[2]:6.4f})  "
                  f"→ 小群: {lg_result.pointgroup:6s}  "
                  f"对称操作: {lg_result.num_operations}")
        print()

    # 4. 指定 k 点小群
    if args.kpoint:
        _print_header("小群（Little Group）分析")
        lg = LittleGroupAnalyzer(analyzer)
        labels = args.label or []
        for i, k in enumerate(args.kpoint):
            label = labels[i] if i < len(labels) else ""
            lg_result = lg.analyze_kpoint(np.array(k), label=label)
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

    # 5. k 路径分析
    if args.path:
        _print_header("k 路径小群分析")
        nodes = _parse_path(args.path)
        lg = LittleGroupAnalyzer(analyzer)
        results = lg.analyze_path(nodes, npoints=11)

        current_pg = None
        for r in results:
            if r.pointgroup != current_pg:
                print(f"  {r.kpoint_label:15s}  →  {r.pointgroup:6s}  ({r.num_operations} ops)")
                current_pg = r.pointgroup
        print()

    # 6. 对称操作对 (k, σ) 的作用
    if args.pauli_action:
        _print_header("对称操作对 (k, σ) 的作用")
        if args.kpoint:
            lg = LittleGroupAnalyzer(analyzer)
            labels = args.label or []
            for i, k in enumerate(args.kpoint):
                label = labels[i] if i < len(labels) else f"k{i}"
                lg_result = lg.analyze_kpoint(k, label=label)
                print(f"\nk 点: {label} ({', '.join(f'{x:.4f}' for x in k)})")
                print(report_symmetry_actions(
                    analyzer, little_group_ops=lg_result.operations))
        else:
            print(report_symmetry_actions(analyzer))
        print()

    # 7. k·p 模型
    if args.kp:
        _print_header("k·p 模型（对称性允许项）")
        lg = LittleGroupAnalyzer(analyzer)
        if args.kpoint:
            labels = args.label or []
            for i, k in enumerate(args.kpoint):
                label = labels[i] if i < len(labels) else f"k{i}"
                lg_result = lg.analyze_kpoint(k, label=label)
                print(f"\nk 点: {label} ({', '.join(f'{x:.4f}' for x in k)})")
                kp = KpModel(lg_result.operations, result.lattice)
                print(kp.report(max_order=args.kp_order))
        else:
            lg_result = lg.analyze_kpoint([0, 0, 0], label="Γ")
            kp = KpModel(lg_result.operations, result.lattice)
            print(kp.report(max_order=args.kp_order))
        print()

    # 8. 磁对称性分析
    if args.magmom is not None or args.magnetic:
        mag_str = args.magmom or ""
        magmoms = [float(x) for x in mag_str.replace(",", " ").split() if x.strip()]
        if len(magmoms) != structure.total_atoms:
            print(f"错误: 磁矩数 ({len(magmoms)}) 不等于总原子数 ({structure.total_atoms})")
            return 1

        _print_header("磁对称性 (Magnetic Space Group) 分析")
        from .magnetic_symmetry import MagneticSymmetryAnalyzer
        mag_analyzer = MagneticSymmetryAnalyzer(structure, symprec=args.symprec)
        mag_result = mag_analyzer.analyze(magmoms)
        print(mag_analyzer.summary(mag_result))
        print()

    # 9. 张量性质
    if args.tensor:
        _print_header("压电 & 自旋霍尔张量")
        print(full_tensor_report(analyzer))
        print()

    return 0


def _interactive_ask_file(symprec: float) -> int:
    """解析器没给文件时, 询问路径后进入菜单"""
    print("=" * 58)
    print("  vasp-symmetry — VASP 结构对称性分析工具")
    print("=" * 58)
    print()
    try:
        path = input("  请输入 POSCAR 文件路径: ").strip()
        if not path:
            return 0
        p = Path(path)
        if not p.exists():
            print(f"  错误: 找不到 '{path}'")
            return 1
        structure = read_poscar(p)
        return run_interactive(structure, symprec=symprec)
    except (EOFError, KeyboardInterrupt):
        return 0
    except Exception as e:
        print(f"  错误: {e}")
        return 1


# ============================================================
# 辅助函数
# ============================================================

def _print_header(title: str):
    width = 60
    print("─" * width)
    print(f"  {title}")
    print("─" * width)


def _parse_path(path_str: str) -> list[tuple[str, list[float]]]:
    """解析路径参数 'G,0,0,0 X,0.5,0,0.5' """
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
