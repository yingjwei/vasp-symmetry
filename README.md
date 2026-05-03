# vasp-symmetry

VASP 结构对称性分析工具包。读取 POSCAR 文件，检测所有对称操作（旋转轴、镜面、反演、滑移面、螺旋轴），以矩阵形式输出，并支持 k 点小群（Little Group）分析。

## 依赖

```bash
pip install numpy scipy spglib pymatgen
```

## 快速开始

```bash
python run.py POSCAR
python run.py POSCAR --detail
python run.py POSCAR -k 0 0 0 --label G
python run.py POSCAR --high-sym
python run.py POSCAR --path "G,0,0,0 X,0.5,0,0.5"
```

## 功能

- **对称性分析**: 自动检测空间群、点群
- **对称操作分类**: C2/C3/C4/C6 旋转轴、镜面、反演、滑移面(a/b/c/n/d)、螺旋轴(2₁/3₁/4₁)
- **矩阵表示**: 每个对称操作输出 3×3 旋转矩阵和平移向量
- **小群分析**: 给定 k 点，找出空间中保持该 k 点不变的对称子群
- **k 路径分析**: 沿高对称路径追踪对称性变化

## 结构

```
vasp_symmetry/
├── run.py                    # 入口
├── vasp_symmetry/
│   ├── poscar_reader.py      # POSCAR 解析
│   ├── symmetry.py           # 对称性检测与分类
│   ├── little_group.py       # 小群分析
│   └── cli.py                # CLI 接口
└── examples/
    ├── POSCAR_Si             # 金钢石硅
    └── POSCAR_NaCl           # 氯化钠
```

## 文档

详细文档见 [docs/index.html](https://yingjwei.github.io/vasp-symmetry/)
