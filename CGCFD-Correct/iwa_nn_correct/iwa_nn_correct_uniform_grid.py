#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
通用 2D 结构化网格 SIMPLE 求解器与神经网络双残差误差补偿框架
================================================================================

【功能特性】
 1. 自动解析 Gmsh 2.2 (.msh) 网格文件，将非结构节点与单元重构为 2D 结构化网格 (i, j)
 2. 通过配置字典 (Config Dict / JSON) 灵活指定任意边界条件:
    - 'wall'            : 固壁/运动壁面 (u=u_w, v=v_w, ∂p/∂n=0)
    - 'inlet'           : 速度入口 (u=u_in, v=v_in, ∂p/∂n=0)
    - 'pressure_outlet' : 压力出口 (∂u/∂n=0, ∂v/∂n=0, p=p_out, p'=0)
 3. 统一采用向量化五对角稀疏矩阵直接求解器 (spsolve)，兼容任意边界组合
 4. 完备的动量 (Res_u, Res_v) + Rhie-Chow 连续性 (Res_div) 双残差计算与 NN 代理模型修正

================================================================================
【🚨 网格支持与限制说明 (外部导入必看) 🚨】
当前求解器针对正交均匀网格进行了高度优化，以换取极致的计算速度，请遵守以下限制：

1. 网格均匀性限制：
   - 当前离散算子设定全局恒定的 dx 和 dy，**暂不支持非均匀网格**（如边界层局部渐变加密）。
   - 导入的网格在 X 和 Y 方向上的节点间距必须保持严格等距。

2. 外部 Gmsh 网格绘制规范：
   若改用 Gmsh 软件自定义绘制网格并导入，请务必严格满足以下 4 个条件，否则解析器将报错：
   - [格式要求] 导出时必须选择 "Gmsh 2.2 ASCII" 格式（切勿使用默认的 v4.1 格式）。
   - [拓扑要求] 必须是完全正交的结构化四边形网格。需使用 Transfinite 功能，并务必勾选 Recombine 将三角形强制合并为四边形。
   - [分布要求] 节点分布必须是绝对均匀的（Transfinite Curve 设置中的 Progression 必须为 1）。
   - [物理组命名] 必须在 Gmsh 中正确设置物理组 (Physical Groups)：
     * 需为内部流体面创建 2D 物理组（如命名为 "fluid"）。
     * 需为四条边界线段分别创建 1D 物理组（如 "inlet", "outlet", "top_wall", "bottom_wall"）。
     * ⚠️ Gmsh 中定义的 1D 边界名称，必须与 Python 代码 config 字典中 `boundary_conditions` 下配置的键名完全一致！
================================================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RectBivariateSpline

# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# 模块 1: Gmsh (.msh) 网格生成与结构化重构解析器
# ============================================================
def export_rect_gmsh(filename, nx, ny, Lx, Ly):
    """
    生成与 Pointwise 导出格式完全兼容的 Gmsh 2.2 ASCII 矩形四边网格 (.msh)
    用于在没有外部密网格文件时自动生成测试网格。
    """
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        # 1. 写入节点
        f.write("$Nodes\n")
        f.write(f"{nx * ny}\n")
        for j in range(ny):
            for i in range(nx):
                node_id = j * nx + i + 1
                f.write(f"{node_id:4d} {x[i]:.10g} {y[j]:.10g} 0\n")
        f.write("$EndNodes\n")

        # 2. 写入单元 (2D Quad=3, 1D Line=1)
        num_quads = (nx - 1) * (ny - 1)
        num_lines = 2 * (nx - 1) + 2 * (ny - 1)
        f.write("$Elements\n")
        f.write(f"{num_quads + num_lines}\n")
        elem_id = 1
        # 内部流体四边形单元 (PhysGrp = 1 "fluid")
        for j in range(ny - 1):
            for i in range(nx - 1):
                n1 = j * nx + i + 1
                n2 = n1 + 1
                n3 = n2 + nx
                n4 = n1 + nx
                f.write(f"{elem_id:4d}  3 2 1 1 {n1:4d} {n2:4d} {n3:4d} {n4:4d}\n")
                elem_id += 1
        # bottom (PhysGrp = 2)
        for i in range(nx - 1):
            n1, n2 = i + 1, i + 2
            f.write(f"{elem_id:4d}  1 2 2 2 {n1:4d} {n2:4d}\n")
            elem_id += 1
        # left (PhysGrp = 3)
        for j in range(ny - 1):
            n1, n2 = (j + 1) * nx + 1, j * nx + 1
            f.write(f"{elem_id:4d}  1 2 3 3 {n1:4d} {n2:4d}\n")
            elem_id += 1
        # right (PhysGrp = 4)
        for j in range(ny - 1):
            n1, n2 = j * nx + nx, (j + 1) * nx + nx
            f.write(f"{elem_id:4d}  1 2 4 4 {n1:4d} {n2:4d}\n")
            elem_id += 1
        # top (PhysGrp = 5)
        for i in range(nx - 1):
            n1, n2 = (ny - 1) * nx + i + 2, (ny - 1) * nx + i + 1
            f.write(f"{elem_id:4d}  1 2 5 5 {n1:4d} {n2:4d}\n")
            elem_id += 1
        f.write("$EndElements\n")

        # 3. 写入物理组名称
        f.write("$PhysicalNames\n5\n")
        f.write('2 1 "fluid"\n1 2 "bottom"\n1 3 "left"\n1 4 "right"\n1 5 "top"\n')
        f.write("$EndPhysicalNames\n")


class StructuredGmshMesh:
    """
    读取 Gmsh 2.2 (.msh) 网格文件，并将非结构化节点/单元列表转换为 2D 结构化网格拓扑。
    """
    def __init__(self, msh_filepath, tol_decimals=6):
        self.filepath = msh_filepath
        self.tol_decimals = tol_decimals

        self.raw_nodes = {}          # node_id -> (x, y, z)
        self.phys_names = {}         # (dim, tag) -> name_str
        self.boundary_nodes = {}     # phys_name -> set of node_ids
        self.quad_elements = []      # list of (n1, n2, n3, n4)

        self._parse_msh()
        self._build_structured_topology()

    def _parse_msh(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        # ===== 第一遍扫描：优先提取 $PhysicalNames（防止其出现在文件末尾）=====
        idx = 0
        n_lines = len(lines)
        while idx < n_lines:
            if lines[idx] == "$PhysicalNames":
                idx += 1
                num_names = int(lines[idx])
                for _ in range(num_names):
                    idx += 1
                    parts = lines[idx].split()
                    dim, tag = int(parts[0]), int(parts[1])
                    name = parts[2].strip('"')
                    self.phys_names[(dim, tag)] = name
                    if dim == 1:
                        self.boundary_nodes[name] = set()
                break
            idx += 1

        # 建立 1D 物理组 tag -> 物理名称 的查找表
        tag_to_bname = {tag: name for (dim, tag), name in self.phys_names.items() if dim == 1}

        # ===== 第二遍扫描：解析 $Nodes 和 $Elements =====
        idx = 0
        while idx < n_lines:
            line = lines[idx]
            if line == "$Comments":
                while idx < n_lines and lines[idx] != "$EndComments":
                    idx += 1
            elif line == "$Nodes":
                idx += 1
                num_nodes = int(lines[idx])
                for _ in range(num_nodes):
                    idx += 1
                    parts = lines[idx].split()
                    nid = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    self.raw_nodes[nid] = (x, y, z)
            elif line == "$Elements":
                idx += 1
                num_elems = int(lines[idx])
                for _ in range(num_elems):
                    idx += 1
                    parts = [int(p) for p in lines[idx].split()]
                    elm_type = parts[1]
                    num_tags = parts[2]
                    phys_tag = parts[3]
                    node_list = parts[3 + num_tags:]
                    if elm_type == 1:  # 1D 边界线段单元
                        bname = tag_to_bname.get(phys_tag, f"tag_{phys_tag}")
                        self.boundary_nodes.setdefault(bname, set()).update(node_list)
                    elif elm_type == 3:  # 2D 四边形单元
                        self.quad_elements.append(node_list)
            idx += 1

    def _build_structured_topology(self):
        # 1. 提取所有节点坐标并消除浮点截断噪音 (如 0.6000000000000001 -> 0.6)
        nids = np.array(list(self.raw_nodes.keys()), dtype=int)
        coords = np.array([self.raw_nodes[nid][:2] for nid in nids], dtype=float)
        rounded_coords = np.round(coords, self.tol_decimals)

        self.x = np.unique(rounded_coords[:, 0])
        self.y = np.unique(rounded_coords[:, 1])
        self.nx = len(self.x)
        self.ny = len(self.y)

        if self.nx * self.ny != len(nids):
            raise ValueError(
                f"网格非标准正交结构化网格: 唯一坐标乘积 ({self.nx}x{self.ny}={self.nx*self.ny}) "
                f"!= 节点总数 ({len(nids)})"
            )

        self.Lx = float(self.x[-1] - self.x[0])
        self.Ly = float(self.y[-1] - self.y[0])
        self.dx = self.Lx / (self.nx - 1)
        self.dy = self.Ly / (self.ny - 1)

        # 检查网格均匀性
        if not (np.allclose(np.diff(self.x), self.dx, atol=1e-5) and
                np.allclose(np.diff(self.y), self.dy, atol=1e-5)):
            raise ValueError("当前离散算子要求正交均匀网格，检测到非均匀网格步长！")

        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # 2. 建立 Gmsh node_id <-> 结构化索引 (i, j) 的双向映射
        ix_all = np.searchsorted(self.x, rounded_coords[:, 0])
        iy_all = np.searchsorted(self.y, rounded_coords[:, 1])

        self.node_to_ij = {}
        self.ij_to_node = np.zeros((self.nx, self.ny), dtype=int)
        for nid, i, j in zip(nids, ix_all, iy_all):
            self.node_to_ij[nid] = (int(i), int(j))
            self.ij_to_node[i, j] = nid

        # 3. 识别四条结构化边界 ('west', 'east', 'south', 'north') 对应的 Gmsh 物理名称
        side_node_sets = {
            'west':  set(self.ij_to_node[0, :].tolist()),
            'east':  set(self.ij_to_node[-1, :].tolist()),
            'south': set(self.ij_to_node[:, 0].tolist()),
            'north': set(self.ij_to_node[:, -1].tolist()),
        }

        self.side_to_phys_name = {}
        self.phys_name_to_side = {}
        for side, s_nodes in side_node_sets.items():
            for bname, b_nodes in self.boundary_nodes.items():
                # 若 Gmsh 物理边界节点集合与某条结构化边完全重合 (或包含关系)
                if len(b_nodes) > 1 and b_nodes.issubset(s_nodes):
                    self.side_to_phys_name[side] = bname
                    self.phys_name_to_side[bname] = side
                    break
            if side not in self.side_to_phys_name:
                raise RuntimeError(f"未能在 .msh 文件中找到对应结构化边 '{side}' 的 1D Physical Group!")


# ============================================================
# 模块 2: 五对角稀疏矩阵向量化组装器
# ============================================================
def build_penta_matrix(aP, aE, aW, aN, aS):
    nix, niy = aP.shape
    n = nix * niy
    ii = np.arange(nix)[:, None] * niy + np.arange(niy)[None, :]
    rows_list, cols_list, vals_list = [], [], []

    # 主对角元 (P)
    rows_list.append(ii.ravel())
    cols_list.append(ii.ravel())
    vals_list.append(aP.ravel())

    # 东邻居 (E: i+1, j)
    rows_list.append(ii[:-1, :].ravel())
    cols_list.append((ii[:-1, :] + niy).ravel())
    vals_list.append(-aE[:-1, :].ravel())

    # 西邻居 (W: i-1, j)
    rows_list.append(ii[1:, :].ravel())
    cols_list.append((ii[1:, :] - niy).ravel())
    vals_list.append(-aW[1:, :].ravel())

    # 北邻居 (N: i, j+1)
    rows_list.append(ii[:, :-1].ravel())
    cols_list.append((ii[:, :-1] + 1).ravel())
    vals_list.append(-aN[:, :-1].ravel())

    # 南邻居 (S: i, j-1)
    rows_list.append(ii[:, 1:].ravel())
    cols_list.append((ii[:, 1:] - 1).ravel())
    vals_list.append(-aS[:, 1:].ravel())

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)

    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


# ============================================================
# 模块 3: 通用配置驱动 SIMPLE 求解器
# ============================================================
class UniversalSIMPLE:
    """
    基于 StructuredGmshMesh 与边界配置字典的通用 2D 不可压 SIMPLE 求解器。
    支持注入神经网络预测的双残差补偿源项 (Sp_nn_u, Sp_nn_v, Sp_nn_div)。
    """
    def __init__(self, mesh: StructuredGmshMesh, config: dict):
        self.mesh = mesh
        self.config = config
        self.nx, self.ny = mesh.nx, mesh.ny
        self.dx, self.dy = mesh.dx, mesh.dy
        self.Lx, self.Ly = mesh.Lx, mesh.Ly
        self.x, self.y = mesh.x, mesh.y
        self.X, self.Y = mesh.X, mesh.Y

        fluid_cfg = config.get('fluid', {})
        self.rho = float(fluid_cfg.get('rho', 1.0))
        self.mu = float(fluid_cfg.get('mu', 0.01))

        # 解析四条结构化边 ('west', 'east', 'south', 'north') 的边界配置
        raw_bc_cfg = config.get('boundary_conditions', {})
        self.side_bc = {}
        for side in ['west', 'east', 'south', 'north']:
            phys_name = mesh.side_to_phys_name[side]
            if phys_name in raw_bc_cfg:
                self.side_bc[side] = raw_bc_cfg[phys_name]
            elif side in raw_bc_cfg:
                self.side_bc[side] = raw_bc_cfg[side]
            else:
                raise KeyError(f"配置文件中缺少边界 '{phys_name}' (对应方位 '{side}') 的定义!")

        # 检查是否存在压力出口 (决定压力场是否有 Dirichlet 锚点)
        self.has_pressure_outlet = any(
            bc['type'] == 'pressure_outlet' for bc in self.side_bc.values()
        )

        sh = (self.nx, self.ny)
        self.u = np.zeros(sh)
        self.v = np.zeros(sh)
        self.p = np.zeros(sh)

        self.aE = np.zeros(sh); self.aW = np.zeros(sh)
        self.aN = np.zeros(sh); self.aS = np.zeros(sh)
        self.aP = np.zeros(sh); self.Sp = np.zeros(sh)

        self.sl = (slice(1, -1), slice(1, -1))

        # 神经网络双残差补偿源项
        self.Sp_nn_u = None
        self.Sp_nn_v = None
        self.Sp_nn_div = None

        self._init_fields()

    def _init_fields(self):
        init_cfg = self.config.get('initial_conditions', {'u': 0.0, 'v': 0.0, 'p': 0.0})
        self.u[:, :] = float(init_cfg.get('u', 0.0))
        self.v[:, :] = float(init_cfg.get('v', 0.0))
        self.p[:, :] = float(init_cfg.get('p', 0.0))
        self._bc_uv()
        self._bc_p()

    def _apply_side_dirichlet_uv(self, side, bc):
        u_val = float(bc.get('u', 0.0))
        v_val = float(bc.get('v', 0.0))
        if side == 'west':
            self.u[0, :] = u_val; self.v[0, :] = v_val
        elif side == 'east':
            self.u[-1, :] = u_val; self.v[-1, :] = v_val
        elif side == 'south':
            self.u[:, 0] = u_val; self.v[:, 0] = v_val
        elif side == 'north':
            self.u[:, -1] = u_val; self.v[:, -1] = v_val

    def _bc_uv(self):
        """
        按优先级施加的速度边界条件:
          1. 先处理 pressure_outlet (Neumann 零梯度导出)
          2. 再处理 inlet (固定速度入口)
          3. 最后处理 wall (无滑移/运动滑移壁面，确保角点处固壁条件优先)
        """
        # Step 1: 压力出口零梯度
        for side, bc in self.side_bc.items():
            if bc['type'] == 'pressure_outlet':
                if side == 'west':
                    self.u[0, 1:-1] = self.u[1, 1:-1]; self.v[0, 1:-1] = self.v[1, 1:-1]
                elif side == 'east':
                    self.u[-1, 1:-1] = self.u[-2, 1:-1]; self.v[-1, 1:-1] = self.v[-2, 1:-1]
                elif side == 'south':
                    self.u[1:-1, 0] = self.u[1:-1, 1]; self.v[1:-1, 0] = self.v[1:-1, 1]
                elif side == 'north':
                    self.u[1:-1, -1] = self.u[1:-1, -2]; self.v[1:-1, -1] = self.v[1:-1, -2]

        # Step 2: 速度入口 (Dirichlet)
        for side, bc in self.side_bc.items():
            if bc['type'] == 'inlet':
                self._apply_side_dirichlet_uv(side, bc)

        # Step 3: 固壁边界 (Dirichlet，覆盖角点以保证物理自洽)
        for side, bc in self.side_bc.items():
            if bc['type'] == 'wall':
                self._apply_side_dirichlet_uv(side, bc)

    def _bc_p(self):
        if not self.has_pressure_outlet:
            # 全封闭域/无压力出口：锚定内部节点 (1,1) 为压力参考零点
            self.p -= self.p[1, 1]

        # 先施加 Neumann 零梯度边界 ('wall' 或 'inlet')
        for side, bc in self.side_bc.items():
            if bc['type'] in ('wall', 'inlet'):
                if side == 'west':    self.p[0, :] = self.p[1, :]
                elif side == 'east':  self.p[-1, :] = self.p[-2, :]
                elif side == 'south': self.p[:, 0] = self.p[:, 1]
                elif side == 'north': self.p[:, -1] = self.p[:, -2]

        # 后施加 Dirichlet 压力出口 ('pressure_outlet')
        for side, bc in self.side_bc.items():
            if bc['type'] == 'pressure_outlet':
                p_val = float(bc.get('p', 0.0))
                if side == 'west':    self.p[0, :] = p_val
                elif side == 'east':  self.p[-1, :] = p_val
                elif side == 'south': self.p[:, 0] = p_val
                elif side == 'north': self.p[:, -1] = p_val

    def _bc_p_prime(self, pp):
        if not self.has_pressure_outlet:
            pp -= pp[1, 1]

        for side, bc in self.side_bc.items():
            if bc['type'] in ('wall', 'inlet'):
                if side == 'west':    pp[0, :] = pp[1, :]
                elif side == 'east':  pp[-1, :] = pp[-2, :]
                elif side == 'south': pp[:, 0] = pp[:, 1]
                elif side == 'north': pp[:, -1] = pp[:, -2]

        for side, bc in self.side_bc.items():
            if bc['type'] == 'pressure_outlet':
                if side == 'west':    pp[0, :] = 0.0
                elif side == 'east':  pp[-1, :] = 0.0
                elif side == 'south': pp[:, 0] = 0.0
                elif side == 'north': pp[:, -1] = 0.0

    @staticmethod
    def _gradient(f, dx, dy):
        Gx = np.zeros_like(f)
        Gy = np.zeros_like(f)
        Gx[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dx)
        Gy[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dy)
        Gx[0, :]  = (-3 * f[0, :]  + 4 * f[1, :]  - f[2, :])  / (2 * dx)
        Gx[-1, :] = ( 3 * f[-1, :] - 4 * f[-2, :] + f[-3, :]) / (2 * dx)
        Gy[:, 0]  = (-3 * f[:, 0]  + 4 * f[:, 1]  - f[:, 2])  / (2 * dy)
        Gy[:, -1] = ( 3 * f[:, -1] - 4 * f[:, -2] + f[:, -3]) / (2 * dy)
        return Gx, Gy

    def _assemble_momentum(self, direction):
        dx, dy = self.dx, self.dy
        rho, mu = self.rho, self.mu
        sl = self.sl

        ui = self.u[sl]
        vi = self.v[sl]

        De = Dw = mu / dx ** 2
        Dn = Ds = mu / dy ** 2

        # 一阶迎风对流格式
        aE_c = np.where(ui >= 0, 0.0, -rho * ui / dx)
        aW_c = np.where(ui >= 0,  rho * ui / dx, 0.0)
        aN_c = np.where(vi >= 0, 0.0, -rho * vi / dy)
        aS_c = np.where(vi >= 0,  rho * vi / dy, 0.0)

        self.aE[sl] = De + aE_c
        self.aW[sl] = Dw + aW_c
        self.aN[sl] = Dn + aN_c
        self.aS[sl] = Ds + aS_c
        self.aP[sl] = self.aE[sl] + self.aW[sl] + self.aN[sl] + self.aS[sl]

        Gx, Gy = self._gradient(self.p, dx, dy)
        if direction == 'u':
            self.Sp[sl] = -Gx[sl]
            if self.Sp_nn_u is not None:
                self.Sp[sl] += self.Sp_nn_u[sl]
        else:
            self.Sp[sl] = -Gy[sl]
            if self.Sp_nn_v is not None:
                self.Sp[sl] += self.Sp_nn_v[sl]

    def _solve_scalar(self, phi, alpha):
        sl = self.sl
        nx, ny = self.nx, self.ny
        nix, niy = nx - 2, ny - 2

        phi_old = phi.copy()
        aP_ur = (self.aP[sl] / alpha).copy()
        aE_in = self.aE[sl].copy()
        aW_in = self.aW[sl].copy()
        aN_in = self.aN[sl].copy()
        aS_in = self.aS[sl].copy()

        S_ur = self.Sp[sl] + (1.0 - alpha) / alpha * self.aP[sl] * phi_old[sl]
        rhs = S_ur.reshape((nix, niy)).copy()

        # 根据四条边界的物理类型自动处理动量方程矩阵边界项:
        # - Dirichlet ('wall', 'inlet'): 移至右端项 rhs += a_nb * phi_bc
        # - Neumann ('pressure_outlet'): 零梯度隐式折减主对角元 aP_ur -= a_nb
        if self.side_bc['west']['type'] == 'pressure_outlet':
            aP_ur[0, :] -= aW_in[0, :]
        else:
            rhs[0, :] += aW_in[0, :] * phi[0, 1:-1]

        if self.side_bc['east']['type'] == 'pressure_outlet':
            aP_ur[-1, :] -= aE_in[-1, :]
        else:
            rhs[-1, :] += aE_in[-1, :] * phi[nx - 1, 1:-1]

        if self.side_bc['south']['type'] == 'pressure_outlet':
            aP_ur[:, 0] -= aS_in[:, 0]
        else:
            rhs[:, 0] += aS_in[:, 0] * phi[1:-1, 0]

        if self.side_bc['north']['type'] == 'pressure_outlet':
            aP_ur[:, -1] -= aN_in[:, -1]
        else:
            rhs[:, -1] += aN_in[:, -1] * phi[1:-1, ny - 1]

        A = build_penta_matrix(aP_ur, aE_in, aW_in, aN_in, aS_in)
        phi_internal = spsolve(A, rhs.ravel())
        phi[sl] = phi_internal.reshape((nix, niy), order='C')

    def _rc_face_velocities(self, alpha_u):
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        d = np.zeros((nx, ny))
        # 包含欠松弛因子的一致性 Rhie-Chow 动量插值系数 d = alpha_u / aP
        d[1:-1, 1:-1] = alpha_u / self.aP[1:-1, 1:-1]
        d[0, :]  = d[1, :]
        d[-1, :] = d[-2, :]
        d[:, 0]  = d[:, 1]
        d[:, -1] = d[:, -2]

        Gx, Gy = self._gradient(self.p, dx, dy)

        d_bar_e = 0.5 * (d[:-1, :] + d[1:, :])
        Gx_bar_e = 0.5 * (Gx[:-1, :] + Gx[1:, :])
        Gx_compact_e = (self.p[1:, :] - self.p[:-1, :]) / dx
        u_e = 0.5 * (self.u[:-1, :] + self.u[1:, :]) + d_bar_e * (Gx_bar_e - Gx_compact_e)

        d_bar_n = 0.5 * (d[:, :-1] + d[:, 1:])
        Gy_bar_n = 0.5 * (Gy[:, :-1] + Gy[:, 1:])
        Gy_compact_n = (self.p[:, 1:] - self.p[:, :-1]) / dy
        v_n = 0.5 * (self.v[:, :-1] + self.v[:, 1:]) + d_bar_n * (Gy_bar_n - Gy_compact_n)

        return u_e, v_n, d

    def _solve_pressure_correction(self, alpha_u):
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        nix, niy = nx - 2, ny - 2
        sl = self.sl

        u_e, v_n, d = self._rc_face_velocities(alpha_u)

        div = np.zeros((nx, ny))
        div[sl] = (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) / dx + \
                  (v_n[1:-1, 1:] - v_n[1:-1, :-1]) / dy

        # 注入 NN 预测的连续性方程散度残差 Sp_nn_div
        if self.Sp_nn_div is not None:
            b = -(div[sl] - self.Sp_nn_div[sl]).copy()
        else:
            b = -div[sl].copy()

        mass_resid = float(np.max(np.abs(b)))

        aE_pc = 0.5 * (d[1:-1, 1:-1] + d[2:, 1:-1])  / dx ** 2
        aW_pc = 0.5 * (d[:-2, 1:-1]  + d[1:-1, 1:-1]) / dx ** 2
        aN_pc = 0.5 * (d[1:-1, 1:-1] + d[1:-1, 2:])  / dy ** 2
        aS_pc = 0.5 * (d[1:-1, :-2]  + d[1:-1, 1:-1]) / dy ** 2
        aP_pc = aE_pc + aW_pc + aN_pc + aS_pc

        # 根据边界条件处理压力修正方程系数:
        # - 'wall' / 'inlet': 压力零梯度 Neumann (∂p'/∂n = 0) -> 主对角元减去越界邻居系数
        # - 'pressure_outlet': 定压边界 Dirichlet (p' = 0) -> 主对角元保持不变，越界项自然为 0
        if self.side_bc['west']['type'] in ('wall', 'inlet'):
            aP_pc[0, :] -= aW_pc[0, :]
        if self.side_bc['east']['type'] in ('wall', 'inlet'):
            aP_pc[-1, :] -= aE_pc[-1, :]
        if self.side_bc['south']['type'] in ('wall', 'inlet'):
            aP_pc[:, 0] -= aS_pc[:, 0]
        if self.side_bc['north']['type'] in ('wall', 'inlet'):
            aP_pc[:, -1] -= aN_pc[:, -1]

        # 若全域无压力出口 (纯 Neumann 泊松问题)，固定内部节点 (0,0) 即全局 (1,1) 的 p'=0
        if not self.has_pressure_outlet:
            aE_pc[0, 0] = 0.0
            aW_pc[0, 0] = 0.0
            aN_pc[0, 0] = 0.0
            aS_pc[0, 0] = 0.0
            b[0, 0] = 0.0

        A_p = build_penta_matrix(aP_pc, aE_pc, aW_pc, aN_pc, aS_pc)
        pp_internal = spsolve(A_p, b.ravel())

        pp = np.zeros((nx, ny))
        pp[sl] = pp_internal.reshape((nix, niy), order='C')
        self._bc_p_prime(pp)

        return pp, d, mass_resid

    def _correct_fields(self, pp, d, alpha_p):
        dx, dy = self.dx, self.dy
        sl = self.sl

        self.p[sl] += alpha_p * pp[sl]
        Gx_pp, Gy_pp = self._gradient(pp, dx, dy)
        self.u[sl] -= d[sl] * Gx_pp[sl]
        self.v[sl] -= d[sl] * Gy_pp[sl]

        self._bc_uv()
        self._bc_p()

    def _rel_change(self, phi_new, phi_old):
        denom = np.max(np.abs(phi_old))
        if denom < 1e-30:
            return float(np.mean(np.abs(phi_new - phi_old)))
        return float(np.mean(np.abs(phi_new - phi_old)) / denom)

    def solve(self, verbose=True):
        s_cfg = self.config.get('solver', {})
        max_iter = int(s_cfg.get('max_iter', 3000))
        tol      = float(s_cfg.get('tol', 1e-6))
        alpha_u  = float(s_cfg.get('alpha_u', 0.65))
        alpha_v  = float(s_cfg.get('alpha_v', 0.65))
        alpha_p  = float(s_cfg.get('alpha_p', 0.25))

        if verbose:
            print("=" * 68)
            print(f"  通用 SIMPLE 求解器 | 案例: {self.config.get('case_name', 'Unnamed')} "
                  f"| 网格: {self.nx} x {self.ny}")
            if self.Sp_nn_u is not None:
                print("  ✓ 已启用神经网络双残差源项补偿 (Res_u + Res_v + Res_div)")
            print("=" * 68)

        u_old, v_old = self.u.copy(), self.v.copy()

        for it in range(1, max_iter + 1):
            self._assemble_momentum('u')
            self._solve_scalar(self.u, alpha_u)
            self._bc_uv()

            self._assemble_momentum('v')
            self._solve_scalar(self.v, alpha_v)
            self._bc_uv()

            pp, d, mass_resid = self._solve_pressure_correction(alpha_u)
            self._correct_fields(pp, d, alpha_p)

            res_u = self._rel_change(self.u, u_old)
            res_v = self._rel_change(self.v, v_old)
            u_old, v_old = self.u.copy(), self.v.copy()

            if verbose and (it % 200 == 0 or it == 1):
                u_max = np.max(np.abs(self.u))
                print(f"  iter {it:5d} | res_u={res_u:.2e} | res_v={res_v:.2e} | "
                      f"div_err={mass_resid:.2e} | u_max={u_max:.4f}")

            if res_u < tol and res_v < tol and mass_resid < tol and it > 15:
                if verbose:
                    print(f"  >> 收敛! iter = {it}, res_u={res_u:.2e}, div_err={mass_resid:.2e}")
                break
        else:
            if verbose:
                print(f"  >> 达到最大迭代次数 {max_iter}, res_u={res_u:.2e}, div_err={mass_resid:.2e}")

        return res_u, res_v, mass_resid


# ============================================================
# 模块 4: 完备离散双残差计算器 (与通用边界条件完全自洽)
# ============================================================
def compute_discrete_residual(solver: UniversalSIMPLE, u_s, v_s, p_s):
    """
    将密网格采样流场 (u_s, v_s, p_s) 载入粗网格求解器模板，
    严格同步粗网格边界条件后计算自洽的动量截断残差 (Res_u, Res_v) 与连续性散度残差 (Res_div)。
    """
    alpha_u = float(solver.config.get('solver', {}).get('alpha_u', 0.65))

    solver.u = u_s.copy()
    solver.v = v_s.copy()
    solver.p = p_s.copy()
    solver._bc_uv()
    solver._bc_p()

    sl = solver.sl

    # 1. 组装粗网格动量系数
    solver._assemble_momentum('u')
    Gx, Gy = solver._gradient(solver.p, solver.dx, solver.dy)
    Sp_u = -Gx[sl]
    Sp_v = -Gy[sl]

    # 2. 计算动量离散残差: Res = aP*phi_P - sum(a_nb*phi_nb) - Sp
    # 注意: 由于 solver._bc_uv() 已经把 pressure_outlet 边界设为等于内部相邻节点值，
    # 因此 a_nb * phi_boundary 自动等价于 _solve_scalar 中主对角线折减 (aP - a_nb)*phi_P!
    Res_u = (solver.aP[sl] * solver.u[sl]
             - solver.aE[sl] * solver.u[2:, 1:-1]
             - solver.aW[sl] * solver.u[:-2, 1:-1]
             - solver.aN[sl] * solver.u[1:-1, 2:]
             - solver.aS[sl] * solver.u[1:-1, :-2]
             - Sp_u)

    Res_v = (solver.aP[sl] * solver.v[sl]
             - solver.aE[sl] * solver.v[2:, 1:-1]
             - solver.aW[sl] * solver.v[:-2, 1:-1]
             - solver.aN[sl] * solver.v[1:-1, 2:]
             - solver.aS[sl] * solver.v[1:-1, :-2]
             - Sp_v)

    # 3. 计算采样场在粗网格 Rhie-Chow 界面速度下的离散散度残差
    u_e, v_n, _ = solver._rc_face_velocities(alpha_u)
    Res_div = (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) / solver.dx + \
              (v_n[1:-1, 1:] - v_n[1:-1, :-1]) / solver.dy

    return Res_u, Res_v, Res_div, solver.u.copy(), solver.v.copy(), solver.p.copy()


# ============================================================
# 模块 5: 多尺度傅里叶特征神经网络误差补偿器
# ============================================================
class FourierMLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=96, num_freqs=6):
        super().__init__()
        freqs = torch.arange(1, num_freqs + 1, dtype=torch.float32) * np.pi
        self.register_buffer('freqs', freqs)
        feat_dim = in_dim + in_dim * num_freqs * 2

        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        proj = x.unsqueeze(-1) * self.freqs
        proj = proj.reshape(x.shape[0], -1)
        feats = torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(feats)


class UniversalNeuralCorrector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        self.stats = {}

    def prepare_training_data(self, solver: UniversalSIMPLE, res_u, res_v, res_div):
        # 归一化内部节点坐标至 [0, 1] x [0, 1]
        X_in = (solver.X[1:-1, 1:-1].ravel() - solver.x[0]) / solver.Lx
        Y_in = (solver.Y[1:-1, 1:-1].ravel() - solver.y[0]) / solver.Ly
        X_train = torch.tensor(np.column_stack([X_in, Y_in]), dtype=torch.float32, device=self.device)

        targets = {'u': res_u.ravel(), 'v': res_v.ravel(), 'div': res_div.ravel()}
        Y_tensors = {}
        for k, val in targets.items():
            mean = float(np.mean(val))
            std = float(np.std(val)) + 1e-12
            self.stats[k] = (mean, std)
            norm_val = (val - mean) / std
            Y_tensors[k] = torch.tensor(norm_val, dtype=torch.float32, device=self.device).unsqueeze(1)

        return X_train, Y_tensors

    def train(self, X_train, Y_tensors, nn_cfg: dict):
        epochs      = int(nn_cfg.get('epochs', 3500))
        lr          = float(nn_cfg.get('lr', 3e-3))
        hidden_dim  = int(nn_cfg.get('hidden_dim', 96))
        num_freqs   = int(nn_cfg.get('num_freqs', 6))

        print(f"\n{'='*68}")
        print(f"  训练 FourierMLP 代理模型 | 样本数: {len(X_train)} | 设备: {self.device}")
        print(f"{'='*68}")

        criterion = nn.MSELoss()
        optimizers, schedulers = {}, {}

        for k in ['u', 'v', 'div']:
            self.models[k] = FourierMLP(2, hidden_dim, num_freqs).to(self.device)
            optimizers[k] = torch.optim.Adam(self.models[k].parameters(), lr=lr, weight_decay=1e-7)
            schedulers[k] = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers[k], T_max=epochs, eta_min=1e-6
            )

        for epoch in range(1, epochs + 1):
            losses = {}
            for k in ['u', 'v', 'div']:
                optimizers[k].zero_grad()
                pred = self.models[k](X_train)
                loss = criterion(pred, Y_tensors[k])
                loss.backward()
                optimizers[k].step()
                schedulers[k].step()
                losses[k] = loss.item()

            if epoch % 500 == 0 or epoch == 1:
                cur_lr = optimizers['u'].param_groups[0]['lr']
                print(f"  Epoch {epoch:4d} | Norm_MSE_u={losses['u']:.3e} | "
                      f"Norm_MSE_v={losses['v']:.3e} | Norm_MSE_div={losses['div']:.3e} | LR={cur_lr:.2e}")

        print("  ✓ 神经网络代理模型训练完成")

    def predict(self, solver: UniversalSIMPLE):
        nx, ny = solver.nx, solver.ny
        X_all = np.column_stack([
            (solver.X.ravel() - solver.x[0]) / solver.Lx,
            (solver.Y.ravel() - solver.y[0]) / solver.Ly
        ])
        X_tensor = torch.tensor(X_all, dtype=torch.float32, device=self.device)

        preds = {}
        with torch.no_grad():
            for k in ['u', 'v', 'div']:
                mean, std = self.stats[k]
                raw = self.models[k](X_tensor).cpu().numpy().reshape(nx, ny)
                field = raw * std + mean
                field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = 0.0
                preds[k] = field

        # 若为无压力出口的封闭域，强制内部散度残差零均值以保证泊松方程可解性
        if not solver.has_pressure_outlet:
            sl = solver.sl
            preds['div'][sl] -= np.mean(preds['div'][sl])

        return preds['u'], preds['v'], preds['div']


# ============================================================
# 模块 6: 通用全流程驱动管线 (Pipeline Runner)
# ============================================================
def run_universal_pipeline(config: dict, plot_results=True):
    case_name = config.get('case_name', 'flow_case')
    dense_msh_path  = config['meshes']['dense_msh']
    coarse_msh_path = config['meshes']['coarse_msh']

    # 1. 读取并重构粗/密 Gmsh 结构化网格
    print(f"\n>>> 正在读取并转换 Gmsh 网格文件...")
    mesh_dense  = StructuredGmshMesh(dense_msh_path)
    mesh_coarse = StructuredGmshMesh(coarse_msh_path)
    print(f"  密网格 ({dense_msh_path})  -> 结构化维度: {mesh_dense.nx} x {mesh_dense.ny}, "
          f"物理域: [{mesh_dense.Lx} x {mesh_dense.Ly}]")
    print(f"  粗网格 ({coarse_msh_path}) -> 结构化维度: {mesh_coarse.nx} x {mesh_coarse.ny}, "
          f"边界映射: {mesh_coarse.side_to_phys_name}")

    # 2. 密网格求解 (高精度参考解)
    solver_dense = UniversalSIMPLE(mesh_dense, config)
    solver_dense.solve(verbose=True)

    # 3. 在粗网格节点上采样密网格流场
    interp_u = RectBivariateSpline(mesh_dense.x, mesh_dense.y, solver_dense.u)
    interp_v = RectBivariateSpline(mesh_dense.x, mesh_dense.y, solver_dense.v)
    interp_p = RectBivariateSpline(mesh_dense.x, mesh_dense.y, solver_dense.p)

    solver_template = UniversalSIMPLE(mesh_coarse, config)
    u_raw = interp_u(mesh_coarse.x, mesh_coarse.y)
    v_raw = interp_v(mesh_coarse.x, mesh_coarse.y)
    p_raw = interp_p(mesh_coarse.x, mesh_coarse.y)

    # 4. 计算粗网格自洽双残差 (Res_u, Res_v, Res_div)
    res_u, res_v, res_div, u_sampled, v_sampled, p_sampled = compute_discrete_residual(
        solver_template, u_raw, v_raw, p_raw
    )
    print(f"\n  [粗网格离散残差统计]")
    print(f"  Res_u   范围: [{res_u.min():.4e}, {res_u.max():.4e}]")
    print(f"  Res_v   范围: [{res_v.min():.4e}, {res_v.max():.4e}]")
    print(f"  Res_div 范围: [{res_div.min():.4e}, {res_div.max():.4e}]")

    # 5. 训练神经网络代理模型并预测源项
    corrector = UniversalNeuralCorrector()
    X_train, Y_tensors = corrector.prepare_training_data(solver_template, res_u, res_v, res_div)
    corrector.train(X_train, Y_tensors, config.get('nn_config', {}))
    res_u_pred, res_v_pred, res_div_pred = corrector.predict(solver_template)

    # 6. 粗网格无修正基准求解
    print("\n>>> 运行粗网格基准求解 (Original Coarse)...")
    solver_coarse_orig = UniversalSIMPLE(mesh_coarse, config)
    solver_coarse_orig.solve(verbose=True)

    # 7. 粗网格注入 NN 源项修正求解
    print("\n>>> 运行粗网格神经网络补偿求解 (NN-Corrected Coarse)...")
    solver_coarse_nn = UniversalSIMPLE(mesh_coarse, config)
    solver_coarse_nn.Sp_nn_u   = res_u_pred
    solver_coarse_nn.Sp_nn_v   = res_v_pred
    solver_coarse_nn.Sp_nn_div = res_div_pred
    solver_coarse_nn.solve(verbose=True)

    # 8. 评估内部流场节点精度提升
    sl = solver_coarse_orig.sl
    err_u_orig_max = np.max(np.abs(u_sampled[sl] - solver_coarse_orig.u[sl]))
    err_u_nn_max   = np.max(np.abs(u_sampled[sl] - solver_coarse_nn.u[sl]))
    rmse_u_orig    = np.sqrt(np.mean((u_sampled[sl] - solver_coarse_orig.u[sl]) ** 2))
    rmse_u_nn      = np.sqrt(np.mean((u_sampled[sl] - solver_coarse_nn.u[sl]) ** 2))

    imp_max  = (1.0 - err_u_nn_max / (err_u_orig_max + 1e-30)) * 100
    imp_rmse = (1.0 - rmse_u_nn / (rmse_u_orig + 1e-30)) * 100

    print("\n" + "=" * 68)
    print(f"  [{case_name}] 内部流场节点速度 u 误差对比汇总:")
    print(f"  Max Error : 原粗网格 = {err_u_orig_max:.4e} -> NN修正后 = {err_u_nn_max:.4e} (降低 {imp_max:.2f}%)")
    print(f"  RMSE      : 原粗网格 = {rmse_u_orig:.4e} -> NN修正后 = {rmse_u_nn:.4e} (降低 {imp_rmse:.2f}%)")
    print("=" * 68)

    if plot_results:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        fig.suptitle(f"Case: {case_name} | Velocity Magnitude | RMSE Reduction: {imp_rmse:.1f}%",
                     fontsize=13, fontweight='bold')
        for ax, sv, title in zip(
            axes,
            [solver_dense, solver_coarse_orig, solver_coarse_nn],
            [f"Dense Ref ({mesh_dense.nx}x{mesh_dense.ny})",
             f"Coarse Orig ({mesh_coarse.nx}x{mesh_coarse.ny})",
             f"Coarse NN-Corrected ({mesh_coarse.nx}x{mesh_coarse.ny})"]
        ):
            vel = np.sqrt(sv.u ** 2 + sv.v ** 2)
            cf = ax.contourf(sv.X.T, sv.Y.T, vel.T, levels=25, cmap='viridis')
            plt.colorbar(cf, ax=ax)
            ax.set_title(title)
            ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(f"./iwa_nn_correct_uniform_grid_solution/{case_name}_comparison.png", dpi=150)
        plt.show()

    return solver_dense, solver_coarse_orig, solver_coarse_nn


# ============================================================
# 示例主程序: 配置文件驱动演示 (Lid-Driven Cavity & Channel Flow)
# ============================================================
if __name__ == '__main__':
    os.makedirs("./meshes", exist_ok=True)
    os.makedirs("./iwa_nn_correct_uniform_grid_solution", exist_ok=True)

    # --------------------------------------------------------
    # 示例 1: 顶盖驱动方腔流配置 (四面均为 wall，top 给定切向速度)
    # --------------------------------------------------------
    export_rect_gmsh("./meshes/cavity_dense_81x81.msh", nx=81, ny=81, Lx=1.0, Ly=1.0)
    # 粗网格可直接填写你上传的 msh 文件路径，或使用自动生成的 11x11 网格
    export_rect_gmsh("./meshes/cavity_coarse_11x11.msh", nx=11, ny=11, Lx=1.0, Ly=1.0)

    config_lid_driven = {
        "case_name": "Lid_Driven_Cavity_Re100",
        "meshes": {
            "dense_msh": "./meshes/cavity_dense_81x81.msh",
            "coarse_msh": "./meshes/cavity_coarse_11x11.msh"
        },
        "fluid": {
            "rho": 1.0,
            "mu": 0.01
        },
        "initial_conditions": {
            "u": 0.0, "v": 0.0, "p": 0.0
        },
        # 直接使用 Gmsh $PhysicalNames 中的名称配置边界条件
        "boundary_conditions": {
            "left":   {"type": "wall", "u": 0.0, "v": 0.0},
            "right":  {"type": "wall", "u": 0.0, "v": 0.0},
            "bottom": {"type": "wall", "u": 0.0, "v": 0.0},
            "top":    {"type": "wall", "u": 1.0, "v": 0.0}
        },
        "solver": {
            "max_iter": 2000, "tol": 1e-7,
            "alpha_u": 0.7, "alpha_v": 0.7, "alpha_p": 0.3
        },
        "nn_config": {
            "epochs": 4000, "lr": 3e-3, "hidden_dim": 96, "num_freqs": 6
        }
    }

    # --------------------------------------------------------
    # 示例 2: 2D 槽道流配置 (left=inlet, right=pressure_outlet, 上下=wall)
    # --------------------------------------------------------
    export_rect_gmsh("./meshes/channel_dense_81x41.msh", nx=81, ny=41, Lx=10.0, Ly=1.0)
    export_rect_gmsh("./meshes/channel_coarse_21x11.msh", nx=21, ny=11, Lx=10.0, Ly=1.0)

    config_channel_flow = {
        "case_name": "Channel_Flow_Re100",
        "meshes": {
            "dense_msh": "./meshes/channel_dense_81x41.msh",
            "coarse_msh": "./meshes/channel_coarse_21x11.msh"
        },
        "fluid": {
            "rho": 1.0,
            "mu": 0.01
        },
        "initial_conditions": {
            "u": 1.0, "v": 0.0, "p": 0.0
        },
        "boundary_conditions": {
            "left":   {"type": "inlet",           "u": 1.0, "v": 0.0},
            "right":  {"type": "pressure_outlet", "p": 0.0},
            "bottom": {"type": "wall",            "u": 0.0, "v": 0.0},
            "top":    {"type": "wall",            "u": 0.0, "v": 0.0}
        },
        "solver": {
            "max_iter": 3000, "tol": 1e-6,
            "alpha_u": 0.6, "alpha_v": 0.6, "alpha_p": 0.2
        },
        "nn_config": {
            "epochs": 3000, "lr": 2e-3, "hidden_dim": 64, "num_freqs": 4
        }
    }

    # 运行两个算例验证通用性
    run_universal_pipeline(config_lid_driven, plot_results=True)
    run_universal_pipeline(config_channel_flow, plot_results=True)

    #with open("config.json", "r", encoding="utf-8") as f:
    #cfg = json.load(f)
    #run_universal_pipeline(cfg)