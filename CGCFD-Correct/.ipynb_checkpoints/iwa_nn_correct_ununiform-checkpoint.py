#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 2D 非均匀/均匀结构化四边形网格 SIMPLE 求解器与神经网络双残差误差补偿框架
================================================================================
核心特性:
  1. 完整支持任意拉伸/壁面加密的正交结构化四边形 Gmsh (.msh) 网格
  2. 两遍扫描解析 Gmsh 2.2 文件，不受 $PhysicalNames 块在文件首尾位置的影响
  3. 配置文件驱动边界条件 ('wall', 'inlet', 'pressure_outlet')，自动识别封闭域压力参考点
  4. 非均匀网格二阶有限体积/有限差分离散 + 距离加权 Rhie-Chow 插值 + 五对角稀疏直接求解 (spsolve)
  5. 动量 (Res_u, Res_v) + 连续性 (Res_div) 双残差计算与 FourierMLP 神经网络代理补偿
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

# 固定随机种子以保证可复现性
torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# 模块 1: Gmsh (.msh) 均匀/非均匀拉伸网格生成与结构化重构解析器
# ============================================================
def generate_1d_coords(L, n, stretch=0.0):
    """
    生成 [0, L] 区间上的 1D 节点坐标:
      - stretch == 0.0 : 均匀分布
      - stretch >  0.0 : 双曲正切 (tanh) 两端壁面对称加密分布 (典型取值 1.2 ~ 2.5)
    """
    xi = np.linspace(0.0, 1.0, n)
    if stretch <= 1e-8:
        return L * xi
    s = 2.0 * xi - 1.0  # 映射到 [-1, 1]
    return 0.5 * L * (1.0 + np.tanh(stretch * s) / np.tanh(stretch))


def export_rect_gmsh(filename, nx, ny, Lx, Ly, stretch_x=0.0, stretch_y=0.0):
    """
    生成与 Pointwise 导出格式完全兼容的 Gmsh 2.2 ASCII 四边形网格 (.msh)
    支持设置 stretch_x / stretch_y 生成壁面加密的非均匀结构化网格。
    """
    x = generate_1d_coords(Lx, nx, stretch_x)
    y = generate_1d_coords(Ly, ny, stretch_y)

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
    完整支持均匀与任意拉伸非均匀的正交结构化四边形网格。
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

        # ===== 第一遍扫描：优先提取 $PhysicalNames（兼容 Pointwise 将其放在文件末尾）=====
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
        # 1. 提取节点坐标并消除浮点截断误差
        nids = np.array(list(self.raw_nodes.keys()), dtype=int)
        coords = np.array([self.raw_nodes[nid][:2] for nid in nids], dtype=float)
        rounded_coords = np.round(coords, self.tol_decimals)

        self.x = np.unique(rounded_coords[:, 0])
        self.y = np.unique(rounded_coords[:, 1])
        self.nx = len(self.x)
        self.ny = len(self.y)

        if self.nx * self.ny != len(nids):
            raise ValueError(
                f"网格非正交结构化四边形拓扑: 唯一坐标乘积 ({self.nx}x{self.ny}={self.nx*self.ny}) "
                f"!= 节点总数 ({len(nids)})"
            )

        self.Lx = float(self.x[-1] - self.x[0])
        self.Ly = float(self.y[-1] - self.y[0])
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # 2. 建立 Gmsh node_id <-> 结构化索引 (i, j) 的双向映射
        ix_all = np.searchsorted(self.x, rounded_coords[:, 0])
        iy_all = np.searchsorted(self.y, rounded_coords[:, 1])

        self.node_to_ij = {}
        self.ij_to_node = np.zeros((self.nx, self.ny), dtype=int)
        for nid, i, j in zip(nids, ix_all, iy_all):
            self.node_to_ij[nid] = (int(i), int(j))
            self.ij_to_node[i, j] = nid

        # 使用真实节点坐标还原更精确的 1D 坐标数组 (避免过度舍入影响极密网格)
        self.x = np.array([self.raw_nodes[self.ij_to_node[i, 0]][0] for i in range(self.nx)], dtype=float)
        self.y = np.array([self.raw_nodes[self.ij_to_node[0, j]][1] for j in range(self.ny)], dtype=float)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # 3. 预计算支持任意非均匀拉伸网格的局部步长与插值权重张量 (利用广播机制)
        # 单元面间距: hx[i] = x[i+1] - x[i], shape=(nx-1, 1); hy[j] = y[j+1] - y[j], shape=(1, ny-1)
        self.hx = np.diff(self.x)[:, None]
        self.hy = np.diff(self.y)[None, :]

        # 内部节点 (i=1..nx-2, j=1..ny-2) 到东/西/北/南邻居节点的真实距离
        self.dx_E = self.hx[1:, :]                         # (nx-2, 1)
        self.dx_W = self.hx[:-1, :]                        # (nx-2, 1)
        self.dx_P = 0.5 * (self.dx_E + self.dx_W)          # (nx-2, 1) 控制体宽度 (x_e - x_w)

        self.dy_N = self.hy[:, 1:]                         # (1, ny-2)
        self.dy_S = self.hy[:, :-1]                        # (1, ny-2)
        self.dy_P = 0.5 * (self.dy_N + self.dy_S)          # (1, ny-2) 控制体高度 (y_n - y_s)

        # 内部节点控制体面积 (2D 体积)，用于纯 Neumann 泊松方程的体积加权零均值约束
        self.cell_vol = self.dx_P * self.dy_P              # (nx-2, ny-2)

        # 判断网格是否为非均匀拉伸网格
        self.is_uniform = (
            np.allclose(self.hx, self.Lx / (self.nx - 1), rtol=1e-3) and
            np.allclose(self.hy, self.Ly / (self.ny - 1), rtol=1e-3)
        )

        # 4. 识别四条结构化边界 ('west', 'east', 'south', 'north') 对应的 Gmsh 物理名称
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
# 模块 3: 通用非均匀网格 SIMPLE 求解器
# ============================================================
class UniversalSIMPLE:
    """
    支持任意非均匀正交结构化四边形网格与配置化边界条件的 2D 不可压 SIMPLE 求解器。
    """
    def __init__(self, mesh: StructuredGmshMesh, config: dict):
        self.mesh = mesh
        self.config = config
        self.nx, self.ny = mesh.nx, mesh.ny
        self.Lx, self.Ly = mesh.Lx, mesh.Ly
        self.x, self.y = mesh.x, mesh.y
        self.X, self.Y = mesh.X, mesh.Y

        fluid_cfg = config.get('fluid', {})
        self.rho = float(fluid_cfg.get('rho', 1.0))
        self.mu = float(fluid_cfg.get('mu', 0.01))

        # 解析四条结构化边 ('west', 'east', 'south', 'north') 的边界配置 (支持物理名与方位别名回退)
        raw_bc_cfg = config.get('boundary_conditions', {})
        side_aliases = {
            'west':  ['west', 'left'],
            'east':  ['east', 'right'],
            'south': ['south', 'bottom'],
            'north': ['north', 'top']
        }
        self.side_bc = {}
        for side in ['west', 'east', 'south', 'north']:
            phys_name = mesh.side_to_phys_name[side]
            if phys_name in raw_bc_cfg:
                self.side_bc[side] = raw_bc_cfg[phys_name]
            else:
                matched = False
                for alias in side_aliases[side]:
                    if alias in raw_bc_cfg:
                        self.side_bc[side] = raw_bc_cfg[alias]
                        matched = True
                        break
                if not matched:
                    raise KeyError(f"配置文件中缺少边界 '{phys_name}' (对应方位 '{side}') 的定义!")

        # 自动检测是否存在压力出口边界 (决定压力泊松方程是否需要锚定参考点)
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
        # 1. 先处理 pressure_outlet (零梯度导出)
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

        # 2. 再处理 inlet (速度入口)
        for side, bc in self.side_bc.items():
            if bc['type'] == 'inlet':
                self._apply_side_dirichlet_uv(side, bc)

        # 3. 最后处理 wall (确保交界角点处无滑移壁面优先)
        for side, bc in self.side_bc.items():
            if bc['type'] == 'wall':
                self._apply_side_dirichlet_uv(side, bc)

    def _bc_p(self):
        if not self.has_pressure_outlet:
            self.p -= self.p[1, 1]

        for side, bc in self.side_bc.items():
            if bc['type'] in ('wall', 'inlet'):
                if side == 'west':    self.p[0, :] = self.p[1, :]
                elif side == 'east':  self.p[-1, :] = self.p[-2, :]
                elif side == 'south': self.p[:, 0] = self.p[:, 1]
                elif side == 'north': self.p[:, -1] = self.p[:, -2]

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

    def _gradient(self, f):
        """
        使用 np.gradient(..., edge_order=2) 计算非均匀正交网格上的二阶精度梯度。
        在均匀网格下严格退化为中心差分与三点单侧边界公式；在非均匀网格下自动采用二阶拉格朗日权重。
        """
        Gx = np.gradient(f, self.x, axis=0, edge_order=2)
        Gy = np.gradient(f, self.y, axis=1, edge_order=2)
        return Gx, Gy

    def _assemble_momentum(self, direction):
        rho, mu = self.rho, self.mu
        sl = self.sl
        m = self.mesh

        ui = self.u[sl]
        vi = self.v[sl]

        # 非均匀网格二阶扩散系数: D_e = mu / (dx_P * dx_E)
        De = mu / (m.dx_P * m.dx_E)
        Dw = mu / (m.dx_P * m.dx_W)
        Dn = mu / (m.dy_P * m.dy_N)
        Ds = mu / (m.dy_P * m.dy_S)

        # 非均匀网格一阶迎风对流系数
        aE_c = np.where(ui >= 0, 0.0, -rho * ui / m.dx_E)
        aW_c = np.where(ui >= 0,  rho * ui / m.dx_W, 0.0)
        aN_c = np.where(vi >= 0, 0.0, -rho * vi / m.dy_N)
        aS_c = np.where(vi >= 0,  rho * vi / m.dy_S, 0.0)

        self.aE[sl] = De + aE_c
        self.aW[sl] = Dw + aW_c
        self.aN[sl] = Dn + aN_c
        self.aS[sl] = Ds + aS_c
        self.aP[sl] = self.aE[sl] + self.aW[sl] + self.aN[sl] + self.aS[sl]

        Gx, Gy = self._gradient(self.p)
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
        m = self.mesh

        d = np.zeros((nx, ny))
        d[1:-1, 1:-1] = alpha_u / self.aP[1:-1, 1:-1]
        d[0, :]  = d[1, :]
        d[-1, :] = d[-2, :]
        d[:, 0]  = d[:, 1]
        d[:, -1] = d[:, -2]

        Gx, Gy = self._gradient(self.p)

        # 东界面 e (位于节点 i 与 i+1 中点，距离为 m.hx)
        d_bar_e = 0.5 * (d[:-1, :] + d[1:, :])
        Gx_bar_e = 0.5 * (Gx[:-1, :] + Gx[1:, :])
        Gx_compact_e = (self.p[1:, :] - self.p[:-1, :]) / m.hx
        u_e = 0.5 * (self.u[:-1, :] + self.u[1:, :]) + d_bar_e * (Gx_bar_e - Gx_compact_e)

        # 北界面 n (位于节点 j 与 j+1 中点，距离为 m.hy)
        d_bar_n = 0.5 * (d[:, :-1] + d[:, 1:])
        Gy_bar_n = 0.5 * (Gy[:, :-1] + Gy[:, 1:])
        Gy_compact_n = (self.p[:, 1:] - self.p[:, :-1]) / m.hy
        v_n = 0.5 * (self.v[:, :-1] + self.v[:, 1:]) + d_bar_n * (Gy_bar_n - Gy_compact_n)

        return u_e, v_n, d

    def _solve_pressure_correction(self, alpha_u):
        nx, ny = self.nx, self.ny
        nix, niy = nx - 2, ny - 2
        sl = self.sl
        m = self.mesh

        u_e, v_n, d = self._rc_face_velocities(alpha_u)

        # 非均匀控制体散度: (u_e - u_w) / dx_P + (v_n - v_s) / dy_P
        div = np.zeros((nx, ny))
        div[sl] = (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) / m.dx_P + \
                  (v_n[1:-1, 1:] - v_n[1:-1, :-1]) / m.dy_P

        if self.Sp_nn_div is not None:
            b = -(div[sl] - self.Sp_nn_div[sl]).copy()
        else:
            b = -div[sl].copy()

        mass_resid = float(np.max(np.abs(b)))

        # 非均匀压力修正泊松系数
        aE_pc = 0.5 * (d[1:-1, 1:-1] + d[2:, 1:-1])  / (m.dx_P * m.dx_E)
        aW_pc = 0.5 * (d[:-2, 1:-1]  + d[1:-1, 1:-1]) / (m.dx_P * m.dx_W)
        aN_pc = 0.5 * (d[1:-1, 1:-1] + d[1:-1, 2:])  / (m.dy_P * m.dy_N)
        aS_pc = 0.5 * (d[1:-1, :-2]  + d[1:-1, 1:-1]) / (m.dy_P * m.dy_S)
        aP_pc = aE_pc + aW_pc + aN_pc + aS_pc

        if self.side_bc['west']['type'] in ('wall', 'inlet'):
            aP_pc[0, :] -= aW_pc[0, :]
        if self.side_bc['east']['type'] in ('wall', 'inlet'):
            aP_pc[-1, :] -= aE_pc[-1, :]
        if self.side_bc['south']['type'] in ('wall', 'inlet'):
            aP_pc[:, 0] -= aS_pc[:, 0]
        if self.side_bc['north']['type'] in ('wall', 'inlet'):
            aP_pc[:, -1] -= aN_pc[:, -1]

        if not self.has_pressure_outlet:
            aE_pc[0, 0] = aW_pc[0, 0] = aN_pc[0, 0] = aS_pc[0, 0] = 0.0
            b[0, 0] = 0.0

        A_p = build_penta_matrix(aP_pc, aE_pc, aW_pc, aN_pc, aS_pc)
        pp_internal = spsolve(A_p, b.ravel())

        pp = np.zeros((nx, ny))
        pp[sl] = pp_internal.reshape((nix, niy), order='C')
        self._bc_p_prime(pp)

        return pp, d, mass_resid

    def _correct_fields(self, pp, d, alpha_p):
        sl = self.sl
        self.p[sl] += alpha_p * pp[sl]
        Gx_pp, Gy_pp = self._gradient(pp)
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

        grid_type_str = "均匀正交网格" if self.mesh.is_uniform else "非均匀拉伸网格"
        if verbose:
            print("=" * 72)
            print(f"  通用 SIMPLE 求解器 | 案例: {self.config.get('case_name', 'Unnamed')} "
                  f"| 网格: {self.nx}x{self.ny} ({grid_type_str})")
            if self.Sp_nn_u is not None:
                print("  ✓ 已启用神经网络双残差源项补偿 (Res_u + Res_v + Res_div)")
            print("=" * 72)

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
# 模块 4: 非均匀网格完备离散双残差计算器
# ============================================================
def compute_discrete_residual(solver: UniversalSIMPLE, u_s, v_s, p_s):
    """
    将密网格采样流场 (u_s, v_s, p_s) 载入粗网格求解器模板，
    严格同步粗网格边界条件后计算非均匀粗网格自洽算子下的双残差 (Res_u, Res_v, Res_div)。
    """
    alpha_u = float(solver.config.get('solver', {}).get('alpha_u', 0.65))
    m = solver.mesh

    solver.u = u_s.copy()
    solver.v = v_s.copy()
    solver.p = p_s.copy()
    solver._bc_uv()
    solver._bc_p()

    sl = solver.sl

    # 1. 组装粗网格动量方程系数
    solver._assemble_momentum('u')
    Gx, Gy = solver._gradient(solver.p)
    Sp_u = -Gx[sl]
    Sp_v = -Gy[sl]

    # 2. 计算动量离散残差
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

    # 3. 计算非均匀粗网格 Rhie-Chow 界面速度下的离散散度残差
    u_e, v_n, _ = solver._rc_face_velocities(alpha_u)
    Res_div = (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) / m.dx_P + \
              (v_n[1:-1, 1:] - v_n[1:-1, :-1]) / m.dy_P

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
        epochs     = int(nn_cfg.get('epochs', 3500))
        lr         = float(nn_cfg.get('lr', 3e-3))
        hidden_dim = int(nn_cfg.get('hidden_dim', 96))
        num_freqs  = int(nn_cfg.get('num_freqs', 6))

        print(f"\n{'='*72}")
        print(f"  训练 FourierMLP 代理模型 | 样本数: {len(X_train)} | 设备: {self.device}")
        print(f"{'='*72}")

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

        # 若为无压力出口的封闭域，施加基于非均匀控制体面积 (cell_vol) 的加权零均值约束
        if not solver.has_pressure_outlet:
            sl = solver.sl
            vol = solver.mesh.cell_vol
            vol_weighted_mean = np.sum(preds['div'][sl] * vol) / np.sum(vol)
            preds['div'][sl] -= vol_weighted_mean

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
    print(f"  密网格 ({dense_msh_path})  -> 维度: {mesh_dense.nx}x{mesh_dense.ny}, "
          f"dx范围: [{mesh_dense.hx.min():.4f}, {mesh_dense.hx.max():.4f}], "
          f"dy范围: [{mesh_dense.hy.min():.4f}, {mesh_dense.hy.max():.4f}]")
    print(f"  粗网格 ({coarse_msh_path}) -> 维度: {mesh_coarse.nx}x{mesh_coarse.ny}, "
          f"边界映射: {mesh_coarse.side_to_phys_name}")

    # 2. 密网格求解 (高精度参考解)
    solver_dense = UniversalSIMPLE(mesh_dense, config)
    solver_dense.solve(verbose=True)

    # 3. 在非均匀粗网格节点上双三次样条采样密网格流场
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

    print("\n" + "=" * 72)
    print(f"  [{case_name}] 内部流场节点速度 u 误差对比汇总:")
    print(f"  Max Error : 原粗网格 = {err_u_orig_max:.4e} -> NN修正后 = {err_u_nn_max:.4e} (降低 {imp_max:.2f}%)")
    print(f"  RMSE      : 原粗网格 = {rmse_u_orig:.4e} -> NN修正后 = {rmse_u_nn:.4e} (降低 {imp_rmse:.2f}%)")
    print("=" * 72)

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
            # 绘制非均匀网格线以直观检查壁面加密情况
            if sv.nx <= 25:
                ax.plot(sv.X, sv.Y, 'w-', lw=0.4, alpha=0.5)
                ax.plot(sv.X.T, sv.Y.T, 'w-', lw=0.4, alpha=0.5)
            plt.colorbar(cf, ax=ax)
            ax.set_title(title)
            ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(f"./{case_name}_comparison.png", dpi=150)
        print(f"  图像已保存至: ./{case_name}_comparison.png")
        plt.show()

    return solver_dense, solver_coarse_orig, solver_coarse_nn


# ============================================================
# 主程序演示: 包含【非均匀壁面加密】的顶盖驱动流与槽道流算例
# ============================================================
if __name__ == '__main__':
    os.makedirs("./meshes", exist_ok=True)

    # --------------------------------------------------------
    # 示例 1: 四周壁面拉伸加密的顶盖驱动方腔流 (stretch_x=1.5, stretch_y=1.5)
    # --------------------------------------------------------
    export_rect_gmsh("./meshes/cavity_dense_stretched.msh",
                     nx=81, ny=81, Lx=1.0, Ly=1.0, stretch_x=1.5, stretch_y=1.5)
    export_rect_gmsh("./meshes/cavity_coarse_stretched.msh",
                     nx=13, ny=13, Lx=1.0, Ly=1.0, stretch_x=1.5, stretch_y=1.5)

    config_lid_driven = {
        "case_name": "Lid_Driven_Stretched_Re100",
        "meshes": {
            "dense_msh": "./meshes/cavity_dense_stretched.msh",
            "coarse_msh": "./meshes/cavity_coarse_stretched.msh"
        },
        "fluid": {
            "rho": 1.0,
            "mu": 0.01
        },
        "initial_conditions": {
            "u": 0.0, "v": 0.0, "p": 0.0
        },
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
    # 示例 2: 上下壁面边界层加密的 2D 槽道流 (stretch_y=1.8)
    # --------------------------------------------------------
    export_rect_gmsh("./meshes/channel_dense_stretched.msh",
                     nx=81, ny=41, Lx=10.0, Ly=1.0, stretch_x=0.0, stretch_y=1.8)
    export_rect_gmsh("./meshes/channel_coarse_stretched.msh",
                     nx=21, ny=11, Lx=10.0, Ly=1.0, stretch_x=0.0, stretch_y=1.8)

    config_channel_flow = {
        "case_name": "Channel_Flow_Stretched_Re100",
        "meshes": {
            "dense_msh": "./meshes/channel_dense_stretched.msh",
            "coarse_msh": "./meshes/channel_coarse_stretched.msh"
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

    # 运行两个非均匀拉伸网格算例
    run_universal_pipeline(config_lid_driven, plot_results=True)
    run_universal_pipeline(config_channel_flow, plot_results=True)