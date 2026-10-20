#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================================
DeepSIMPLE_2D: 通用 2D 多块掩模结构化网格 SIMPLE 求解器与神经网络残差补偿框架 (基础掩模版)
==================================================================================================

【核心功能与物理引擎】
 1. 复杂几何支持: 突破传统结构化网格的单块矩形限制，全面支持后台阶、L型、内部障碍物等多块正交网格。
    程序自动扫描 Gmsh 文件中的无网格空洞，构建固体验模 (Solid Mask) 进行流固隔离。
 2. 智能边界系统: 直接基于 Gmsh 1D 物理单元施加复杂分段边界条件，内置别名容错机制 (Alias Fallback)，
    并搭载万能法向探测算法，完美适应具有折角的复杂 Neumann 零梯度边界外推。
 3. 掩码感知算子: 在计算二阶空间梯度和 Rhie-Chow 动量插值时，自动识别流固交界面，
    切断固体节点污染，保证纯流体区域的物理自洽。
 4. 稀疏矩阵加速: 冻结固体区节点并采用极大值主对角元锁死，全域保持五对角稀疏矩阵向量化求解 (spsolve)。

【SciML 代理模型与局限性】
 - 本版本采用常规的双线性插值采样与全局 FourierMLP 进行残差预测。
 - ⚠️ 局限性说明:
   1. 线性插值在流固交界面容易产生斜率跳变 (Slope Jumps)。
   2. 纯傅里叶网络在拟合台阶拐角处的“高频脉冲残差”时，易引发吉布斯振荡 (Gibbs Phenomenon)。
   (若需解决上述数值不稳定问题，请使用本框架的“高精度增强版”)
==================================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator

# 固定随机种子以保证可复现性
torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# 模块 1: 支持任意台阶/多块正交四边形的 Gmsh (.msh) 结构化重构解析器
# ============================================================
class StructuredGmshMesh:
    """
    读取 Gmsh 2.2 (.msh) 网格文件，并将其映射到外包正交结构化网格 (nx, ny)。
    自动识别缺失节点的固体障碍区域 (solid_mask) 以及任意分段 1D 物理边界。
    """
    def __init__(self, msh_filepath, tol_decimals=6):
        self.filepath = msh_filepath
        self.tol_decimals = tol_decimals

        self.raw_nodes = {}          # node_id -> (x, y, z)
        self.phys_names = {}         # (dim, tag) -> name_str
        self.boundary_nodes = {}     # phys_name -> set of node_ids
        self.boundary_edges = {}     # phys_name -> list of (n1, n2)
        self.quad_elements = []      # list of (n1, n2, n3, n4)

        self._parse_msh()
        self._build_structured_topology()

    def _parse_msh(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        # 第一遍扫描：提取 $PhysicalNames（兼容 Pointwise 将其放在文件末尾）
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
                        self.boundary_edges[name] = []
                break
            idx += 1

        tag_to_bname = {tag: name for (dim, tag), name in self.phys_names.items() if dim == 1}

        # 第二遍扫描：解析 $Nodes 和 $Elements
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
                        self.boundary_edges.setdefault(bname, []).append(tuple(node_list))
                    elif elm_type == 3:  # 2D 四边形单元
                        self.quad_elements.append(node_list)
            idx += 1

    def _build_structured_topology(self):
        nids = np.array(list(self.raw_nodes.keys()), dtype=int)
        coords = np.array([self.raw_nodes[nid][:2] for nid in nids], dtype=float)
        rounded_coords = np.round(coords, self.tol_decimals)

        self.x = np.unique(rounded_coords[:, 0])
        self.y = np.unique(rounded_coords[:, 1])
        self.nx = len(self.x)
        self.ny = len(self.y)

        # 建立 Gmsh node_id <-> 结构化索引 (i, j) 的双向映射
        ix_all = np.searchsorted(self.x, rounded_coords[:, 0])
        iy_all = np.searchsorted(self.y, rounded_coords[:, 1])

        self.node_to_ij = {}
        self.ij_to_node = np.full((self.nx, self.ny), -1, dtype=int)
        for nid, i, j in zip(nids, ix_all, iy_all):
            self.node_to_ij[nid] = (int(i), int(j))
            self.ij_to_node[i, j] = nid

        # 校验所有有效节点是否无冲突地落入正交网格交点
        valid_count = int(np.sum(self.ij_to_node >= 0))
        if valid_count != len(nids):
            raise ValueError(
                f"存在坐标重合或非正交错位节点: 唯一映射节点数 ({valid_count}) != 实际节点总数 ({len(nids)})"
            )

        # 精确还原 1D 坐标数组 (在每个 i 列或 j 行取第一个有效真实节点的坐标)
        exact_x = np.zeros(self.nx, dtype=float)
        for i in range(self.nx):
            valid_nids = self.ij_to_node[i, self.ij_to_node[i, :] >= 0]
            exact_x[i] = self.raw_nodes[int(valid_nids[0])][0]

        exact_y = np.zeros(self.ny, dtype=float)
        for j in range(self.ny):
            valid_nids = self.ij_to_node[self.ij_to_node[:, j] >= 0, j]
            exact_y[j] = self.raw_nodes[int(valid_nids[0])][1]

        self.x = exact_x
        self.y = exact_y
        self.Lx = float(self.x[-1] - self.x[0])
        self.Ly = float(self.y[-1] - self.y[0])
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # 构建流体/固体掩码: True 表示存在网格节点 (流体或边界), False 表示台阶等固体空洞区
        self.valid_mask = (self.ij_to_node >= 0)
        self.solid_mask = ~self.valid_mask
        self.has_obstacle = bool(np.any(self.solid_mask))

        # 为每个物理边界生成 2D 布尔掩码与法向导出映射
        self.bc_masks = {}
        all_b_mask = np.zeros((self.nx, self.ny), dtype=bool)
        for bname, b_nids in self.boundary_nodes.items():
            mask = np.zeros((self.nx, self.ny), dtype=bool)
            for nid in b_nids:
                if nid in self.node_to_ij:
                    i, j = self.node_to_ij[nid]
                    mask[i, j] = True
            self.bc_masks[bname] = mask
            all_b_mask |= mask

        # 内部流体求解节点: 属于有效网格点，且不在任何 1D 物理边界上
        self.fluid_internal = self.valid_mask & (~all_b_mask)
        # 确保外框一圈不会误入 internal 切片 (1:-1, 1:-1)
        self.fluid_internal[0, :] = False
        self.fluid_internal[-1, :] = False
        self.fluid_internal[:, 0] = False
        self.fluid_internal[:, -1] = False

        # 预计算局部步长张量
        self.hx = np.diff(self.x)[:, None]                   # (nx-1, 1)
        self.hy = np.diff(self.y)[None, :]                   # (1, ny-1)

        self.dx_E = self.hx[1:, :]                           # (nx-2, 1)
        self.dx_W = self.hx[:-1, :]                          # (nx-2, 1)
        self.dx_P = 0.5 * (self.dx_E + self.dx_W)            # (nx-2, 1)

        self.dy_N = self.hy[:, 1:]                           # (1, ny-2)
        self.dy_S = self.hy[:, :-1]                          # (1, ny-2)
        self.dy_P = 0.5 * (self.dy_N + self.dy_S)            # (1, ny-2)

        self.cell_vol = self.dx_P * self.dy_P                # (nx-2, ny-2)

        self.is_uniform = (
            np.allclose(self.hx, self.Lx / (self.nx - 1), rtol=1e-2) and
            np.allclose(self.hy, self.Ly / (self.ny - 1), rtol=1e-2)
        )


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
# 模块 3: 支持多块/后台阶掩码的通用 SIMPLE 求解器
# ============================================================
class UniversalSIMPLE:
    """
    支持任意非均匀正交结构化四边形网格、多块 L 型台阶网格与配置化边界条件的 2D 不可压 SIMPLE 求解器。
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

        # 绑定物理边界条件
        raw_bc_cfg = config.get('boundary_conditions', {})
        self.bc_cfg = {}
        for bname in mesh.bc_masks.keys():
            if bname not in raw_bc_cfg:
                raise KeyError(f"配置文件 boundary_conditions 中缺少网格物理边界 '{bname}' 的定义!")
            cfg_item = dict(raw_bc_cfg[bname])
            # 自动纠错：防止用户将名称为 'inlet' 且速度非零的入口误配成 type='wall'
            if bname.lower() == 'inlet' and cfg_item.get('type') == 'wall':
                if abs(float(cfg_item.get('u', 0.0))) > 0 or abs(float(cfg_item.get('v', 0.0))) > 0:
                    cfg_item['type'] = 'inlet'
            self.bc_cfg[bname] = cfg_item

        self.has_pressure_outlet = any(
            bc['type'] == 'pressure_outlet' for bc in self.bc_cfg.values()
        )

        # 预构建边界点到最近流体内部点的零梯度外推映射表 (用于壁面/入口 Neumann 压力边界及出口速度边界)
        self._build_neumann_extrapolation_maps()

        sh = (self.nx, self.ny)
        self.u = np.zeros(sh)
        self.v = np.zeros(sh)
        self.p = np.zeros(sh)

        self.aE = np.zeros(sh); self.aW = np.zeros(sh)
        self.aN = np.zeros(sh); self.aS = np.zeros(sh)
        self.aP = np.zeros(sh); self.Sp = np.zeros(sh)

        self.sl = (slice(1, -1), slice(1, -1))
        self.active_in = self.mesh.fluid_internal[self.sl]  # (nx-2, ny-2) 内部流体布尔掩码

        # 寻找第一个有效内部流体节点作为封闭域压力参考点
        active_indices = np.argwhere(self.active_in)
        self.ref_p_ij = tuple(active_indices[0])  # 相对于 (1:-1, 1:-1) 的局部索引

        # 神经网络双残差补偿源项
        self.Sp_nn_u = None
        self.Sp_nn_v = None
        self.Sp_nn_div = None

        self._init_fields()

    def _build_neumann_extrapolation_maps(self):
        """
        为所有边界节点寻找指向流体内部的法向邻居索引，支持台阶折角与任意方位边界。
        """
        self.neumann_pairs = {}  # bname -> (b_i, b_j, nbr_i, nbr_j)
        fl = self.mesh.fluid_internal
        val = self.mesh.valid_mask

        for bname, mask in self.mesh.bc_masks.items():
            bi_list, bj_list, ni_list, nj_list = [], [], [], []
            b_indices = np.argwhere(mask)
            for i, j in b_indices:
                # 优先寻找正交方向上的流体内部节点
                found = False
                for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.nx and 0 <= nj < self.ny and fl[ni, nj]:
                        bi_list.append(i); bj_list.append(j)
                        ni_list.append(ni); nj_list.append(nj)
                        found = True
                        break
                if not found:
                    # 角点回退：寻找对角线方向的流体内部节点或有效节点
                    for di, dj in [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.nx and 0 <= nj < self.ny and val[ni, nj] and (ni != i or nj != j):
                            bi_list.append(i); bj_list.append(j)
                            ni_list.append(ni); nj_list.append(nj)
                            break
            self.neumann_pairs[bname] = (
                np.array(bi_list, dtype=int),
                np.array(bj_list, dtype=int),
                np.array(ni_list, dtype=int),
                np.array(nj_list, dtype=int),
            )

    def _init_fields(self):
        init_cfg = self.config.get('initial_conditions', {'u': 0.0, 'v': 0.0, 'p': 0.0})
        self.u[:, :] = float(init_cfg.get('u', 0.0))
        self.v[:, :] = float(init_cfg.get('v', 0.0))
        self.p[:, :] = float(init_cfg.get('p', 0.0))
        self.u[self.mesh.solid_mask] = 0.0
        self.v[self.mesh.solid_mask] = 0.0
        self.p[self.mesh.solid_mask] = 0.0
        self._bc_uv()
        self._bc_p()

    def _bc_uv(self):
        # 固体障碍区速度恒为 0
        self.u[self.mesh.solid_mask] = 0.0
        self.v[self.mesh.solid_mask] = 0.0

        # 1. 先处理 pressure_outlet (速度零梯度导出)
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'pressure_outlet':
                bi, bj, ni, nj = self.neumann_pairs[bname]
                self.u[bi, bj] = self.u[ni, nj]
                self.v[bi, bj] = self.v[ni, nj]

        # 2. 再处理 inlet (给定入口速度)
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'inlet':
                mask = self.mesh.bc_masks[bname]
                self.u[mask] = float(bc.get('u', 0.0))
                self.v[mask] = float(bc.get('v', 0.0))

        # 3. 最后处理 wall (确保壁面与入口/出口交界角点严格满足无滑移条件)
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'wall':
                mask = self.mesh.bc_masks[bname]
                self.u[mask] = float(bc.get('u', 0.0))
                self.v[mask] = float(bc.get('v', 0.0))

    def _bc_p(self):
        if not self.has_pressure_outlet:
            ri, rj = self.ref_p_ij
            self.p -= self.p[ri + 1, rj + 1]

        # 1. 壁面和速度入口采用零法向梯度外推
        for bname, bc in self.bc_cfg.items():
            if bc['type'] in ('wall', 'inlet'):
                bi, bj, ni, nj = self.neumann_pairs[bname]
                self.p[bi, bj] = self.p[ni, nj]

        # 2. 压力出口固定给定压力
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'pressure_outlet':
                mask = self.mesh.bc_masks[bname]
                self.p[mask] = float(bc.get('p', 0.0))

        self.p[self.mesh.solid_mask] = 0.0

    def _bc_p_prime(self, pp):
        if not self.has_pressure_outlet:
            ri, rj = self.ref_p_ij
            pp -= pp[ri + 1, rj + 1]

        for bname, bc in self.bc_cfg.items():
            if bc['type'] in ('wall', 'inlet'):
                bi, bj, ni, nj = self.neumann_pairs[bname]
                pp[bi, bj] = pp[ni, nj]

        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'pressure_outlet':
                mask = self.mesh.bc_masks[bname]
                pp[mask] = 0.0

        pp[self.mesh.solid_mask] = 0.0

    def _gradient(self, f):
        """
        掩码感知 (Mask-Aware) 二阶精度梯度算子:
        对于邻居落入固体台阶区的内部节点，自动切换为流体侧单侧差分，避免固体区零值污染。
        """
        Gx = np.gradient(f, self.x, axis=0, edge_order=2)
        Gy = np.gradient(f, self.y, axis=1, edge_order=2)

        if self.mesh.has_obstacle:
            val = self.mesh.valid_mask
            # 修正 x 方向梯度: 若西侧 (i-1) 是固体且东侧 (i+1) 有效，使用前向差分
            west_solid = val[1:-1, :] & (~val[:-2, :]) & val[2:, :]
            if np.any(west_solid):
                dx_f = (self.x[2:] - self.x[1:-1])[:, None]
                Gx[1:-1, :] = np.where(west_solid, (f[2:, :] - f[1:-1, :]) / dx_f, Gx[1:-1, :])

            # 若东侧 (i+1) 是固体且西侧 (i-1) 有效，使用后向差分
            east_solid = val[1:-1, :] & (~val[2:, :]) & val[:-2, :]
            if np.any(east_solid):
                dx_b = (self.x[1:-1] - self.x[:-2])[:, None]
                Gx[1:-1, :] = np.where(east_solid, (f[1:-1, :] - f[:-2, :]) / dx_b, Gx[1:-1, :])

            # 修正 y 方向梯度: 若南侧 (j-1) 是固体且北侧 (j+1) 有效，使用前向差分
            south_solid = val[:, 1:-1] & (~val[:, :-2]) & val[:, 2:]
            if np.any(south_solid):
                dy_f = (self.y[2:] - self.y[1:-1])[None, :]
                Gy[:, 1:-1] = np.where(south_solid, (f[:, 2:] - f[:, 1:-1]) / dy_f, Gy[:, 1:-1])

            # 若北侧 (j+1) 是固体且南侧 (j-1) 有效，使用后向差分
            north_solid = val[:, 1:-1] & (~val[:, 2:]) & val[:, :-2]
            if np.any(north_solid):
                dy_b = (self.y[1:-1] - self.y[:, :-2])[None, :]
                Gy[:, 1:-1] = np.where(north_solid, (f[:, 1:-1] - f[:, :-2]) / dy_b, Gy[:, 1:-1])

        Gx[self.mesh.solid_mask] = 0.0
        Gy[self.mesh.solid_mask] = 0.0
        return Gx, Gy

    def _assemble_momentum(self, direction):
        rho, mu = self.rho, self.mu
        sl = self.sl
        m = self.mesh

        ui = self.u[sl]
        vi = self.v[sl]

        De = mu / (m.dx_P * m.dx_E)
        Dw = mu / (m.dx_P * m.dx_W)
        Dn = mu / (m.dy_P * m.dy_N)
        Ds = mu / (m.dy_P * m.dy_S)

        aE_c = np.where(ui >= 0, 0.0, -rho * ui / m.dx_E)
        aW_c = np.where(ui >= 0,  rho * ui / m.dx_W, 0.0)
        aN_c = np.where(vi >= 0, 0.0, -rho * vi / m.dy_N)
        aS_c = np.where(vi >= 0,  rho * vi / m.dy_S, 0.0)

        self.aE[sl] = De + aE_c
        self.aW[sl] = Dw + aW_c
        self.aN[sl] = Dn + aN_c
        self.aS[sl] = Ds + aS_c
        self.aP[sl] = self.aE[sl] + self.aW[sl] + self.aN[sl] + self.aS[sl]

        # 非活动节点（固体区或内部台阶边界）设为单位对角，避免除以零
        inactive = ~self.active_in
        self.aP[sl][inactive] = 1.0
        self.aE[sl][inactive] = 0.0
        self.aW[sl][inactive] = 0.0
        self.aN[sl][inactive] = 0.0
        self.aS[sl][inactive] = 0.0

        Gx, Gy = self._gradient(self.p)
        if direction == 'u':
            self.Sp[sl] = -Gx[sl]
            if self.Sp_nn_u is not None:
                self.Sp[sl] += self.Sp_nn_u[sl]
        else:
            self.Sp[sl] = -Gy[sl]
            if self.Sp_nn_v is not None:
                self.Sp[sl] += self.Sp_nn_v[sl]

        self.Sp[sl][inactive] = 0.0

    def _solve_scalar(self, phi, alpha):
        sl = self.sl
        nx, ny = self.nx, self.ny
        nix, niy = nx - 2, ny - 2
        active = self.active_in

        phi_old = phi.copy()
        aP_ur = (self.aP[sl] / alpha).copy()
        aE_in = self.aE[sl].copy()
        aW_in = self.aW[sl].copy()
        aN_in = self.aN[sl].copy()
        aS_in = self.aS[sl].copy()

        S_ur = self.Sp[sl] + (1.0 - alpha) / alpha * self.aP[sl] * phi_old[sl]
        rhs = S_ur.reshape((nix, niy)).copy()

        # 处理四个外框边界 (i=0, i=nx-1, j=0, j=ny-1) 的隐式零梯度或 Dirichlet 移项
        # 西外边界 (i=0)
        out_w = np.zeros(niy, dtype=bool)
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'pressure_outlet':
                out_w |= self.mesh.bc_masks[bname][0, 1:-1]
        aP_ur[0, out_w] -= aW_in[0, out_w]
        rhs[0, ~out_w]  += aW_in[0, ~out_w] * phi[0, 1:-1][~out_w]

        # 东外边界 (i=nx-1)
        out_e = np.zeros(niy, dtype=bool)
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'pressure_outlet':
                out_e |= self.mesh.bc_masks[bname][-1, 1:-1]
        aP_ur[-1, out_e] -= aE_in[-1, out_e]
        rhs[-1, ~out_e]  += aE_in[-1, ~out_e] * phi[nx - 1, 1:-1][~out_e]

        # 南外边界 (j=0)
        out_s = np.zeros(nix, dtype=bool)
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'pressure_outlet':
                out_s |= self.mesh.bc_masks[bname][1:-1, 0]
        aP_ur[out_s, 0] -= aS_in[out_s, 0]
        rhs[~out_s, 0]  += aS_in[~out_s, 0] * phi[1:-1, 0][~out_s]

        # 北外边界 (j=ny-1)
        out_n = np.zeros(nix, dtype=bool)
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'pressure_outlet':
                out_n |= self.mesh.bc_masks[bname][1:-1, -1]
        aP_ur[out_n, -1] -= aN_in[out_n, -1]
        rhs[~out_n, -1]  += aN_in[~out_n, -1] * phi[1:-1, ny - 1][~out_n]

        # 对位于 (1:-1, 1:-1) 内部切片中的非流体点 (如台阶壁面与台阶内部固体区) 实施对角元锁定
        inactive = ~active
        aP_ur[inactive] = 1.0
        aE_in[inactive] = 0.0
        aW_in[inactive] = 0.0
        aN_in[inactive] = 0.0
        aS_in[inactive] = 0.0
        rhs[inactive]   = phi[sl][inactive]

        A = build_penta_matrix(aP_ur, aE_in, aW_in, aN_in, aS_in)
        phi_internal = spsolve(A, rhs.ravel())
        phi[sl] = phi_internal.reshape((nix, niy), order='C')

    def _rc_face_velocities(self, alpha_u):
        nx, ny = self.nx, self.ny
        m = self.mesh

        d = np.zeros((nx, ny))
        d[1:-1, 1:-1] = np.where(self.active_in, alpha_u / self.aP[1:-1, 1:-1], 0.0)

        # 将 d 零梯度延拓到有效边界
        for bname in self.bc_cfg.keys():
            bi, bj, ni, nj = self.neumann_pairs[bname]
            d[bi, bj] = d[ni, nj]
        d[m.solid_mask] = 0.0

        Gx, Gy = self._gradient(self.p)

        # 东界面 e (位于 i 与 i+1 之间)
        valid_e = m.valid_mask[:-1, :] & m.valid_mask[1:, :]
        d_bar_e = np.where(valid_e, 0.5 * (d[:-1, :] + d[1:, :]), 0.0)
        Gx_bar_e = 0.5 * (Gx[:-1, :] + Gx[1:, :])
        Gx_compact_e = (self.p[1:, :] - self.p[:-1, :]) / m.hx
        u_e = np.where(
            valid_e,
            0.5 * (self.u[:-1, :] + self.u[1:, :]) + d_bar_e * (Gx_bar_e - Gx_compact_e),
            0.0
        )

        # 北界面 n (位于 j 与 j+1 之间)
        valid_n = m.valid_mask[:, :-1] & m.valid_mask[:, 1:]
        d_bar_n = np.where(valid_n, 0.5 * (d[:, :-1] + d[:, 1:]), 0.0)
        Gy_bar_n = 0.5 * (Gy[:, :-1] + Gy[:, 1:])
        Gy_compact_n = (self.p[:, 1:] - self.p[:, :-1]) / m.hy
        v_n = np.where(
            valid_n,
            0.5 * (self.v[:, :-1] + self.v[:, 1:]) + d_bar_n * (Gy_bar_n - Gy_compact_n),
            0.0
        )

        return u_e, v_n, d

    def _solve_pressure_correction(self, alpha_u):
        nx, ny = self.nx, self.ny
        nix, niy = nx - 2, ny - 2
        sl = self.sl
        m = self.mesh
        active = self.active_in

        u_e, v_n, d = self._rc_face_velocities(alpha_u)

        div = np.zeros((nx, ny))
        div[sl] = (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) / m.dx_P + \
                  (v_n[1:-1, 1:] - v_n[1:-1, :-1]) / m.dy_P
        div[sl][~active] = 0.0

        if self.Sp_nn_div is not None:
            b = -(div[sl] - self.Sp_nn_div[sl]).copy()
        else:
            b = -div[sl].copy()
        b[~active] = 0.0

        mass_resid = float(np.max(np.abs(b[active])))

        # 识别压力出口掩码，除压力出口外的所有边界 (wall/inlet/solid) 对 p' 均为零法向通量 (Neumann)
        p_out_mask = np.zeros((nx, ny), dtype=bool)
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'pressure_outlet':
                p_out_mask |= m.bc_masks[bname]

        # 只有当邻居是流体内部节点，或者是压力出口节点时，才存在非零压力修正通量连接
        conn_p = m.fluid_internal | p_out_mask

        aE_pc = np.where(active & conn_p[2:, 1:-1],  0.5 * (d[1:-1, 1:-1] + d[2:, 1:-1])  / (m.dx_P * m.dx_E), 0.0)
        aW_pc = np.where(active & conn_p[:-2, 1:-1], 0.5 * (d[:-2, 1:-1]  + d[1:-1, 1:-1]) / (m.dx_P * m.dx_W), 0.0)
        aN_pc = np.where(active & conn_p[1:-1, 2:],  0.5 * (d[1:-1, 1:-1] + d[1:-1, 2:])  / (m.dy_P * m.dy_N), 0.0)
        aS_pc = np.where(active & conn_p[1:-1, :-2], 0.5 * (d[1:-1, :-2]  + d[1:-1, 1:-1]) / (m.dy_P * m.dy_S), 0.0)
        aP_pc = aE_pc + aW_pc + aN_pc + aS_pc

        # 非活动节点锁死为 p' = 0
        inactive = ~active
        aP_pc[inactive] = 1.0
        aE_pc[inactive] = aW_pc[inactive] = aN_pc[inactive] = aS_pc[inactive] = 0.0
        b[inactive] = 0.0

        if not self.has_pressure_outlet:
            ri, rj = self.ref_p_ij
            aP_pc[ri, rj] = 1.0
            aE_pc[ri, rj] = aW_pc[ri, rj] = aN_pc[ri, rj] = aS_pc[ri, rj] = 0.0
            b[ri, rj] = 0.0

        A_p = build_penta_matrix(aP_pc, aE_pc, aW_pc, aN_pc, aS_pc)
        pp_internal = spsolve(A_p, b.ravel())

        pp = np.zeros((nx, ny))
        pp[sl] = pp_internal.reshape((nix, niy), order='C')
        self._bc_p_prime(pp)

        return pp, d, mass_resid

    def _correct_fields(self, pp, d, alpha_p):
        sl = self.sl
        active = self.active_in

        self.p[sl][active] += alpha_p * pp[sl][active]
        Gx_pp, Gy_pp = self._gradient(pp)
        self.u[sl][active] -= d[sl][active] * Gx_pp[sl][active]
        self.v[sl][active] -= d[sl][active] * Gy_pp[sl][active]

        self._bc_uv()
        self._bc_p()

    def _rel_change(self, phi_new, phi_old):
        mask = self.mesh.fluid_internal
        denom = np.max(np.abs(phi_old[mask]))
        if denom < 1e-30:
            return float(np.mean(np.abs(phi_new[mask] - phi_old[mask])))
        return float(np.mean(np.abs(phi_new[mask] - phi_old[mask])) / denom)

    def solve(self, verbose=True):
        s_cfg = self.config.get('solver', {})
        max_iter = int(s_cfg.get('max_iter', 3000))
        tol      = float(s_cfg.get('tol', 1e-6))
        alpha_u  = float(s_cfg.get('alpha_u', 0.65))
        alpha_v  = float(s_cfg.get('alpha_v', 0.65))
        alpha_p  = float(s_cfg.get('alpha_p', 0.25))

        topo_str = "多块台阶结构化网格" if self.mesh.has_obstacle else (
            "均匀正交网格" if self.mesh.is_uniform else "非均匀拉伸网格"
        )
        if verbose:
            print("=" * 72)
            print(f"  通用 SIMPLE 求解器 | 案例: {self.config.get('case_name', 'Unnamed')} "
                  f"| 外包维度: {self.nx}x{self.ny} ({topo_str})")
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
                u_max = np.max(np.abs(self.u[self.mesh.valid_mask]))
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
# 模块 4: 掩码感知的完备离散双残差计算器
# ============================================================
def compute_discrete_residual(solver: UniversalSIMPLE, u_s, v_s, p_s):
    """
    将密网格采样流场 (u_s, v_s, p_s) 载入粗网格求解器模板，
    严格同步粗网格边界条件后计算仅在流体内部节点 (active_in) 上的双残差。
    """
    alpha_u = float(solver.config.get('solver', {}).get('alpha_u', 0.65))
    m = solver.mesh
    sl = solver.sl
    active = solver.active_in

    solver.u = u_s.copy()
    solver.v = v_s.copy()
    solver.p = p_s.copy()
    solver._bc_uv()
    solver._bc_p()

    # 1. 组装粗网格动量方程系数
    solver._assemble_momentum('u')
    Gx, Gy = solver._gradient(solver.p)
    Sp_u = -Gx[sl]
    Sp_v = -Gy[sl]

    # 对存在压力出口的外边界，修正有效主对角元 (与 _solve_scalar 严格一致)
    aP_eff = solver.aP[sl].copy()
    for bname, bc in solver.bc_cfg.items():
        if bc['type'] == 'pressure_outlet':
            bmask = m.bc_masks[bname]
            aP_eff[0,  bmask[0, 1:-1]]  -= solver.aW[sl][0,  bmask[0, 1:-1]]
            aP_eff[-1, bmask[-1, 1:-1]] -= solver.aE[sl][-1, bmask[-1, 1:-1]]
            aP_eff[bmask[1:-1, 0],  0]  -= solver.aS[sl][bmask[1:-1, 0],  0]
            aP_eff[bmask[1:-1, -1], -1] -= solver.aN[sl][bmask[1:-1, -1], -1]

    # 2. 计算动量离散残差 (非流体内部点置 0)
    Res_u = np.where(
        active,
        aP_eff * solver.u[sl]
        - solver.aE[sl] * solver.u[2:, 1:-1]
        - solver.aW[sl] * solver.u[:-2, 1:-1]
        - solver.aN[sl] * solver.u[1:-1, 2:]
        - solver.aS[sl] * solver.u[1:-1, :-2]
        - Sp_u,
        0.0
    )

    Res_v = np.where(
        active,
        aP_eff * solver.v[sl]
        - solver.aE[sl] * solver.v[2:, 1:-1]
        - solver.aW[sl] * solver.v[:-2, 1:-1]
        - solver.aN[sl] * solver.v[1:-1, 2:]
        - solver.aS[sl] * solver.v[1:-1, :-2]
        - Sp_v,
        0.0
    )

    # 3. 计算粗网格 Rhie-Chow 界面速度下的离散散度残差
    u_e, v_n, _ = solver._rc_face_velocities(alpha_u)
    Res_div = np.where(
        active,
        (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) / m.dx_P +
        (v_n[1:-1, 1:] - v_n[1:-1, :-1]) / m.dy_P,
        0.0
    )

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
        active = solver.active_in
        X_in = (solver.X[1:-1, 1:-1][active] - solver.x[0]) / solver.Lx
        Y_in = (solver.Y[1:-1, 1:-1][active] - solver.y[0]) / solver.Ly
        X_train = torch.tensor(np.column_stack([X_in, Y_in]), dtype=torch.float32, device=self.device)

        targets = {'u': res_u[active], 'v': res_v[active], 'div': res_div[active]}
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
        print(f"  训练 FourierMLP 代理模型 | 有效流体样本数: {len(X_train)} | 设备: {self.device}")
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
                # 仅在有效流体内部节点保留源项补偿
                field[~solver.mesh.fluid_internal] = 0.0
                preds[k] = field

        if not solver.has_pressure_outlet:
            sl = solver.sl
            active = solver.active_in
            vol = solver.mesh.cell_vol[active]
            vol_weighted_mean = np.sum(preds['div'][sl][active] * vol) / np.sum(vol)
            preds['div'][sl][active] -= vol_weighted_mean

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
    print(f"  密网格 ({dense_msh_path})  -> 外包维度: {mesh_dense.nx}x{mesh_dense.ny}, "
          f"有效节点数: {int(mesh_dense.valid_mask.sum())}, 物理边界: {list(mesh_dense.bc_masks.keys())}")
    print(f"  粗网格 ({coarse_msh_path}) -> 外包维度: {mesh_coarse.nx}x{mesh_coarse.ny}, "
          f"有效节点数: {int(mesh_coarse.valid_mask.sum())}, 内部流体节点数: {int(mesh_coarse.fluid_internal.sum())}")

    # 2. 密网格求解 (高精度参考解)
    solver_dense = UniversalSIMPLE(mesh_dense, config)
    solver_dense.solve(verbose=True)

    # 3. 将密网格流场插值采样到粗网格节点上
    pts_coarse = np.column_stack([mesh_coarse.X.ravel(), mesh_coarse.Y.ravel()])
    interp_u = RegularGridInterpolator((mesh_dense.x, mesh_dense.y), solver_dense.u, method='linear')
    interp_v = RegularGridInterpolator((mesh_dense.x, mesh_dense.y), solver_dense.v, method='linear')
    interp_p = RegularGridInterpolator((mesh_dense.x, mesh_dense.y), solver_dense.p, method='linear')

    u_raw = interp_u(pts_coarse).reshape(mesh_coarse.nx, mesh_coarse.ny)
    v_raw = interp_v(pts_coarse).reshape(mesh_coarse.nx, mesh_coarse.ny)
    p_raw = interp_p(pts_coarse).reshape(mesh_coarse.nx, mesh_coarse.ny)

    # 4. 计算粗网格自洽双残差 (Res_u, Res_v, Res_div)
    solver_template = UniversalSIMPLE(mesh_coarse, config)
    res_u, res_v, res_div, u_sampled, v_sampled, p_sampled = compute_discrete_residual(
        solver_template, u_raw, v_raw, p_raw
    )
    act = solver_template.active_in
    print(f"\n  [粗网格流体内部节点离散残差统计]")
    print(f"  Res_u   范围: [{res_u[act].min():.4e}, {res_u[act].max():.4e}]")
    print(f"  Res_v   范围: [{res_v[act].min():.4e}, {res_v[act].max():.4e}]")
    print(f"  Res_div 范围: [{res_div[act].min():.4e}, {res_div[act].max():.4e}]")

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

    # 8. 评估内部流体节点精度提升
    fl_mask = mesh_coarse.fluid_internal
    err_u_orig_max = np.max(np.abs(u_sampled[fl_mask] - solver_coarse_orig.u[fl_mask]))
    err_u_nn_max   = np.max(np.abs(u_sampled[fl_mask] - solver_coarse_nn.u[fl_mask]))
    rmse_u_orig    = np.sqrt(np.mean((u_sampled[fl_mask] - solver_coarse_orig.u[fl_mask]) ** 2))
    rmse_u_nn      = np.sqrt(np.mean((u_sampled[fl_mask] - solver_coarse_nn.u[fl_mask]) ** 2))

    imp_max  = (1.0 - err_u_nn_max / (err_u_orig_max + 1e-30)) * 100
    imp_rmse = (1.0 - rmse_u_nn / (rmse_u_orig + 1e-30)) * 100

    print("\n" + "=" * 72)
    print(f"  [{case_name}] 内部流场节点速度 u 误差对比汇总:")
    print(f"  Max Error : 原粗网格 = {err_u_orig_max:.4e} -> NN修正后 = {err_u_nn_max:.4e} (降低 {imp_max:.2f}%)")
    print(f"  RMSE      : 原粗网格 = {rmse_u_orig:.4e} -> NN修正后 = {rmse_u_nn:.4e} (降低 {imp_rmse:.2f}%)")
    print("=" * 72)

    if plot_results:
        fig, axes = plt.subplots(3, 1, figsize=(12, 9.5))
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
            # 将固体台阶区域掩码设为 NaN，使绘图时自动留白显示台阶几何
            vel_plot = np.where(sv.mesh.valid_mask, vel, np.nan)
            cf = ax.contourf(sv.X.T, sv.Y.T, vel_plot.T, levels=25, cmap='viridis')
            if sv.mesh.has_obstacle:
                ax.set_facecolor('#dcdcdc')
            plt.colorbar(cf, ax=ax)
            ax.set_title(title)
            ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(f"./{case_name}_comparison.png", dpi=150)
        print(f"  图像已保存至: ./{case_name}_comparison.png")
        plt.show()

    return solver_dense, solver_coarse_orig, solver_coarse_nn


if __name__ == '__main__':
    config_backward = {
        "case_name": "backward",
        "meshes": {
            "dense_msh": "./meshes/back_fine.msh",
            "coarse_msh": "./meshes/back_coarse.msh"
        },
        "fluid": {
            "rho": 1.0,
            "mu": 0.01
        },
        "initial_conditions": {
            "u": 0.0, "v": 0.0, "p": 0.0
        },
        "boundary_conditions": {
            "wall":   {"type": "wall", "u": 0.0, "v": 0.0},
            "outlet": {"type": "pressure_outlet", "p": 0.0},
            "inlet":  {"type": "inlet", "u": 1.0, "v": 0.0}
        },
        "solver": {
            "max_iter": 2000, "tol": 1e-7,
            "alpha_u": 0.7, "alpha_v": 0.7, "alpha_p": 0.3
        },
        "nn_config": {
            "epochs": 4000, "lr": 3e-3, "hidden_dim": 96, "num_freqs": 6
        }
    }

    run_universal_pipeline(config_backward, plot_results=True)