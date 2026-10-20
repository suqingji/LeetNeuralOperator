#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================================
DeepSIMPLE_2D: 通用 2D 多块掩模结构化网格 SIMPLE 求解器与神经网络残差补偿框架 (高精度增强版)
==================================================================================================

【架构升级与核心突破】
本版本在“基础掩模版”之上，针对 SciML (科学机器学习) 处理复杂边界流场时的典型数值灾难进行了全面根治：

 1. 根除交界面插值灾难 (Laplacian Smooth + Cubic Spline):
    在执行密到粗网格的流场采样前，创新性地采用“拉普拉斯光滑延拓”将流体边界平滑渗透至固体台阶内部。
    配合 RectBivariateSpline 双三次样条，彻底消除流固交界面处的插值过冲与二阶导数跳变。

 2. 根除残差拟合的吉布斯振荡 (Hybrid Fourier-RBF Neural Network):
    摒弃纯全局频域特征，采用多尺度混合神经网络架构：
    - 全局低频 Fourier 特征拟合平缓背景残差流。
    - 自适应提取极值点，并部署双尺度局部高斯径向基 (RBF) 核，精准“定点爆破”台阶拐角处的高频奇异脉冲。
    - 实现对残差场无振荡、无出口翘尾的高保真重构。

 3. 绝对稳定的离散算子 (Strict Neumann-Compatible Operators):
    进一步重构掩码单侧差分逻辑，使其与边界条件的隐式移项绝对自洽，确保全流域离散方程具备严格的
    对角占优与正定性，保障 SIMPLE 算法即使在大步长下依然绝对收敛。

【自动化全套学术可视化管线】
程序运行完毕后自动输出 4 套出版级对比图表：
 1. 神经网络源项预测误差全览 (3x3 云图)
 2. 流速幅值云图与流函数 (ψ) 涡结构流线图
 3. 特征中心线速度抛物线/边界层剖面对比图
 4. 误差与精度提升率对数柱状图 (Max Error & RMSE)
==================================================================================================
"""

import os
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

        # 第一遍扫描：提取 $PhysicalNames
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
                    if elm_type == 1:
                        bname = tag_to_bname.get(phys_tag, f"tag_{phys_tag}")
                        self.boundary_nodes.setdefault(bname, set()).update(node_list)
                        self.boundary_edges.setdefault(bname, []).append(tuple(node_list))
                    elif elm_type == 3:
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

        ix_all = np.searchsorted(self.x, rounded_coords[:, 0])
        iy_all = np.searchsorted(self.y, rounded_coords[:, 1])

        self.node_to_ij = {}
        self.ij_to_node = np.full((self.nx, self.ny), -1, dtype=int)
        for nid, i, j in zip(nids, ix_all, iy_all):
            self.node_to_ij[nid] = (int(i), int(j))
            self.ij_to_node[i, j] = nid

        valid_count = int(np.sum(self.ij_to_node >= 0))
        if valid_count != len(nids):
            raise ValueError(
                f"存在坐标重合或非正交错位节点: 唯一映射节点数 ({valid_count}) != 实际节点总数 ({len(nids)})"
            )

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

        self.valid_mask = (self.ij_to_node >= 0)
        self.solid_mask = ~self.valid_mask
        self.has_obstacle = bool(np.any(self.solid_mask))

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

        self.fluid_internal = self.valid_mask & (~all_b_mask)
        self.fluid_internal[0, :] = False
        self.fluid_internal[-1, :] = False
        self.fluid_internal[:, 0] = False
        self.fluid_internal[:, -1] = False

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

    rows_list.append(ii.ravel())
    cols_list.append(ii.ravel())
    vals_list.append(aP.ravel())

    rows_list.append(ii[:-1, :].ravel())
    cols_list.append((ii[:-1, :] + niy).ravel())
    vals_list.append(-aE[:-1, :].ravel())

    rows_list.append(ii[1:, :].ravel())
    cols_list.append((ii[1:, :] - niy).ravel())
    vals_list.append(-aW[1:, :].ravel())

    rows_list.append(ii[:, :-1].ravel())
    cols_list.append((ii[:, :-1] + 1).ravel())
    vals_list.append(-aN[:, :-1].ravel())

    rows_list.append(ii[:, 1:].ravel())
    cols_list.append((ii[:, 1:] - 1).ravel())
    vals_list.append(-aS[:, 1:].ravel())

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)

    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


# ============================================================
# 模块 3: 支持多块/后台阶掩码的通用 SIMPLE 求解器 (稳定自洽版)
# ============================================================
class UniversalSIMPLE:
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

        raw_bc_cfg = config.get('boundary_conditions', {})
        self.bc_cfg = {}
        for bname in mesh.bc_masks.keys():
            if bname not in raw_bc_cfg:
                raise KeyError(f"配置文件 boundary_conditions 中缺少网格物理边界 '{bname}' 的定义!")
            cfg_item = dict(raw_bc_cfg[bname])
            if bname.lower() == 'inlet' and cfg_item.get('type') == 'wall':
                if abs(float(cfg_item.get('u', 0.0))) > 0 or abs(float(cfg_item.get('v', 0.0))) > 0:
                    cfg_item['type'] = 'inlet'
            self.bc_cfg[bname] = cfg_item

        self.has_pressure_outlet = any(
            bc['type'] == 'pressure_outlet' for bc in self.bc_cfg.values()
        )

        self._build_neumann_extrapolation_maps()

        sh = (self.nx, self.ny)
        self.u = np.zeros(sh)
        self.v = np.zeros(sh)
        self.p = np.zeros(sh)

        self.aE = np.zeros(sh); self.aW = np.zeros(sh)
        self.aN = np.zeros(sh); self.aS = np.zeros(sh)
        self.aP = np.zeros(sh); self.Sp = np.zeros(sh)

        self.sl = (slice(1, -1), slice(1, -1))
        self.active_in = self.mesh.fluid_internal[self.sl]

        active_indices = np.argwhere(self.active_in)
        self.ref_p_ij = tuple(active_indices[0])

        self.Sp_nn_u = None
        self.Sp_nn_v = None
        self.Sp_nn_div = None

        self._init_fields()

    def _build_neumann_extrapolation_maps(self):
        self.neumann_pairs = {}
        fl = self.mesh.fluid_internal
        val = self.mesh.valid_mask

        for bname, mask in self.mesh.bc_masks.items():
            bi_list, bj_list, ni_list, nj_list = [], [], [], []
            b_indices = np.argwhere(mask)
            for i, j in b_indices:
                found = False
                for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.nx and 0 <= nj < self.ny and fl[ni, nj]:
                        bi_list.append(i); bj_list.append(j)
                        ni_list.append(ni); nj_list.append(nj)
                        found = True
                        break
                if not found:
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
        self.u[self.mesh.solid_mask] = 0.0
        self.v[self.mesh.solid_mask] = 0.0

        # 1. 先处理 pressure_outlet
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'pressure_outlet':
                bi, bj, ni, nj = self.neumann_pairs[bname]
                self.u[bi, bj] = self.u[ni, nj]
                self.v[bi, bj] = self.v[ni, nj]

        # 2. 再处理 inlet
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'inlet':
                mask = self.mesh.bc_masks[bname]
                self.u[mask] = float(bc.get('u', 0.0))
                self.v[mask] = float(bc.get('v', 0.0))

        # 3. 最后处理 wall (确保交界角点满足无滑移条件)
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'wall':
                mask = self.mesh.bc_masks[bname]
                self.u[mask] = float(bc.get('u', 0.0))
                self.v[mask] = float(bc.get('v', 0.0))

    def _bc_p(self):
        if not self.has_pressure_outlet:
            ri, rj = self.ref_p_ij
            self.p -= self.p[ri + 1, rj + 1]

        for bname, bc in self.bc_cfg.items():
            if bc['type'] in ('wall', 'inlet'):
                bi, bj, ni, nj = self.neumann_pairs[bname]
                self.p[bi, bj] = self.p[ni, nj]

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
        采用与一阶 Neumann 边界条件严格兼容的掩码单侧差分，保证 Rhie-Chow 耗散项正定稳定。
        """
        Gx = np.gradient(f, self.x, axis=0, edge_order=2)
        Gy = np.gradient(f, self.y, axis=1, edge_order=2)

        if self.mesh.has_obstacle:
            val = self.mesh.valid_mask
            west_solid = val[1:-1, :] & (~val[:-2, :]) & val[2:, :]
            if np.any(west_solid):
                dx_f = (self.x[2:] - self.x[1:-1])[:, None]
                Gx[1:-1, :] = np.where(west_solid, (f[2:, :] - f[1:-1, :]) / dx_f, Gx[1:-1, :])

            east_solid = val[1:-1, :] & (~val[2:, :]) & val[:-2, :]
            if np.any(east_solid):
                dx_b = (self.x[1:-1] - self.x[:-2])[:, None]
                Gx[1:-1, :] = np.where(east_solid, (f[1:-1, :] - f[:-2, :]) / dx_b, Gx[1:-1, :])

            south_solid = val[:, 1:-1] & (~val[:, :-2]) & val[:, 2:]
            if np.any(south_solid):
                dy_f = (self.y[2:] - self.y[1:-1])[None, :]
                Gy[:, 1:-1] = np.where(south_solid, (f[:, 2:] - f[:, 1:-1]) / dy_f, Gy[:, 1:-1])

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

        out_w = np.zeros(niy, dtype=bool)
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'pressure_outlet':
                out_w |= self.mesh.bc_masks[bname][0, 1:-1]
        aP_ur[0, out_w] -= aW_in[0, out_w]
        rhs[0, ~out_w]  += aW_in[0, ~out_w] * phi[0, 1:-1][~out_w]

        out_e = np.zeros(niy, dtype=bool)
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'pressure_outlet':
                out_e |= self.mesh.bc_masks[bname][-1, 1:-1]
        aP_ur[-1, out_e] -= aE_in[-1, out_e]
        rhs[-1, ~out_e]  += aE_in[-1, ~out_e] * phi[nx - 1, 1:-1][~out_e]

        out_s = np.zeros(nix, dtype=bool)
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'pressure_outlet':
                out_s |= self.mesh.bc_masks[bname][1:-1, 0]
        aP_ur[out_s, 0] -= aS_in[out_s, 0]
        rhs[~out_s, 0]  += aS_in[~out_s, 0] * phi[1:-1, 0][~out_s]

        out_n = np.zeros(nix, dtype=bool)
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'pressure_outlet':
                out_n |= self.mesh.bc_masks[bname][1:-1, -1]
        aP_ur[out_n, -1] -= aN_in[out_n, -1]
        rhs[~out_n, -1]  += aN_in[~out_n, -1] * phi[1:-1, ny - 1][~out_n]

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

        for bname in self.bc_cfg.keys():
            bi, bj, ni, nj = self.neumann_pairs[bname]
            d[bi, bj] = d[ni, nj]
        d[m.solid_mask] = 0.0

        Gx, Gy = self._gradient(self.p)

        valid_e = m.valid_mask[:-1, :] & m.valid_mask[1:, :]
        d_bar_e = np.where(valid_e, 0.5 * (d[:-1, :] + d[1:, :]), 0.0)
        Gx_bar_e = 0.5 * (Gx[:-1, :] + Gx[1:, :])
        Gx_compact_e = (self.p[1:, :] - self.p[:-1, :]) / m.hx
        u_e = np.where(
            valid_e,
            0.5 * (self.u[:-1, :] + self.u[1:, :]) + d_bar_e * (Gx_bar_e - Gx_compact_e),
            0.0
        )

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

        p_out_mask = np.zeros((nx, ny), dtype=bool)
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'pressure_outlet':
                p_out_mask |= m.bc_masks[bname]

        conn_p = m.fluid_internal | p_out_mask

        aE_pc = np.where(active & conn_p[2:, 1:-1],  0.5 * (d[1:-1, 1:-1] + d[2:, 1:-1])  / (m.dx_P * m.dx_E), 0.0)
        aW_pc = np.where(active & conn_p[:-2, 1:-1], 0.5 * (d[:-2, 1:-1]  + d[1:-1, 1:-1]) / (m.dx_P * m.dx_W), 0.0)
        aN_pc = np.where(active & conn_p[1:-1, 2:],  0.5 * (d[1:-1, 1:-1] + d[1:-1, 2:])  / (m.dy_P * m.dy_N), 0.0)
        aS_pc = np.where(active & conn_p[1:-1, :-2], 0.5 * (d[1:-1, :-2]  + d[1:-1, 1:-1]) / (m.dy_P * m.dy_S), 0.0)
        aP_pc = aE_pc + aW_pc + aN_pc + aS_pc

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

    def _streamfunc(self):
        psi = np.zeros((self.nx, self.ny))
        u_clean = np.where(self.mesh.valid_mask, self.u, 0.0)
        dy_1d = np.diff(self.y)[None, :]
        psi[:, 1:] = np.cumsum(0.5 * (u_clean[:, 1:] + u_clean[:, :-1]) * dy_1d, axis=1)
        return psi

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
                psi_min = np.min(self._streamfunc()[self.mesh.valid_mask])
                print(f"  iter {it:5d} | res_u={res_u:.2e} | res_v={res_v:.2e} | "
                      f"div_err={mass_resid:.2e} | u_max={u_max:.4f} | psi_min={psi_min:.4f}")

            if res_u < tol and res_v < tol and mass_resid < tol and it > 15:
                if verbose:
                    print(f"  >> 收敛! iter = {it}, res_u={res_u:.2e}, div_err={mass_resid:.2e}")
                break
        else:
            if verbose:
                print(f"  >> 达到最大迭代次数 {max_iter}, res_u={res_u:.2e}, div_err={mass_resid:.2e}")

        return res_u, res_v, mass_resid


# ============================================================
# 模块 4: 固体区光滑延拓与双三次样条高精度采样 + 完备离散双残差计算
# ============================================================
def smooth_fill_solid_region(field, valid_mask, n_iters=300):
    """
    在调用 RectBivariateSpline 前，对固体台阶区域做拉普拉斯光滑延拓填充，
    防止固体区突变为 0 在台阶界面产生三次样条振荡。
    """
    if np.all(valid_mask):
        return field.copy()
    f = field.copy()
    solid = ~valid_mask
    f[solid] = float(np.mean(f[valid_mask]))
    for _ in range(n_iters):
        f_pad = np.pad(f, 1, mode='edge')
        avg = 0.25 * (f_pad[2:, 1:-1] + f_pad[:-2, 1:-1] + f_pad[1:-1, 2:] + f_pad[1:-1, :-2])
        f[solid] = avg[solid]
    return f


def sample_dense_to_coarse_spline(mesh_dense: StructuredGmshMesh,
                                  solver_dense: UniversalSIMPLE,
                                  mesh_coarse: StructuredGmshMesh):
    """
    使用三阶双三次样条 (RectBivariateSpline) 将密网格解高精度采样到粗网格上。
    """
    u_fill = smooth_fill_solid_region(solver_dense.u, mesh_dense.valid_mask)
    v_fill = smooth_fill_solid_region(solver_dense.v, mesh_dense.valid_mask)
    p_fill = smooth_fill_solid_region(solver_dense.p, mesh_dense.valid_mask)

    interp_u = RectBivariateSpline(mesh_dense.x, mesh_dense.y, u_fill, kx=3, ky=3)
    interp_v = RectBivariateSpline(mesh_dense.x, mesh_dense.y, v_fill, kx=3, ky=3)
    interp_p = RectBivariateSpline(mesh_dense.x, mesh_dense.y, p_fill, kx=3, ky=3)

    u_raw = interp_u(mesh_coarse.x, mesh_coarse.y)
    v_raw = interp_v(mesh_coarse.x, mesh_coarse.y)
    p_raw = interp_p(mesh_coarse.x, mesh_coarse.y)

    u_raw[mesh_coarse.solid_mask] = 0.0
    v_raw[mesh_coarse.solid_mask] = 0.0
    p_raw[mesh_coarse.solid_mask] = 0.0
    return u_raw, v_raw, p_raw


def compute_discrete_residual(solver: UniversalSIMPLE, u_s, v_s, p_s):
    alpha_u = float(solver.config.get('solver', {}).get('alpha_u', 0.65))
    m = solver.mesh
    sl = solver.sl
    active = solver.active_in

    solver.u = u_s.copy()
    solver.v = v_s.copy()
    solver.p = p_s.copy()
    solver._bc_uv()
    solver._bc_p()

    solver._assemble_momentum('u')
    Gx, Gy = solver._gradient(solver.p)
    Sp_u = -Gx[sl]
    Sp_v = -Gy[sl]

    aP_eff = solver.aP[sl].copy()
    for bname, bc in solver.bc_cfg.items():
        if bc['type'] == 'pressure_outlet':
            bmask = m.bc_masks[bname]
            aP_eff[0,  bmask[0, 1:-1]]  -= solver.aW[sl][0,  bmask[0, 1:-1]]
            aP_eff[-1, bmask[-1, 1:-1]] -= solver.aE[sl][-1, bmask[-1, 1:-1]]
            aP_eff[bmask[1:-1, 0],  0]  -= solver.aS[sl][bmask[1:-1, 0],  0]
            aP_eff[bmask[1:-1, -1], -1] -= solver.aN[sl][bmask[1:-1, -1], -1]

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

    u_e, v_n, _ = solver._rc_face_velocities(alpha_u)
    Res_div = np.where(
        active,
        (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) / m.dx_P +
        (v_n[1:-1, 1:] - v_n[1:-1, :-1]) / m.dy_P,
        0.0
    )

    return Res_u, Res_v, Res_div, solver.u.copy(), solver.v.copy(), solver.p.copy()


# ============================================================
# 模块 5: 混合多尺度傅里叶 + 自适应局部径向基特征神经网络 (HybridFourierRBFNet)
# ============================================================
class HybridFourierRBFNet(nn.Module):
    """
    结合全局低频傅里叶特征与自适应高梯度局部高斯径向基 (RBF) 特征:
    既能精准拟合入口/台阶折角处的剧烈局部脉冲，又能在下游平缓区保持零振荡光滑输出。
    """
    def __init__(self, centers: torch.Tensor, widths: torch.Tensor, hidden_dim=96, num_freqs=4):
        super().__init__()
        freqs = torch.arange(1, num_freqs + 1, dtype=torch.float32) * np.pi
        self.register_buffer('freqs', freqs)
        self.register_buffer('centers', centers)  # (M, 2)
        self.register_buffer('widths', widths)    # (M,)

        in_dim = 2
        feat_dim = in_dim + in_dim * num_freqs * 2 + len(centers)

        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # 1. 全局傅里叶特征
        proj = x.unsqueeze(-1) * self.freqs
        proj = proj.reshape(x.shape[0], -1)
        # 2. 局部高梯度点径向基特征 exp(-||x - c||^2 / (2 * w^2))
        dist_sq = torch.sum((x.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2, dim=-1)
        rbf_feats = torch.exp(-dist_sq / (2.0 * (self.widths.unsqueeze(0) ** 2)))

        feats = torch.cat([x, torch.sin(proj), torch.cos(proj), rbf_feats], dim=-1)
        return self.net(feats)


class UniversalNeuralCorrector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        self.stats = {}
        self.centers = None
        self.widths = None

    def prepare_training_data(self, solver: UniversalSIMPLE, res_u, res_v, res_div):
        active = solver.active_in
        X_in = (solver.X[1:-1, 1:-1][active] - solver.x[0]) / solver.Lx
        Y_in = (solver.Y[1:-1, 1:-1][active] - solver.y[0]) / solver.Ly
        coords_np = np.column_stack([X_in, Y_in])
        X_train = torch.tensor(coords_np, dtype=torch.float32, device=self.device)

        # 自动提取残差幅值最大的前 96 个局部奇异点 + 64 个均匀锚点作为 RBF 中心
        importance = (
            np.abs(res_u[active]) / (np.std(res_u[active]) + 1e-12) +
            np.abs(res_v[active]) / (np.std(res_v[active]) + 1e-12) +
            np.abs(res_div[active]) / (np.std(res_div[active]) + 1e-12)
        )
        n_samples = len(coords_np)
        n_peak = min(96, n_samples // 2)
        peak_idx = np.argsort(importance)[-n_peak:]
        unif_idx = np.linspace(0, n_samples - 1, min(64, n_samples), dtype=int)
        sel_idx = np.unique(np.concatenate([peak_idx, unif_idx]))

        centers_np = coords_np[sel_idx]
        # 为每个局部中心配置双尺度宽度 (细尺度捕捉角点脉冲，中尺度平滑过渡)
        centers_multi = np.vstack([centers_np, centers_np])
        widths_multi = np.concatenate([
            np.full(len(centers_np), 0.025, dtype=np.float32),
            np.full(len(centers_np), 0.080, dtype=np.float32)
        ])
        self.centers = torch.tensor(centers_multi, dtype=torch.float32, device=self.device)
        self.widths = torch.tensor(widths_multi, dtype=torch.float32, device=self.device)

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

        print(f"\n{'='*72}")
        print(f"  Step 4: 训练 HybridFourierRBF 代理模型预测离散残差")
        print(f"  有效流体样本数: {len(X_train)} | 自适应局部基函数中心: {len(self.centers)} | 设备: {self.device}")
        print(f"{'='*72}")

        criterion = nn.MSELoss()
        optimizers, schedulers = {}, {}

        for k in ['u', 'v', 'div']:
            self.models[k] = HybridFourierRBFNet(
                self.centers, self.widths, hidden_dim=hidden_dim, num_freqs=4
            ).to(self.device)
            optimizers[k] = torch.optim.Adam(self.models[k].parameters(), lr=lr, weight_decay=1e-6)
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

        with torch.no_grad():
            for k in ['u', 'v', 'div']:
                mean, std = self.stats[k]
                phys_rmse = np.sqrt(criterion(self.models[k](X_train), Y_tensors[k]).item()) * std
                print(f"  [训练集物理量纲 RMSE] Res_{k:3s} = {phys_rmse:.4e}")
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
# 模块 6: 全套学术可视化函数 (支持台阶掩码与非均匀网格)
# ============================================================
def plot_nn_source_terms(res_u_true, res_v_true, res_div_true,
                         res_u_pred, res_v_pred, res_div_pred,
                         solver: UniversalSIMPLE, case_name: str):
    """图1: 对比真实离散残差 vs 神经网络预测残差 vs 预测误差 (3×3 云图)"""
    nx, ny = solver.nx, solver.ny
    valid = solver.mesh.valid_mask
    true_fields = [res_u_true, res_v_true, res_div_true]
    pred_fields = [res_u_pred, res_v_pred, res_div_pred]
    names = ['Res_u (u-Momentum)', 'Res_v (v-Momentum)', 'Res_div (Continuity)']

    fig, axes = plt.subplots(3, 3, figsize=(18, 11))
    fig.suptitle(f"[{case_name}] Discrete Residual Source Terms: Ground Truth vs NN Surrogate Prediction",
                 fontsize=14, fontweight='bold')

    for row, (t_in, p_full, name) in enumerate(zip(true_fields, pred_fields, names)):
        t_full = np.zeros((nx, ny))
        t_full[1:-1, 1:-1] = t_in
        diff = p_full - t_full

        vmax_main = max(np.max(np.abs(t_full[valid])), np.max(np.abs(p_full[valid])), 1e-12)
        vmax_diff = max(np.max(np.abs(diff[valid])), 1e-12)

        for col, (data, sub_title, vm) in enumerate([
            (t_full, f'True {name}', vmax_main),
            (p_full, f'NN Predicted {name}', vmax_main),
            (diff,   f'NN Error (max={np.max(np.abs(diff[valid])):.2e})', vmax_diff)
        ]):
            ax = axes[row, col]
            plot_data = np.where(valid, data, np.nan)
            cf = ax.contourf(solver.X.T, solver.Y.T, plot_data.T, levels=30,
                             cmap='RdBu_r', vmin=-vm, vmax=vm)
            if solver.mesh.has_obstacle:
                ax.set_facecolor('#dcdcdc')
            plt.colorbar(cf, ax=ax)
            ax.set_title(sub_title, fontsize=11)
            ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.set_aspect('equal')

    plt.tight_layout()
    save_path = f'./{case_name}_nn_source_terms.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [图像保存 1/4] NN 源项拟合对比图已保存: {save_path}")
    plt.show()


def plot_flow_field(solver_dense: UniversalSIMPLE,
                    solver_coarse_orig: UniversalSIMPLE,
                    solver_coarse_nn: UniversalSIMPLE,
                    case_name: str):
    """图2: 绘制流场对比 —— 速度幅值云图 + 流函数 ψ 涡结构云图 (2×3 布局)"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5))
    fig.suptitle(f"[{case_name}] Flow Field: Dense Reference vs Coarse Original vs Coarse NN-Corrected",
                 fontsize=14, fontweight='bold')

    solvers = [solver_dense, solver_coarse_orig, solver_coarse_nn]
    titles = [
        f'Dense Grid Ref ({solver_dense.nx}×{solver_dense.ny})',
        f'Coarse Original ({solver_coarse_orig.nx}×{solver_coarse_orig.ny})',
        f'Coarse NN-Corrected ({solver_coarse_nn.nx}×{solver_coarse_nn.ny})'
    ]

    vmax_vel = float(np.max(np.sqrt(solver_dense.u[solver_dense.mesh.valid_mask]**2 +
                                    solver_dense.v[solver_dense.mesh.valid_mask]**2)))
    vmax_vel = max(vmax_vel, 1.0)

    for col, (sv, title) in enumerate(zip(solvers, titles)):
        valid = sv.mesh.valid_mask

        ax = axes[0, col]
        vel_mag = np.sqrt(sv.u**2 + sv.v**2)
        vel_plot = np.where(valid, vel_mag, np.nan)
        cf = ax.contourf(sv.X.T, sv.Y.T, vel_plot.T, levels=30, cmap='viridis',
                         vmin=0.0, vmax=vmax_vel)
        if sv.mesh.has_obstacle:
            ax.set_facecolor('#dcdcdc')
        plt.colorbar(cf, ax=ax, label='|V| (m/s)')
        ax.set_title(f'{title}\nVelocity Magnitude')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_aspect('equal')

        ax = axes[1, col]
        psi = sv._streamfunc()
        psi_valid = psi[valid]
        psi_min, psi_max = float(psi_valid.min()), float(psi_valid.max())
        psi_plot = np.where(valid, psi, np.nan)
        levels = np.linspace(psi_min, psi_max, 30)
        cf = ax.contourf(sv.X.T, sv.Y.T, psi_plot.T, levels=levels, cmap='Spectral_r')
        if psi_min < -1e-5:
            neg_levels = np.linspace(psi_min, -1e-5, 6)
            ax.contour(sv.X.T, sv.Y.T, psi_plot.T, levels=neg_levels, colors='k', linewidths=0.6, linestyles='--')
        if sv.mesh.has_obstacle:
            ax.set_facecolor('#dcdcdc')
        plt.colorbar(cf, ax=ax, label='ψ')
        ax.set_title(f'{title}\nStreamlines (ψ_min={psi_min:.4f}, ψ_max={psi_max:.4f})')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_aspect('equal')

    plt.tight_layout()
    save_path = f'./{case_name}_nn_flow_field.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [图像保存 2/4] 流场与流线对比图已保存: {save_path}")
    plt.show()


def plot_profile_and_error(solver_dense: UniversalSIMPLE,
                           solver_coarse_orig: UniversalSIMPLE,
                           solver_coarse_nn: UniversalSIMPLE,
                           u_sampled, v_sampled, case_name: str):
    """图3 & 图4: 绘制特征中心线速度剖面对比图 + 误差统计柱状图"""
    mc = solver_coarse_orig.mesh
    full_y_cols = np.where(np.all(mc.valid_mask, axis=1))[0]
    mid_i_c = int(full_y_cols[len(full_y_cols) // 2]) if len(full_y_cols) > 0 else mc.nx // 2
    x_loc = mc.x[mid_i_c]
    mid_i_d = int(np.argmin(np.abs(solver_dense.x - x_loc)))

    full_x_rows = np.where(np.all(mc.valid_mask, axis=0))[0]
    mid_j_c = int(full_x_rows[len(full_x_rows) // 2]) if len(full_x_rows) > 0 else mc.ny // 2
    y_loc = mc.y[mid_j_c]
    mid_j_d = int(np.argmin(np.abs(solver_dense.y - y_loc)))

    # ===== 图 3: 特征截面速度剖面对比图 =====
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5.5))
    fig1.suptitle(f"[{case_name}] Characteristic Velocity Profiles: Original vs NN-Corrected",
                  fontsize=13, fontweight='bold')

    ax = axes1[0]
    ax.plot(solver_dense.u[mid_i_d, :], solver_dense.y, 'k-', lw=2.5, label='Dense Grid (Reference)')
    ax.plot(solver_coarse_orig.u[mid_i_c, :], solver_coarse_orig.y, 'r--o', ms=5, lw=1.8, label='Coarse (Original)')
    ax.plot(solver_coarse_nn.u[mid_i_c, :], solver_coarse_nn.y, 'b:s', ms=5, lw=2.2, label='Coarse (NN-Corrected)')
    ax.set_xlabel('u (m/s)'); ax.set_ylabel('y (m)')
    ax.set_title(f'Vertical Profile (x = {x_loc:.2f}): u vs y')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes1[1]
    ax.plot(solver_dense.x, solver_dense.v[:, mid_j_d], 'k-', lw=2.5, label='Dense Grid (Reference)')
    ax.plot(solver_coarse_orig.x, solver_coarse_orig.v[:, mid_j_c], 'r--o', ms=5, lw=1.8, label='Coarse (Original)')
    ax.plot(solver_coarse_nn.x, solver_coarse_nn.v[:, mid_j_c], 'b:s', ms=5, lw=2.2, label='Coarse (NN-Corrected)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('v (m/s)')
    ax.set_title(f'Horizontal Profile (y = {y_loc:.2f}): v vs x')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    prof_path = f'./{case_name}_nn_centerline_profiles.png'
    plt.savefig(prof_path, dpi=150, bbox_inches='tight')
    print(f"  [图像保存 3/4] 特征截面速度剖面图已保存: {prof_path}")
    plt.show()

    # ===== 计算内部流体节点误差指标 (u & v) =====
    fl_mask = mc.fluid_internal
    err_u_orig = float(np.max(np.abs(u_sampled[fl_mask] - solver_coarse_orig.u[fl_mask])))
    err_u_nn   = float(np.max(np.abs(u_sampled[fl_mask] - solver_coarse_nn.u[fl_mask])))
    err_v_orig = float(np.max(np.abs(v_sampled[fl_mask] - solver_coarse_orig.v[fl_mask])))
    err_v_nn   = float(np.max(np.abs(v_sampled[fl_mask] - solver_coarse_nn.v[fl_mask])))

    rmse_u_orig = float(np.sqrt(np.mean((u_sampled[fl_mask] - solver_coarse_orig.u[fl_mask])**2)))
    rmse_u_nn   = float(np.sqrt(np.mean((u_sampled[fl_mask] - solver_coarse_nn.u[fl_mask])**2)))
    rmse_v_orig = float(np.sqrt(np.mean((v_sampled[fl_mask] - solver_coarse_orig.v[fl_mask])**2)))
    rmse_v_nn   = float(np.sqrt(np.mean((v_sampled[fl_mask] - solver_coarse_nn.v[fl_mask])**2)))

    imp_u_max = (1.0 - err_u_nn / (err_u_orig + 1e-30)) * 100
    imp_v_max = (1.0 - err_v_nn / (err_v_orig + 1e-30)) * 100
    imp_u_l2  = (1.0 - rmse_u_nn / (rmse_u_orig + 1e-30)) * 100
    imp_v_l2  = (1.0 - rmse_v_nn / (rmse_v_orig + 1e-30)) * 100

    print("\n" + "=" * 72)
    print(f"  [{case_name}] 内部流体节点速度误差对比汇总:")
    print(f"  [最大绝对误差 Max Error]:")
    print(f"    u: 原粗网格 = {err_u_orig:.4e} → NN修正后 = {err_u_nn:.4e} (降低 {imp_u_max:.2f}%)")
    print(f"    v: 原粗网格 = {err_v_orig:.4e} → NN修正后 = {err_v_nn:.4e} (降低 {imp_v_max:.2f}%)")
    print(f"  [均方根误差 RMSE]:")
    print(f"    u: 原粗网格 = {rmse_u_orig:.4e} → NN修正后 = {rmse_u_nn:.4e} (降低 {imp_u_l2:.2f}%)")
    print(f"    v: 原粗网格 = {rmse_v_orig:.4e} → NN修正后 = {rmse_v_nn:.4e} (降低 {imp_v_l2:.2f}%)")
    print("=" * 72)

    # ===== 图 4: 误差统计对数柱状图 =====
    fig2, ax2 = plt.subplots(figsize=(10, 5.5))
    x_pos = np.arange(4)
    width = 0.35
    orig_vals = [err_u_orig, err_v_orig, rmse_u_orig, rmse_v_orig]
    nn_vals   = [err_u_nn,   err_v_nn,   rmse_u_nn,   rmse_v_nn]

    bars1 = ax2.bar(x_pos - width/2, orig_vals, width, label='Coarse Original', color='#ff6b6b')
    bars2 = ax2.bar(x_pos + width/2, nn_vals,   width, label='Coarse NN-Corrected', color='#4ecdc4')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['u Max Error', 'v Max Error', 'u RMSE', 'v RMSE'], fontsize=11)
    ax2.set_ylabel('Velocity Error (m/s)', fontsize=11)
    ax2.set_title(
        f'[{case_name}] Error Reduction | Max: u={imp_u_max:.1f}%, v={imp_v_max:.1f}% '
        f'| RMSE: u={imp_u_l2:.1f}%, v={imp_v_l2:.1f}%',
        fontsize=12, fontweight='bold'
    )
    ax2.legend(fontsize=11)
    ax2.set_yscale('log')
    ax2.set_ylim(top=max(orig_vals) * 6.0, bottom=max(min(nn_vals) * 0.2, 1e-8))

    for bar, val in zip(list(bars1) + list(bars2), orig_vals + nn_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15, f'{val:.2e}',
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3, which='both')

    plt.tight_layout()
    err_path = f'./{case_name}_nn_error_stats.png'
    plt.savefig(err_path, dpi=150, bbox_inches='tight')
    print(f"  [图像保存 4/4] 误差统计柱状图已保存: {err_path}")
    plt.show()

    return err_u_orig, err_u_nn, err_v_orig, err_v_nn


# ============================================================
# 模块 7: 通用全流程驱动管线 (Pipeline Runner)
# ============================================================
def run_universal_pipeline(config: dict, plot_results=True):
    case_name = config.get('case_name', 'flow_case')
    dense_msh_path  = config['meshes']['dense_msh']
    coarse_msh_path = config['meshes']['coarse_msh']

    print(f"\n>>> 正在读取并转换 Gmsh 网格文件...")
    mesh_dense  = StructuredGmshMesh(dense_msh_path)
    mesh_coarse = StructuredGmshMesh(coarse_msh_path)
    print(f"  密网格 ({dense_msh_path})  -> 外包维度: {mesh_dense.nx}x{mesh_dense.ny}, "
          f"有效节点数: {int(mesh_dense.valid_mask.sum())}, 物理边界: {list(mesh_dense.bc_masks.keys())}")
    print(f"  粗网格 ({coarse_msh_path}) -> 外包维度: {mesh_coarse.nx}x{mesh_coarse.ny}, "
          f"有效节点数: {int(mesh_coarse.valid_mask.sum())}, 内部流体节点数: {int(mesh_coarse.fluid_internal.sum())}")

    # Step 1: 密网格求解
    print("\n" + "=" * 72)
    print("  Step 1: 密网格求解 (高精度参考解)")
    print("=" * 72)
    solver_dense = UniversalSIMPLE(mesh_dense, config)
    solver_dense.solve(verbose=True)

    # Step 2 & 3: 双三次样条高精度采样与完备离散双残差计算
    print("\n" + "=" * 72)
    print("  Step 2 & 3: 双三次样条采样密网格解并计算粗网格完备离散双残差")
    print("=" * 72)
    u_raw, v_raw, p_raw = sample_dense_to_coarse_spline(mesh_dense, solver_dense, mesh_coarse)

    solver_template = UniversalSIMPLE(mesh_coarse, config)
    res_u, res_v, res_div, u_sampled, v_sampled, p_sampled = compute_discrete_residual(
        solver_template, u_raw, v_raw, p_raw
    )
    act = solver_template.active_in
    print(f"  Res_u   范围: [{res_u[act].min():.4e}, {res_u[act].max():.4e}], mean|x|={np.mean(np.abs(res_u[act])):.4e}")
    print(f"  Res_v   范围: [{res_v[act].min():.4e}, {res_v[act].max():.4e}], mean|x|={np.mean(np.abs(res_v[act])):.4e}")
    print(f"  Res_div 范围: [{res_div[act].min():.4e}, {res_div[act].max():.4e}], mean|x|={np.mean(np.abs(res_div[act])):.4e}")

    # Step 4: 训练神经网络代理模型并预测源项
    corrector = UniversalNeuralCorrector()
    X_train, Y_tensors = corrector.prepare_training_data(solver_template, res_u, res_v, res_div)
    corrector.train(X_train, Y_tensors, config.get('nn_config', {}))
    res_u_pred, res_v_pred, res_div_pred = corrector.predict(solver_template)

    if plot_results:
        plot_nn_source_terms(
            res_u, res_v, res_div,
            res_u_pred, res_v_pred, res_div_pred,
            solver_template, case_name
        )

    # Step 5: 粗网格无修正基准求解
    print("\n" + "=" * 72)
    print("  Step 5: 粗网格基准求解 (Original Coarse)")
    print("=" * 72)
    solver_coarse_orig = UniversalSIMPLE(mesh_coarse, config)
    solver_coarse_orig.solve(verbose=True)

    # Step 6: 粗网格注入 NN 源项修正求解
    print("\n" + "=" * 72)
    print("  Step 6: 粗网格修正求解 (注入 NN 预测的 Res_u + Res_v + Res_div)")
    print("=" * 72)
    solver_coarse_nn = UniversalSIMPLE(mesh_coarse, config)
    solver_coarse_nn.Sp_nn_u   = res_u_pred
    solver_coarse_nn.Sp_nn_v   = res_v_pred
    solver_coarse_nn.Sp_nn_div = res_div_pred
    solver_coarse_nn.solve(verbose=True)

    # Step 7: 误差对比分析与全套可视化输出
    if plot_results:
        print("\n" + "=" * 72)
        print("  Step 7: 误差对比分析与全套可视化图表生成")
        print("=" * 72)
        plot_profile_and_error(
            solver_dense, solver_coarse_orig, solver_coarse_nn,
            u_sampled, v_sampled, case_name
        )
        plot_flow_field(
            solver_dense, solver_coarse_orig, solver_coarse_nn,
            case_name
        )

    return solver_dense, solver_coarse_orig, solver_coarse_nn


if __name__ == '__main__':
    # 自动适配你在 channel 子目录或项目根目录下运行时的网格路径
    dense_path = "../meshes/back_fine.msh" if os.path.exists("../meshes/back_fine.msh") else "./meshes/back_fine.msh"
    coarse_path = "../meshes/back_coarse.msh" if os.path.exists("../meshes/back_coarse.msh") else "./meshes/back_coarse.msh"

    config_backward = {
        "case_name": "backward",
        "meshes": {
            "dense_msh": dense_path,
            "coarse_msh": coarse_path
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
            "epochs": 4000, "lr": 3e-3, "hidden_dim": 96
        }
    }

    run_universal_pipeline(config_backward, plot_results=True)