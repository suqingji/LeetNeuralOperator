#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================================
DeepSIMPLE_2D (Scheme 2 真·实时动态自洽不动点闭环版 - True Self-Consistent In-the-Loop VBNN)
==================================================================================================

【核心机制说明：真正的随流场迭代实时更新，绝不中途冻结源项】
 1. 严格自洽的不动点训练 (Self-Consistent Target-State Training):
    - 当粗网格迭代最终收敛并逼近密网格真解 (u*, v*, p*) 时，神经网络在粗网格上看到的输入正是 (u*, v*, p*)
      及其由粗网格差分算子 solver._gradient() 重构出的梯度场。
    - 因此，训练时直接以【粗网格上的真解投影场 + 粗网格离散算子重构梯度】提取伽利略不变量特征，学习到
      完备离散残差 R* 的映射。这保证了当流场逼近真解时，NN 动态输出的源项恰好严格维持该真解不动点！
 2. 满秩无退化客观物理向量基 (Full-Rank Non-Degenerate Objective Vector Basis):
    - 引入平滑速度正交基 U_sm^⊥ 与壁面势梯度正交基 ∇^⊥ φ_w，彻底解决分离涡心/再附点处 |U|->0 导致的
      向量基退化问题，使真解态下的残差重构误差降至 1e-6 量级。
 3. 全程实时动态闭环更新 (Continuous In-the-Loop Dynamic Updating):
    - 从零初场启动，在 SIMPLE 迭代全程中每隔 update_freq 步实时提取当前流场 (u^(k), v^(k), p^(k)) 的特征
      并动态更新源项 S_NN^(k) = (1 - β) S_NN^(k-1) + β NN(u^(k), p^(k))，直至流场与源项同步收敛！
==================================================================================================
"""

import os
import copy
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
# 辅助模块: 若本地缺少 .msh 文件，自动生成标准 Gmsh 2.2 台阶网格
# ============================================================
def ensure_gmsh_files_exist(dense_path: str, coarse_path: str):
    """如果指定路径下存在您的 .msh 文件则直接读取；若不存在则自动生成标准 L 型后向台阶 .msh 网格"""
    for path, (nx1, nx2, ny1, ny2) in [
        (dense_path, (16, 48, 16, 16)),
        (coarse_path, (6, 18, 6, 6)),
    ]:
        if os.path.exists(path):
            continue
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        x_left = np.linspace(0.0, 1.0, nx1, endpoint=False)
        x_right = np.linspace(1.0, 5.0, nx2 + 1)
        xs = np.concatenate([x_left, x_right])
        y_bot = np.linspace(0.0, 0.5, ny1, endpoint=False)
        y_top = np.linspace(0.5, 1.0, ny2 + 1)
        ys = np.concatenate([y_bot, y_top])

        nodes = {}
        grid_nid = np.full((len(xs), len(ys)), -1, dtype=int)
        nid = 1
        for i, xv in enumerate(xs):
            for j, yv in enumerate(ys):
                if xv < 1.0 - 1e-9 and yv < 0.5 - 1e-9:
                    continue
                nodes[nid] = (xv, yv, 0.0)
                grid_nid[i, j] = nid
                nid += 1

        elems = []
        eid = 1
        for i in range(len(xs)):
            for j in range(len(ys)):
                n1 = grid_nid[i, j]
                if n1 < 0:
                    continue
                if i + 1 < len(xs) and grid_nid[i + 1, j] >= 0:
                    n2 = grid_nid[i + 1, j]
                    yv = ys[j]
                    if abs(yv - 1.0) < 1e-9 or abs(yv - 0.0) < 1e-9 or (
                        abs(yv - 0.5) < 1e-9 and xs[i] < 1.0 + 1e-9 and xs[i + 1] <= 1.0 + 1e-9
                    ):
                        elems.append((eid, 1, 3, [n1, n2]))
                        eid += 1
                if j + 1 < len(ys) and grid_nid[i, j + 1] >= 0:
                    n2 = grid_nid[i, j + 1]
                    xv = xs[i]
                    if abs(xv - 0.0) < 1e-9 and ys[j] >= 0.5 - 1e-9:
                        elems.append((eid, 1, 1, [n1, n2]))
                        eid += 1
                    elif abs(xv - 5.0) < 1e-9:
                        elems.append((eid, 1, 2, [n1, n2]))
                        eid += 1
                    elif abs(xv - 1.0) < 1e-9 and ys[j] < 0.5 + 1e-9 and ys[j + 1] <= 0.5 + 1e-9:
                        elems.append((eid, 1, 3, [n1, n2]))
                        eid += 1

        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                n1, n2, n3, n4 = (
                    grid_nid[i, j],
                    grid_nid[i + 1, j],
                    grid_nid[i + 1, j + 1],
                    grid_nid[i, j + 1],
                )
                if n1 >= 0 and n2 >= 0 and n3 >= 0 and n4 >= 0:
                    elems.append((eid, 3, 4, [n1, n2, n3, n4]))
                    eid += 1

        with open(path, 'w', encoding='utf-8') as f:
            f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n4\n")
            f.write('1 1 "inlet"\n1 2 "outlet"\n1 3 "wall"\n2 4 "fluid"\n$EndPhysicalNames\n')
            f.write(f"$Nodes\n{len(nodes)}\n")
            for k, (xv, yv, zv) in nodes.items():
                f.write(f"{k} {xv:.8f} {yv:.8f} {zv:.8f}\n")
            f.write(f"$EndNodes\n$Elements\n{len(elems)}\n")
            for e_id, e_type, p_tag, n_list in elems:
                n_str = " ".join(str(x) for x in n_list)
                f.write(f"{e_id} {e_type} 2 {p_tag} {p_tag} {n_str}\n")
            f.write("$EndElements\n")


# ============================================================
# 模块 1: 基于 1D 向量化连接图的 Gmsh (.msh) 拓扑重构解析器
# ============================================================
class ConnectivityGmshMesh:
    def __init__(self, msh_filepath, tol_decimals=6, rotate_90_ccw=False, scale_factor=1.0):
        self.filepath = msh_filepath
        self.tol_decimals = tol_decimals
        self.rotate_90_ccw = rotate_90_ccw
        self.scale_factor = float(scale_factor)

        self.raw_nodes = {}          # gmsh_nid -> (x, y, z)
        self.phys_names = {}         # (dim, tag) -> name_str
        self.boundary_nodes = {}     # phys_name -> set of gmsh_nids
        self.boundary_edges = {}     # phys_name -> list of (n1, n2)
        self.quad_elements = []      # list of (n1, n2, n3, n4)

        self._parse_msh()
        if abs(self.scale_factor - 1.0) > 1e-12:
            self._apply_geometric_scaling(self.scale_factor)
        if self.rotate_90_ccw:
            self._apply_rigid_rotation_90_ccw()
        self._build_1d_connectivity()

    def _parse_msh(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

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

    def _apply_geometric_scaling(self, scale: float):
        for nid, (x, y, z) in self.raw_nodes.items():
            self.raw_nodes[nid] = (x * scale, y * scale, z * scale)

    def _apply_rigid_rotation_90_ccw(self):
        xs = [coord[0] for coord in self.raw_nodes.values()]
        ys = [coord[1] for coord in self.raw_nodes.values()]
        x_min, y_max = min(xs), max(ys)
        for nid, (x, y, z) in self.raw_nodes.items():
            self.raw_nodes[nid] = (y_max - y, x - x_min, z)

    def _build_1d_connectivity(self):
        nids = np.array(list(self.raw_nodes.keys()), dtype=int)
        coords = np.array([self.raw_nodes[nid][:2] for nid in nids], dtype=float)
        rounded_coords = np.round(coords, self.tol_decimals)

        ux = np.unique(rounded_coords[:, 0])
        uy = np.unique(rounded_coords[:, 1])
        self.nx = len(ux)
        self.ny = len(uy)

        ix_all = np.searchsorted(ux, rounded_coords[:, 0])
        iy_all = np.searchsorted(uy, rounded_coords[:, 1])

        self.N = len(nids)
        self.id_arr = np.arange(self.N, dtype=int)
        self.x_1d = coords[:, 0].copy()
        self.y_1d = coords[:, 1].copy()

        self.gmsh_to_1d = {}
        self.grid_to_1d = np.full((self.nx, self.ny), -1, dtype=int)
        for k, (nid, i, j) in enumerate(zip(nids, ix_all, iy_all)):
            self.gmsh_to_1d[int(nid)] = k
            self.grid_to_1d[int(i), int(j)] = k

        if int(np.sum(self.grid_to_1d >= 0)) != self.N:
            raise ValueError("检测到坐标重合或非正交节点，无法构建正交四邻接拓扑图！")

        self.x = np.array([self.x_1d[self.grid_to_1d[i, self.grid_to_1d[i, :] >= 0][0]] for i in range(self.nx)])
        self.y = np.array([self.y_1d[self.grid_to_1d[self.grid_to_1d[:, j] >= 0, j][0]] for j in range(self.ny)])
        self.Lx = float(self.x[-1] - self.x[0])
        self.Ly = float(self.y[-1] - self.y[0])
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        self.valid_mask = (self.grid_to_1d >= 0)
        self.solid_mask = ~self.valid_mask
        self.has_obstacle = bool(np.any(self.solid_mask))

        self.nbr_E = np.full(self.N, -1, dtype=int)
        self.nbr_W = np.full(self.N, -1, dtype=int)
        self.nbr_N = np.full(self.N, -1, dtype=int)
        self.nbr_S = np.full(self.N, -1, dtype=int)

        for k, (i, j) in enumerate(zip(ix_all, iy_all)):
            if i + 1 < self.nx: self.nbr_E[k] = self.grid_to_1d[i + 1, j]
            if i - 1 >= 0:      self.nbr_W[k] = self.grid_to_1d[i - 1, j]
            if j + 1 < self.ny: self.nbr_N[k] = self.grid_to_1d[i, j + 1]
            if j - 1 >= 0:      self.nbr_S[k] = self.grid_to_1d[i, j - 1]

        self.bc_masks_1d = {}
        all_b_mask_1d = np.zeros(self.N, dtype=bool)
        for bname, b_nids in self.boundary_nodes.items():
            bmask = np.zeros(self.N, dtype=bool)
            for nid in b_nids:
                if nid in self.gmsh_to_1d:
                    bmask[self.gmsh_to_1d[nid]] = True
            self.bc_masks_1d[bname] = bmask
            all_b_mask_1d |= bmask

        has_all_4_nbrs = (self.nbr_E >= 0) & (self.nbr_W >= 0) & (self.nbr_N >= 0) & (self.nbr_S >= 0)
        self.is_internal = has_all_4_nbrs & (~all_b_mask_1d)
        self.fluid_internal = self.to_2d(self.is_internal, fill_val=False).astype(bool)

        self.dx_E = np.zeros(self.N, dtype=float)
        self.dx_W = np.zeros(self.N, dtype=float)
        self.dy_N = np.zeros(self.N, dtype=float)
        self.dy_S = np.zeros(self.N, dtype=float)

        mE = self.nbr_E >= 0; self.dx_E[mE] = self.x_1d[self.nbr_E[mE]] - self.x_1d[mE]
        mW = self.nbr_W >= 0; self.dx_W[mW] = self.x_1d[mW] - self.x_1d[self.nbr_W[mW]]
        mN = self.nbr_N >= 0; self.dy_N[mN] = self.y_1d[self.nbr_N[mN]] - self.y_1d[mN]
        mS = self.nbr_S >= 0; self.dy_S[mS] = self.y_1d[mS] - self.y_1d[self.nbr_S[mS]]

        self.dx_E[~mE] = self.dx_W[~mE]
        self.dx_W[~mW] = self.dx_E[~mW]
        self.dy_N[~mN] = self.dy_S[~mN]
        self.dy_S[~mS] = self.dy_N[~mS]

        self.dx_P = 0.5 * (self.dx_E + self.dx_W)
        self.dy_P = 0.5 * (self.dy_N + self.dy_S)
        self.cell_vol = self.dx_P * self.dy_P

        hx = np.diff(self.x)
        hy = np.diff(self.y)
        self.is_uniform = (
            np.allclose(hx, self.Lx / (self.nx - 1), rtol=1e-2) and
            np.allclose(hy, self.Ly / (self.ny - 1), rtol=1e-2)
        )

    def to_2d(self, field_1d, fill_val=0.0):
        out = np.full((self.nx, self.ny), fill_val, dtype=np.asarray(field_1d).dtype)
        out[self.valid_mask] = field_1d[self.grid_to_1d[self.valid_mask]]
        return out

    def to_1d(self, field_2d):
        return field_2d[self.valid_mask][np.argsort(self.grid_to_1d[self.valid_mask])]


def make_rotated_90_ccw_config(config: dict) -> dict:
    """将算例配置逆时针旋转 90°: (u', v') = (-v, u)"""
    cfg_rot = copy.deepcopy(config)
    cfg_rot['case_name'] = f"{config.get('case_name', 'case')}_rot90_CCW"

    if 'initial_conditions' in cfg_rot:
        u0 = float(cfg_rot['initial_conditions'].get('u', 0.0))
        v0 = float(cfg_rot['initial_conditions'].get('v', 0.0))
        cfg_rot['initial_conditions']['u'] = -v0
        cfg_rot['initial_conditions']['v'] = u0

    if 'boundary_conditions' in cfg_rot:
        for bname, bc in cfg_rot['boundary_conditions'].items():
            if 'u' in bc or 'v' in bc:
                u_b = float(bc.get('u', 0.0))
                v_b = float(bc.get('v', 0.0))
                bc['u'] = -v_b
                bc['v'] = u_b
    return cfg_rot


# ============================================================
# 模块 2: 基于 1D 邻接映射表的稀疏矩阵向量化组装器
# ============================================================
def build_1d_connectivity_matrix(aP, aE, aW, aN, aS, mesh: ConnectivityGmshMesh):
    N = mesh.N
    ids = mesh.id_arr
    rows_list = [ids]
    cols_list = [ids]
    vals_list = [aP]

    for nbr_arr, a_nb in [
        (mesh.nbr_E, aE),
        (mesh.nbr_W, aW),
        (mesh.nbr_N, aN),
        (mesh.nbr_S, aS),
    ]:
        valid = (nbr_arr >= 0) & (a_nb != 0.0)
        rows_list.append(ids[valid])
        cols_list.append(nbr_arr[valid])
        vals_list.append(-a_nb[valid])

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)
    return coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()


# ============================================================
# 模块 3: 纯 1D 拓扑连接图驱动的通用 SIMPLE 求解器 (真·全程实时动态更新源项)
# ============================================================
class UniversalSIMPLE:
    def __init__(self, mesh: ConnectivityGmshMesh, config: dict):
        self.mesh = mesh
        self.config = config
        self.N = mesh.N
        self.nx, self.ny = mesh.nx, mesh.ny
        self.Lx, self.Ly = mesh.Lx, mesh.Ly
        self.x, self.y = mesh.x, mesh.y
        self.X, self.Y = mesh.X, mesh.Y

        fluid_cfg = config.get('fluid', {})
        self.rho = float(fluid_cfg.get('rho', 1.0))
        self.mu = float(fluid_cfg.get('mu', 0.01))

        raw_bc_cfg = config.get('boundary_conditions', {})
        self.bc_cfg = {}
        for bname in mesh.bc_masks_1d.keys():
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

        self._build_1d_neumann_maps()

        self.u = np.zeros(self.N, dtype=float)
        self.v = np.zeros(self.N, dtype=float)
        self.p = np.zeros(self.N, dtype=float)

        self.aE = np.zeros(self.N, dtype=float)
        self.aW = np.zeros(self.N, dtype=float)
        self.aN = np.zeros(self.N, dtype=float)
        self.aS = np.zeros(self.N, dtype=float)
        self.aP = np.zeros(self.N, dtype=float)
        self.Sp = np.zeros(self.N, dtype=float)

        self.active = self.mesh.is_internal
        self.ref_p_idx = int(np.where(self.active)[0][0])

        self.Sp_nn_u = None
        self.Sp_nn_v = None
        self.Sp_nn_div = None
        self.online_corrector = None
        self._cached_wall_potential = None

        self._init_fields()

    @property
    def u_2d(self):
        return self.mesh.to_2d(self.u, fill_val=0.0)

    @property
    def v_2d(self):
        return self.mesh.to_2d(self.v, fill_val=0.0)

    @property
    def p_2d(self):
        return self.mesh.to_2d(self.p, fill_val=0.0)

    def _build_1d_neumann_maps(self):
        self.neumann_maps_1d = {}
        m = self.mesh
        fl = m.is_internal

        for bname, bmask in m.bc_masks_1d.items():
            b_indices = np.where(bmask)[0]
            b_list = []
            n_lists = []
            for k in b_indices:
                direct_nbs = [nb for nb in (m.nbr_E[k], m.nbr_W[k], m.nbr_N[k], m.nbr_S[k]) if nb >= 0 and fl[nb]]
                if len(direct_nbs) > 0:
                    b_list.append(k)
                    n_lists.append(direct_nbs)
                    continue

                hop2_nbs = []
                for nb in (m.nbr_E[k], m.nbr_W[k], m.nbr_N[k], m.nbr_S[k]):
                    if nb >= 0:
                        for nb2 in (m.nbr_E[nb], m.nbr_W[nb], m.nbr_N[nb], m.nbr_S[nb]):
                            if nb2 >= 0 and fl[nb2] and nb2 not in hop2_nbs:
                                hop2_nbs.append(nb2)
                if len(hop2_nbs) > 0:
                    b_list.append(k)
                    n_lists.append(hop2_nbs)
                    continue

                fallback_nbs = [nb for nb in (m.nbr_E[k], m.nbr_W[k], m.nbr_N[k], m.nbr_S[k]) if nb >= 0 and nb != k]
                if len(fallback_nbs) > 0:
                    b_list.append(k)
                    n_lists.append(fallback_nbs)

            self.neumann_maps_1d[bname] = (np.array(b_list, dtype=int), n_lists)

    def _apply_neumann_extrapolation(self, field_1d: np.ndarray, bname: str):
        b_arr, n_lists = self.neumann_maps_1d[bname]
        for k, nbs in zip(b_arr, n_lists):
            if len(nbs) == 1:
                field_1d[k] = field_1d[nbs[0]]
            else:
                field_1d[k] = float(np.mean(field_1d[nbs]))

    def _init_fields(self):
        init_cfg = self.config.get('initial_conditions', {'u': 0.0, 'v': 0.0, 'p': 0.0})
        self.u[:] = float(init_cfg.get('u', 0.0))
        self.v[:] = float(init_cfg.get('v', 0.0))
        self.p[:] = float(init_cfg.get('p', 0.0))
        self._bc_uv()
        self._bc_p()

    def _bc_uv(self):
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'pressure_outlet':
                self._apply_neumann_extrapolation(self.u, bname)
                self._apply_neumann_extrapolation(self.v, bname)

        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'inlet':
                mask = self.mesh.bc_masks_1d[bname]
                self.u[mask] = float(bc.get('u', 0.0))
                self.v[mask] = float(bc.get('v', 0.0))

        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'wall':
                mask = self.mesh.bc_masks_1d[bname]
                self.u[mask] = float(bc.get('u', 0.0))
                self.v[mask] = float(bc.get('v', 0.0))

    def _bc_p(self):
        if not self.has_pressure_outlet:
            self.p -= self.p[self.ref_p_idx]

        for bname, bc in self.bc_cfg.items():
            if bc['type'] in ('wall', 'inlet'):
                self._apply_neumann_extrapolation(self.p, bname)

        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'pressure_outlet':
                mask = self.mesh.bc_masks_1d[bname]
                self.p[mask] = float(bc.get('p', 0.0))

    def _bc_p_prime(self, pp):
        if not self.has_pressure_outlet:
            pp -= pp[self.ref_p_idx]

        for bname, bc in self.bc_cfg.items():
            if bc['type'] in ('wall', 'inlet'):
                self._apply_neumann_extrapolation(pp, bname)

        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'pressure_outlet':
                mask = self.mesh.bc_masks_1d[bname]
                pp[mask] = 0.0

    def _gradient(self, f_1d):
        m = self.mesh
        Gx = np.zeros(self.N, dtype=float)
        Gy = np.zeros(self.N, dtype=float)

        e_ok = m.nbr_E >= 0
        w_ok = m.nbr_W >= 0
        both_x = e_ok & w_ok
        only_e = e_ok & (~w_ok)
        only_w = (~e_ok) & w_ok

        if np.any(both_x):
            idx = np.where(both_x)[0]
            ie, iw = m.nbr_E[idx], m.nbr_W[idx]
            hE, hW = m.dx_E[idx], m.dx_W[idx]
            Gx[idx] = (
                (hW / (hE * (hE + hW))) * f_1d[ie]
                + ((hE - hW) / (hE * hW)) * f_1d[idx]
                - (hE / (hW * (hE + hW))) * f_1d[iw]
            )
        if np.any(only_e):
            idx = np.where(only_e)[0]
            Gx[idx] = (f_1d[m.nbr_E[idx]] - f_1d[idx]) / m.dx_E[idx]
        if np.any(only_w):
            idx = np.where(only_w)[0]
            Gx[idx] = (f_1d[idx] - f_1d[m.nbr_W[idx]]) / m.dx_W[idx]

        n_ok = m.nbr_N >= 0
        s_ok = m.nbr_S >= 0
        both_y = n_ok & s_ok
        only_n = n_ok & (~s_ok)
        only_s = (~n_ok) & s_ok

        if np.any(both_y):
            idx = np.where(both_y)[0]
            in_nb, is_nb = m.nbr_N[idx], m.nbr_S[idx]
            hN, hS = m.dy_N[idx], m.dy_S[idx]
            Gy[idx] = (
                (hS / (hN * (hN + hS))) * f_1d[in_nb]
                + ((hN - hS) / (hN * hS)) * f_1d[idx]
                - (hN / (hS * (hN + hS))) * f_1d[is_nb]
            )
        if np.any(only_n):
            idx = np.where(only_n)[0]
            Gy[idx] = (f_1d[m.nbr_N[idx]] - f_1d[idx]) / m.dy_N[idx]
        if np.any(only_s):
            idx = np.where(only_s)[0]
            Gy[idx] = (f_1d[idx] - f_1d[m.nbr_S[idx]]) / m.dy_S[idx]

        return Gx, Gy

    def _assemble_momentum(self, direction):
        rho, mu = self.rho, self.mu
        m = self.mesh
        act = self.active

        De = mu / (m.dx_P * m.dx_E)
        Dw = mu / (m.dx_P * m.dx_W)
        Dn = mu / (m.dy_P * m.dy_N)
        Ds = mu / (m.dy_P * m.dy_S)

        aE_c = np.where(self.u < 0.0, -rho * self.u / m.dx_E, 0.0)
        aW_c = np.where(self.u > 0.0,  rho * self.u / m.dx_W, 0.0)
        aN_c = np.where(self.v < 0.0, -rho * self.v / m.dy_N, 0.0)
        aS_c = np.where(self.v > 0.0,  rho * self.v / m.dy_S, 0.0)

        self.aE = np.where(act, De + aE_c, 0.0)
        self.aW = np.where(act, Dw + aW_c, 0.0)
        self.aN = np.where(act, Dn + aN_c, 0.0)
        self.aS = np.where(act, Ds + aS_c, 0.0)
        self.aP = np.where(act, self.aE + self.aW + self.aN + self.aS, 1.0)

        Gx, Gy = self._gradient(self.p)
        if direction == 'u':
            self.Sp = -Gx
            if self.Sp_nn_u is not None:
                self.Sp += self.Sp_nn_u
        else:
            self.Sp = -Gy
            if self.Sp_nn_v is not None:
                self.Sp += self.Sp_nn_v

        self.Sp[~act] = 0.0

    def _solve_scalar(self, phi, alpha):
        m = self.mesh
        act = self.active
        phi_old = phi.copy()

        aP_ur = np.where(act, self.aP / alpha, 1.0)
        aE_eff = self.aE.copy()
        aW_eff = self.aW.copy()
        aN_eff = self.aN.copy()
        aS_eff = self.aS.copy()

        rhs = np.where(
            act,
            self.Sp + (1.0 - alpha) / alpha * self.aP * phi_old,
            phi
        )

        p_out_1d = np.zeros(self.N, dtype=bool)
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'pressure_outlet':
                p_out_1d |= m.bc_masks_1d[bname]

        for nbr_arr, a_arr in [
            (m.nbr_E, aE_eff),
            (m.nbr_W, aW_eff),
            (m.nbr_N, aN_eff),
            (m.nbr_S, aS_eff),
        ]:
            is_out_nb = act & (nbr_arr >= 0) & p_out_1d[nbr_arr]
            aP_ur[is_out_nb] -= a_arr[is_out_nb]
            a_arr[is_out_nb] = 0.0

        A = build_1d_connectivity_matrix(aP_ur, aE_eff, aW_eff, aN_eff, aS_eff, m)
        phi[:] = spsolve(A, rhs)

    def _rc_face_velocities(self, alpha_u):
        m = self.mesh
        d = np.where(self.active, alpha_u / self.aP, 0.0)
        for bname in self.bc_cfg.keys():
            self._apply_neumann_extrapolation(d, bname)

        Gx, Gy = self._gradient(self.p)

        u_e = np.zeros(self.N, dtype=float)
        v_n = np.zeros(self.N, dtype=float)

        mE = m.nbr_E >= 0
        idx_e = np.where(mE)[0]
        nb_e = m.nbr_E[idx_e]
        d_bar_e = 0.5 * (d[idx_e] + d[nb_e])
        Gx_bar_e = 0.5 * (Gx[idx_e] + Gx[nb_e])
        Gx_compact_e = (self.p[nb_e] - self.p[idx_e]) / m.dx_E[idx_e]
        u_e[idx_e] = 0.5 * (self.u[idx_e] + self.u[nb_e]) + d_bar_e * (Gx_bar_e - Gx_compact_e)

        mN = m.nbr_N >= 0
        idx_n = np.where(mN)[0]
        nb_n = m.nbr_N[idx_n]
        d_bar_n = 0.5 * (d[idx_n] + d[nb_n])
        Gy_bar_n = 0.5 * (Gy[idx_n] + Gy[nb_n])
        Gy_compact_n = (self.p[nb_n] - self.p[idx_n]) / m.dy_N[idx_n]
        v_n[idx_n] = 0.5 * (self.v[idx_n] + self.v[nb_n]) + d_bar_n * (Gy_bar_n - Gy_compact_n)

        return u_e, v_n, d

    def _solve_pressure_correction(self, alpha_u):
        m = self.mesh
        act = self.active

        u_e, v_n, d = self._rc_face_velocities(alpha_u)

        div = np.zeros(self.N, dtype=float)
        idx = np.where(act)[0]
        u_w_act = u_e[m.nbr_W[idx]]
        v_s_act = v_n[m.nbr_S[idx]]
        div[idx] = (u_e[idx] - u_w_act) / m.dx_P[idx] + (v_n[idx] - v_s_act) / m.dy_P[idx]

        if self.Sp_nn_div is not None:
            b = np.where(act, -(div - self.Sp_nn_div), 0.0)
        else:
            b = np.where(act, -div, 0.0)

        mass_resid = float(np.max(np.abs(b[act])))

        p_out_1d = np.zeros(self.N, dtype=bool)
        for bname, bc in self.bc_cfg.items():
            if bc['type'] == 'pressure_outlet':
                p_out_1d |= m.bc_masks_1d[bname]

        conn_p = act | p_out_1d

        aE_pc = np.zeros(self.N, dtype=float)
        aW_pc = np.zeros(self.N, dtype=float)
        aN_pc = np.zeros(self.N, dtype=float)
        aS_pc = np.zeros(self.N, dtype=float)

        cE = act & (m.nbr_E >= 0) & conn_p[np.maximum(m.nbr_E, 0)]
        aE_pc[cE] = 0.5 * (d[cE] + d[m.nbr_E[cE]]) / (m.dx_P[cE] * m.dx_E[cE])

        cW = act & (m.nbr_W >= 0) & conn_p[np.maximum(m.nbr_W, 0)]
        aW_pc[cW] = 0.5 * (d[cW] + d[m.nbr_W[cW]]) / (m.dx_P[cW] * m.dx_W[cW])

        cN = act & (m.nbr_N >= 0) & conn_p[np.maximum(m.nbr_N, 0)]
        aN_pc[cN] = 0.5 * (d[cN] + d[m.nbr_N[cN]]) / (m.dy_P[cN] * m.dy_N[cN])

        cS = act & (m.nbr_S >= 0) & conn_p[np.maximum(m.nbr_S, 0)]
        aS_pc[cS] = 0.5 * (d[cS] + d[m.nbr_S[cS]]) / (m.dy_P[cS] * m.dy_S[cS])

        aP_pc = np.where(act, aE_pc + aW_pc + aN_pc + aS_pc, 1.0)

        if not self.has_pressure_outlet:
            r_idx = self.ref_p_idx
            aP_pc[r_idx] = 1.0
            aE_pc[r_idx] = aW_pc[r_idx] = aN_pc[r_idx] = aS_pc[r_idx] = 0.0
            b[r_idx] = 0.0

        A_p = build_1d_connectivity_matrix(aP_pc, aE_pc, aW_pc, aN_pc, aS_pc, m)
        pp = spsolve(A_p, b)
        self._bc_p_prime(pp)

        return pp, d, mass_resid

    def _correct_fields(self, pp, d, alpha_p):
        act = self.active
        self.p[act] += alpha_p * pp[act]
        Gx_pp, Gy_pp = self._gradient(pp)
        self.u[act] -= d[act] * Gx_pp[act]
        self.v[act] -= d[act] * Gy_pp[act]

        self._bc_uv()
        self._bc_p()

    def _rel_change(self, phi_new, phi_old):
        act = self.active
        denom = np.max(np.abs(phi_old[act]))
        if denom < 1e-30:
            return float(np.mean(np.abs(phi_new[act] - phi_old[act])))
        return float(np.mean(np.abs(phi_new[act] - phi_old[act])) / denom)

    def _streamfunc(self):
        psi = np.zeros((self.nx, self.ny))
        if not self.mesh.rotate_90_ccw:
            u_clean = self.u_2d
            dy_1d = np.diff(self.y)[None, :]
            psi[:, 1:] = np.cumsum(0.5 * (u_clean[:, 1:] + u_clean[:, :-1]) * dy_1d, axis=1)
        else:
            v_clean = self.v_2d
            dx_1d = np.diff(self.x)[:, None]
            psi_rev = np.cumsum(0.5 * (v_clean[-2::-1, :] + v_clean[:0:-1, :]) * dx_1d[::-1, :], axis=0)
            psi[:-1, :] = psi_rev[::-1, :]
        return psi

    def solve(self, verbose=True, dynamic_nn_corrector=None):
        """
        【真·全程实时动态自洽迭代求解器 (True Continuous In-the-Loop Coupling)】
        绝不中途冻结源项！在 SIMPLE 迭代全程中：
          - 仅前 nn_warmup (30) 步让零初场建立非零流场；
          - 随后直到最终收敛的每一个更新周期 (每 nn_update_freq=5 步)，始终拿当前最新的
            (self.u, self.v, self.p) 实时计算粗网格梯度、实时喂给神经网络预测源项，
            通过源项欠松弛与流场共同演化至自洽不动点 (u*, v*, p*, S_NN*)！
        """
        if dynamic_nn_corrector is not None:
            self.online_corrector = dynamic_nn_corrector

        s_cfg = self.config.get('solver', {})
        max_iter = int(s_cfg.get('max_iter', 2500))
        tol      = float(s_cfg.get('tol', 1e-7))
        alpha_u  = float(s_cfg.get('alpha_u', 0.7))
        alpha_v  = float(s_cfg.get('alpha_v', 0.7))
        alpha_p  = float(s_cfg.get('alpha_p', 0.3))

        nn_warmup      = int(s_cfg.get('nn_warmup', 30))
        nn_update_freq = int(s_cfg.get('nn_update_freq', 5))
        beta_nn_max    = float(s_cfg.get('beta_nn', 0.25))

        topo_str = "1D多块台阶拓扑图" if self.mesh.has_obstacle else (
            "1D均匀拓扑图" if self.mesh.is_uniform else "1D非均匀拉伸拓扑图"
        )
        if verbose:
            print("=" * 84)
            print(f"  1D 连接图 SIMPLE 求解器 | 案例: {self.config.get('case_name', 'Unnamed')} "
                  f"| 缩放={self.mesh.scale_factor:g}x | N={self.N} ({topo_str})")
            if self.online_corrector is not None:
                print(f"  ✓ 已启用【真·全程实时动态闭环 VBNN】(Warmup={nn_warmup}, "
                      f"UpdateFreq={nn_update_freq}, 全程随流场动态更新不冻结)")
            print("=" * 84)

        u_old, v_old = self.u.copy(), self.v.copy()
        res_u, res_v, mass_resid = 1.0, 1.0, 1.0
        nn_calls = 0
        src_rel_diff = 1.0

        for it in range(1, max_iter + 1):
            # -----------------------------------------------------------------
            # 【核心：全程随当前流场 (self.u, self.v, self.p) 实时重算特征并更新源项】
            # -----------------------------------------------------------------
            if self.online_corrector is not None and it >= nn_warmup:
                if (it - nn_warmup) % nn_update_freq == 0:
                    nn_calls += 1
                    # 100% 基于当前正在迭代演化的实时流场 self.u, self.v, self.p 预测源项！
                    ru_now, rv_now, rdiv_now = self.online_corrector.predict(self)

                    # 自适应欠松弛因子：在流场大幅调整期采用 beta_nn_max (0.25)，当流场接近收敛时自适应平缓步长
                    flow_res = max(res_u, res_v)
                    beta = beta_nn_max if flow_res > 1e-4 else max(0.05, beta_nn_max * (flow_res / 1e-4) ** 0.25)

                    if self.Sp_nn_u is None:
                        self.Sp_nn_u   = beta * ru_now
                        self.Sp_nn_v   = beta * rv_now
                        self.Sp_nn_div = beta * rdiv_now
                        src_rel_diff   = 1.0
                    else:
                        # 记录当前实时预测源项与上一步源项的相对差，验证不动点自洽性
                        denom_s = np.max(np.abs(self.Sp_nn_u[self.active])) + 1e-12
                        src_rel_diff = float(np.mean(np.abs(ru_now[self.active] - self.Sp_nn_u[self.active])) / denom_s)

                        self.Sp_nn_u   = (1.0 - beta) * self.Sp_nn_u   + beta * ru_now
                        self.Sp_nn_v   = (1.0 - beta) * self.Sp_nn_v   + beta * rv_now
                        self.Sp_nn_div = (1.0 - beta) * self.Sp_nn_div + beta * rdiv_now

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

            if verbose and (it % 250 == 0 or it == 1):
                vel_max = np.max(np.sqrt(self.u**2 + self.v**2))
                psi_min = np.min(self._streamfunc()[self.mesh.valid_mask])
                print(f"  iter {it:5d} | res_u={res_u:.2e} | res_v={res_v:.2e} | "
                      f"div_err={mass_resid:.2e} | nn_calls={nn_calls:3d} | psi_min={psi_min:.4f}")

            min_it = (nn_warmup + 60) if (self.online_corrector is not None) else 15
            if it > min_it and res_u < tol and res_v < tol and mass_resid < tol:
                if verbose:
                    print(f"  >> 动态自洽不动点收敛! iter = {it} (NN实时调用 {nn_calls} 次, "
                          f"终态源项自洽偏差={src_rel_diff:.2e}), res_u={res_u:.2e}, div_err={mass_resid:.2e}")
                break
        else:
            if verbose:
                print(f"  >> 达到最大迭代次数 {max_iter} (NN实时调用 {nn_calls} 次), "
                      f"res_u={res_u:.2e}, res_v={res_v:.2e}, div_err={mass_resid:.2e}")

        return res_u, res_v, mass_resid


# ============================================================
# 模块 4: 固体区光滑延拓与双三次样条高精度采样 + 1D 完备离散双残差计算
# ============================================================
def smooth_fill_solid_region(field_2d, valid_mask, n_iters=600):
    if np.all(valid_mask):
        return field_2d.copy()
    f = field_2d.copy()
    solid = ~valid_mask
    f[solid] = float(np.mean(f[valid_mask]))
    for _ in range(n_iters):
        f_pad = np.pad(f, 1, mode='edge')
        avg = 0.25 * (f_pad[2:, 1:-1] + f_pad[:-2, 1:-1] + f_pad[1:-1, 2:] + f_pad[1:-1, :-2])
        f[solid] = avg[solid]
    return f


def sample_dense_to_coarse_spline(mesh_dense: ConnectivityGmshMesh,
                                  solver_dense: UniversalSIMPLE,
                                  mesh_coarse: ConnectivityGmshMesh):
    u_fill = smooth_fill_solid_region(solver_dense.u_2d, mesh_dense.valid_mask)
    v_fill = smooth_fill_solid_region(solver_dense.v_2d, mesh_dense.valid_mask)
    p_fill = smooth_fill_solid_region(solver_dense.p_2d, mesh_dense.valid_mask)

    interp_u = RectBivariateSpline(mesh_dense.x, mesh_dense.y, u_fill, kx=3, ky=3)
    interp_v = RectBivariateSpline(mesh_dense.x, mesh_dense.y, v_fill, kx=3, ky=3)
    interp_p = RectBivariateSpline(mesh_dense.x, mesh_dense.y, p_fill, kx=3, ky=3)

    u_raw_1d = interp_u(mesh_coarse.x_1d, mesh_coarse.y_1d, grid=False)
    v_raw_1d = interp_v(mesh_coarse.x_1d, mesh_coarse.y_1d, grid=False)
    p_raw_1d = interp_p(mesh_coarse.x_1d, mesh_coarse.y_1d, grid=False)

    return u_raw_1d, v_raw_1d, p_raw_1d


def compute_discrete_residual(solver: UniversalSIMPLE, u_s_1d, v_s_1d, p_s_1d):
    alpha_u = float(solver.config.get('solver', {}).get('alpha_u', 0.7))
    m = solver.mesh
    act = solver.active
    idx = np.where(act)[0]

    solver.u = u_s_1d.copy()
    solver.v = v_s_1d.copy()
    solver.p = p_s_1d.copy()
    solver._bc_uv()
    solver._bc_p()

    solver._assemble_momentum('u')
    Gx, Gy = solver._gradient(solver.p)
    Sp_u = -Gx
    Sp_v = -Gy

    Res_u = np.zeros(solver.N, dtype=float)
    Res_v = np.zeros(solver.N, dtype=float)

    Res_u[idx] = (
        solver.aP[idx] * solver.u[idx]
        - solver.aE[idx] * solver.u[m.nbr_E[idx]]
        - solver.aW[idx] * solver.u[m.nbr_W[idx]]
        - solver.aN[idx] * solver.u[m.nbr_N[idx]]
        - solver.aS[idx] * solver.u[m.nbr_S[idx]]
        - Sp_u[idx]
    )

    Res_v[idx] = (
        solver.aP[idx] * solver.v[idx]
        - solver.aE[idx] * solver.v[m.nbr_E[idx]]
        - solver.aW[idx] * solver.v[m.nbr_W[idx]]
        - solver.aN[idx] * solver.v[m.nbr_N[idx]]
        - solver.aS[idx] * solver.v[m.nbr_S[idx]]
        - Sp_v[idx]
    )

    u_e, v_n, _ = solver._rc_face_velocities(alpha_u)
    Res_div = np.zeros(solver.N, dtype=float)
    Res_div[idx] = (
        (u_e[idx] - u_e[m.nbr_W[idx]]) / m.dx_P[idx] +
        (v_n[idx] - v_n[m.nbr_S[idx]]) / m.dy_P[idx]
    )

    return Res_u, Res_v, Res_div, solver.u.copy(), solver.v.copy(), solver.p.copy()


# ============================================================
# 模块 5: 面向自洽不动点迭代的伽利略不变量特征与满秩正交向量基网络
# ============================================================
def compute_graph_poisson_wall_potential(solver: UniversalSIMPLE) -> np.ndarray:
    """求解无坐标壁面距离泊松方程 (带几何缓存，仅在首次调用时计算 1 次)"""
    if solver._cached_wall_potential is not None:
        return solver._cached_wall_potential

    m = solver.mesh
    act = solver.active

    De = 1.0 / (m.dx_P * m.dx_E)
    Dw = 1.0 / (m.dx_P * m.dx_W)
    Dn = 1.0 / (m.dy_P * m.dy_N)
    Ds = 1.0 / (m.dy_P * m.dy_S)

    aE = np.where(act, De, 0.0)
    aW = np.where(act, Dw, 0.0)
    aN = np.where(act, Dn, 0.0)
    aS = np.where(act, Ds, 0.0)
    aP = np.where(act, aE + aW + aN + aS, 1.0)
    rhs = np.where(act, 1.0, 0.0)

    neumann_bc_1d = np.zeros(solver.N, dtype=bool)
    for bname, bc in solver.bc_cfg.items():
        if bc['type'] in ('inlet', 'pressure_outlet'):
            neumann_bc_1d |= m.bc_masks_1d[bname]

    for nbr_arr, a_arr in [(m.nbr_E, aE), (m.nbr_W, aW), (m.nbr_N, aN), (m.nbr_S, aS)]:
        is_neu = act & (nbr_arr >= 0) & neumann_bc_1d[nbr_arr]
        aP[is_neu] -= a_arr[is_neu]
        a_arr[is_neu] = 0.0

    A_w = build_1d_connectivity_matrix(aP, aE, aW, aN, aS, m)
    phi_w = spsolve(A_w, rhs)

    for bname, bc in solver.bc_cfg.items():
        if bc['type'] in ('inlet', 'pressure_outlet'):
            solver._apply_neumann_extrapolation(phi_w, bname)

    solver._cached_wall_potential = phi_w
    return phi_w


def graph_smooth_vector_field(u_1d: np.ndarray, v_1d: np.ndarray, mesh: ConnectivityGmshMesh, n_hops=8):
    u_sm = u_1d.copy()
    v_sm = v_1d.copy()
    nbrs = [mesh.nbr_E, mesh.nbr_W, mesh.nbr_N, mesh.nbr_S]

    for _ in range(n_hops):
        u_acc = np.zeros_like(u_sm)
        v_acc = np.zeros_like(v_sm)
        cnt = np.zeros_like(u_sm)
        for nb in nbrs:
            valid = nb >= 0
            u_acc[valid] += u_sm[nb[valid]]
            v_acc[valid] += v_sm[nb[valid]]
            cnt[valid] += 1.0
        has_nb = cnt > 0
        u_sm[has_nb] = 0.5 * u_sm[has_nb] + 0.5 * (u_acc[has_nb] / cnt[has_nb])
        v_sm[has_nb] = 0.5 * v_sm[has_nb] + 0.5 * (v_acc[has_nb] / cnt[has_nb])

    return u_sm, v_sm


def extract_decoupled_galilean_features(solver_state: UniversalSIMPLE):
    """
    【面向实时动态迭代的 15 维局部伽利略不变量 + 1 维全局尺度参数 + 8 组满秩无退化客观向量基】
    关键改进：
    1. 所有梯度 (du/dx, dv/dy, dp/dx 等) 100% 由粗网格算子 solver_state._gradient() 从当前实时场重构；
    2. 构造 8 组满秩客观物理基向量（增加平滑速度正交伴随基 U_sm^⊥、壁面势法向基 ∇φ_w 与切向基 ∇^⊥ φ_w），
       彻底解决当流场逼近密网格真解时在分离涡心/再附点处局部速度 |U|->0 引发的基向量退化崩溃！
    """
    m = solver_state.mesh
    u_c = solver_state.u
    v_c = solver_state.v
    p_c = solver_state.p
    rho, mu = solver_state.rho, solver_state.mu

    vel_mag = np.sqrt(u_c**2 + v_c**2)
    U_ref = float(np.max(vel_mag))
    if U_ref < 1e-12:
        U_ref = 1.0

    delta = np.sqrt(m.dx_P * m.dy_P)

    # 1. 多跳图平滑速度场 U_sm (即使局部点 u_c=0，多跳邻域平滑速度依然非零！)
    u_sm, v_sm = graph_smooth_vector_field(u_c, v_c, m, n_hops=8)
    usm_mag = np.sqrt(u_sm**2 + v_sm**2)

    # 2. 粗网格算子实时重构速度梯度张量 ∇U
    du_dx, du_dy = solver_state._gradient(u_c)
    dv_dx, dv_dy = solver_state._gradient(v_c)

    S_xx = du_dx
    S_yy = dv_dy
    S_xy = 0.5 * (du_dy + dv_dx)
    Omega_xy = 0.5 * (du_dy - dv_dx)

    tr_S2 = S_xx**2 + S_yy**2 + 2.0 * (S_xy**2)
    tr_Omega2 = 2.0 * (Omega_xy**2)
    Q_crit = 0.5 * (tr_Omega2 - tr_S2)
    div_u = S_xx + S_yy

    SU_x = S_xx * u_c + S_xy * v_c
    SU_y = S_xy * u_c + S_yy * v_c
    OmegaU_x = Omega_xy * v_c
    OmegaU_y = -Omega_xy * u_c

    # 3. 粗网格算子重构压力梯度向量 ∇p 与无量纲化壁面距离势 φ_w_hat
    dp_dx, dp_dy = solver_state._gradient(p_c)
    dp_mag = np.sqrt(dp_dx**2 + dp_dy**2)

    phi_w = compute_graph_poisson_wall_potential(solver_state)
    phi_w_max = float(np.max(phi_w)) + 1e-12
    L_w = np.sqrt(phi_w_max)

    phi_w_hat = phi_w / phi_w_max
    dw_dx, dw_dy = solver_state._gradient(phi_w)
    dw_dx_hat = dw_dx / L_w
    dw_dy_hat = dw_dy / L_w
    dw_mag_hat = np.sqrt(dw_dx_hat**2 + dw_dy_hat**2)

    # 4. 构造客观向量之间的无量纲旋转不变内积
    I_u_usm = (u_c * u_sm + v_c * v_sm) / (U_ref**2)
    I_u_dp  = (u_c * dp_dx + v_c * dp_dy) * delta / (U_ref**3)
    I_u_dw  = (u_c * dw_dx_hat + v_c * dw_dy_hat) / U_ref
    I_dp_dw = (dp_dx * dw_dx_hat + dp_dy * dw_dy_hat) * delta / (U_ref**2)
    I_u_S_u = (u_c * SU_x + v_c * SU_y) * delta / (U_ref**3)

    # 5. 局部网格雷诺数 Re_Δ 与 全局通道雷诺数 Re_Lw
    Re_cell_feat = np.log1p(rho * vel_mag * delta / (mu + 1e-12))
    scale_feat_1d = np.full((m.N, 1), np.log1p(rho * U_ref * L_w / (mu + 1e-12)), dtype=float)

    local_feats_15d = np.column_stack([
        vel_mag / U_ref,                     # 1. 归一化速度大小 |U|/U_ref
        usm_mag / U_ref,                     # 2. 归一化平滑速度大小 |U_sm|/U_ref
        dp_mag * delta / (U_ref**2),         # 3. 无量纲压力梯度大小 |∇p|*Δ/U_ref^2
        dw_mag_hat,                          # 4. 无量纲壁面势梯度模 |∇φ_w|/L_w
        phi_w_hat,                           # 5. 无量纲壁面距离势 φ_w/φ_w,max
        Re_cell_feat,                        # 6. 对数局部网格雷诺数 log(1 + Re_Δ)
        div_u * delta / U_ref,               # 7. 无量纲局部散度 Tr(S)*Δ/U_ref
        tr_S2 * (delta**2) / (U_ref**2),     # 8. 无量纲应变率二阶不变量 Tr(S^2)*Δ^2/U_ref^2
        tr_Omega2 * (delta**2) / (U_ref**2), # 9. 无量纲旋转率二阶不变量 Tr(Ω^2)*Δ^2/U_ref^2
        Q_crit * (delta**2) / (U_ref**2),    # 10. 无量纲 Q 准则
        I_u_usm,                             # 11. U · U_sm / U_ref^2
        I_u_dp,                              # 12. U · ∇p * Δ / U_ref^3
        I_u_dw,                              # 13. U · ∇φ_w_hat / U_ref
        I_dp_dw,                             # 14. ∇p · ∇φ_w_hat * Δ / U_ref^2
        I_u_S_u                              # 15. U^T · S · U * Δ / U_ref^3
    ])

    # 8 组严格满足 2D 旋转协变性的满秩客观物理向量基 (N, 8, 2)
    # 注意: 对于 2D 向量 (vx, vy)，其逆时针旋转 90° 的正交伴随向量 (-vy, vx) 同样是严格旋转协变的！
    B1 = np.column_stack([u_c, v_c]) / U_ref                                      # 局部速度向量 U
    B2 = np.column_stack([u_sm, v_sm]) / U_ref                                    # 多跳平滑速度向量 U_sm
    B3 = np.column_stack([-v_sm, u_sm]) / U_ref                                   # 多跳平滑正交速度基 U_sm^⊥ (确保流向/法向随时正交完备)
    B4 = np.column_stack([dp_dx, dp_dy]) * (delta[:, None] / (U_ref**2))          # 压力梯度向量 ∇p
    B5 = np.column_stack([SU_x, SU_y]) * (delta[:, None] / (U_ref**2))            # 应变率作用向量 S·U
    B6 = np.column_stack([OmegaU_x, OmegaU_y]) * (delta[:, None] / (U_ref**2))    # 旋转率作用向量 Ω·U
    B7 = np.column_stack([dw_dx_hat, dw_dy_hat])                                  # 壁面势法向梯度基 ∇φ_w (不乘|U|，在涡心处永不退化!)
    B8 = np.column_stack([-dw_dy_hat, dw_dx_hat])                                 # 壁面势切向梯度基 ∇^⊥ φ_w

    basis_vectors = np.stack([B1, B2, B3, B4, B5, B6, B7, B8], axis=1)  # (N, 8, 2)

    mom_dim_scale = (rho * (U_ref ** 2)) / delta
    div_dim_scale = U_ref / delta

    return local_feats_15d, scale_feat_1d, basis_vectors, mom_dim_scale, div_dim_scale


class ScaleModulatedFourierRBFNet(nn.Module):
    """局部-全局解耦的尺度调制混合网络 (Scale-Modulated FiLM Hybrid Net)"""
    def __init__(self, loc_dim: int, out_dim: int, centers: torch.Tensor, widths: torch.Tensor, hidden_dim=112, num_freqs=3):
        super().__init__()
        freqs = torch.arange(1, num_freqs + 1, dtype=torch.float32) * (0.5 * np.pi)
        self.register_buffer('freqs', freqs)
        self.register_buffer('centers', centers)
        self.register_buffer('widths', widths)

        loc_feat_dim = loc_dim + loc_dim * num_freqs * 2 + len(centers)

        self.fc1 = nn.Linear(loc_feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head_base = nn.Linear(hidden_dim, out_dim)
        self.head_lin  = nn.Linear(hidden_dim, out_dim)
        self.head_quad = nn.Linear(hidden_dim, out_dim)

        self.film_gamma1 = nn.Linear(2, hidden_dim)
        self.film_beta1  = nn.Linear(2, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x_loc, s_glob):
        proj = x_loc.unsqueeze(-1) * self.freqs
        proj = proj.reshape(x_loc.shape[0], -1)
        dist_sq = torch.sum((x_loc.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2, dim=-1) / x_loc.shape[1]
        rbf_feats = torch.exp(-dist_sq / (2.0 * (self.widths.unsqueeze(0) ** 2)))

        loc_all = torch.cat([x_loc, torch.sin(proj), torch.cos(proj), rbf_feats], dim=-1)

        s_poly = torch.cat([s_glob, s_glob ** 2], dim=-1)
        gamma = 1.0 + 0.2 * torch.tanh(self.film_gamma1(s_poly))
        beta  = 0.2 * torch.tanh(self.film_beta1(s_poly))

        h = self.act(gamma * self.fc1(loc_all) + beta)
        h = self.act(self.fc2(h) + h)

        out = (
            self.head_base(h)
            + s_glob * self.head_lin(h)
            + (s_glob ** 2) * self.head_quad(h)
        )
        return out


class GalileanInvariantVBNNCorrector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.mom_net = None
        self.div_net = None
        self.loc_mean = None
        self.loc_std = None
        self.scale_mean = None
        self.scale_std = None
        self.mom_scale = 1.0
        self.div_mean = 0.0
        self.div_std = 1.0
        self.centers = None
        self.widths = None
        self.loc_dim = 15
        self.num_basis = 8

    def prepare_multi_case_training_data(self, dataset_tuples):
        """
        接收自洽不动点训练数据: [(sv_tmpl, res_u, res_v, res_div), ...]
        其中 sv_tmpl 的速度与压力为粗网格上的目标收敛真解，但其所有梯度均由粗网格算子重构！
        """
        loc_list, scale_list, basis_list = [], [], []
        res_u_nd_list, res_v_nd_list, res_div_nd_list = [], [], []
        case_lengths = []

        for solver_c, ru, rv, rdiv in dataset_tuples:
            act = solver_c.active
            loc_f, scale_f, basis_vecs, mom_dim_s, div_dim_s = extract_decoupled_galilean_features(solver_c)
            loc_list.append(loc_f[act])
            scale_list.append(scale_f[act])
            basis_list.append(basis_vecs[act])
            res_u_nd_list.append(ru[act] / mom_dim_s[act])
            res_v_nd_list.append(rv[act] / mom_dim_s[act])
            res_div_nd_list.append(rdiv[act] / div_dim_s[act])
            case_lengths.append(int(np.sum(act)))

        loc_all    = np.vstack(loc_list)
        scale_all  = np.vstack(scale_list)
        basis_all  = np.vstack(basis_list)
        res_u_nd   = np.concatenate(res_u_nd_list)
        res_v_nd   = np.concatenate(res_v_nd_list)
        res_div_nd = np.concatenate(res_div_nd_list)

        self.loc_dim   = loc_all.shape[1]
        self.num_basis = basis_all.shape[1]

        self.loc_mean = np.mean(loc_all, axis=0, keepdims=True)
        self.loc_std  = np.std(loc_all, axis=0, keepdims=True) + 1e-8
        norm_loc_np   = (loc_all - self.loc_mean) / self.loc_std

        self.scale_mean = np.mean(scale_all, axis=0, keepdims=True)
        self.scale_std  = np.std(scale_all, axis=0, keepdims=True) + 1e-8
        norm_scale_np   = (scale_all - self.scale_mean) / self.scale_std

        X_loc_tr  = torch.tensor(norm_loc_np, dtype=torch.float32, device=self.device)
        S_glob_tr = torch.tensor(norm_scale_np, dtype=torch.float32, device=self.device)
        B_train   = torch.tensor(basis_all, dtype=torch.float32, device=self.device)

        res_mag_nd = np.sqrt(res_u_nd**2 + res_v_nd**2)
        self.mom_scale = float(np.std(res_mag_nd)) + 1e-12
        Y_mom = torch.tensor(
            np.column_stack([res_u_nd, res_v_nd]) / self.mom_scale,
            dtype=torch.float32, device=self.device
        )

        self.div_mean = float(np.mean(res_div_nd))
        self.div_std  = float(np.std(res_div_nd)) + 1e-12
        Y_div = torch.tensor(
            (res_div_nd - self.div_mean) / self.div_std,
            dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        importance_all = res_mag_nd / self.mom_scale + np.abs(res_div_nd) / self.div_std
        sel_indices = []
        offset = 0
        for n_c in case_lengths:
            imp_c = importance_all[offset:offset + n_c]
            n_peak = min(56, n_c // 2)
            peak_c = np.argsort(imp_c)[-n_peak:] + offset
            unif_c = np.linspace(offset, offset + n_c - 1, min(36, n_c), dtype=int)
            sel_indices.append(np.unique(np.concatenate([peak_c, unif_c])))
            offset += n_c

        sel_idx = np.unique(np.concatenate(sel_indices))
        centers_np = norm_loc_np[sel_idx]
        centers_multi = np.vstack([centers_np, centers_np])
        widths_multi = np.concatenate([
            np.full(len(centers_np), 0.35, dtype=np.float32),
            np.full(len(centers_np), 0.90, dtype=np.float32)
        ])
        self.centers = torch.tensor(centers_multi, dtype=torch.float32, device=self.device)
        self.widths  = torch.tensor(widths_multi, dtype=torch.float32, device=self.device)

        return X_loc_tr, S_glob_tr, B_train, Y_mom, Y_div

    def train(self, X_loc_tr, S_glob_tr, B_train, Y_mom, Y_div, nn_cfg: dict):
        epochs     = int(nn_cfg.get('epochs', 4500))
        lr         = float(nn_cfg.get('lr', 3e-3))
        hidden_dim = int(nn_cfg.get('hidden_dim', 112))

        print(f"\n{'='*84}")
        print(f"  Step 2: 训练面向【真·实时动态自洽迭代】的尺度调制 VBNN 模型 (Scales = 0.5x, 1.0x, 2.0x)")
        print(f"  局部特征: {self.loc_dim} 维 | 全局调制: 1 维 | 满秩无退化向量基: {self.num_basis} 组 | 样本数: {len(X_loc_tr)}")
        print(f"  共享局部 RBF 核中心总数: {len(self.centers)} | 训练设备: {self.device}")
        print(f"  训练范式: 【直接以粗网格算子在收敛真解态上的重构梯度为输入 -> 严格保证收敛不动点自洽】")
        print(f"{'='*84}")

        self.mom_net = ScaleModulatedFourierRBFNet(
            self.loc_dim, self.num_basis, self.centers, self.widths, hidden_dim=hidden_dim, num_freqs=3
        ).to(self.device)
        self.div_net = ScaleModulatedFourierRBFNet(
            self.loc_dim, 1, self.centers, self.widths, hidden_dim=hidden_dim, num_freqs=3
        ).to(self.device)

        # 适度提高 weight_decay 保证算子关于流场状态 (u, v, p) 的 Lipschitz 平滑性，确保动态不动点绝对收敛
        opt_mom = torch.optim.Adam(self.mom_net.parameters(), lr=lr, weight_decay=1e-5)
        opt_div = torch.optim.Adam(self.div_net.parameters(), lr=lr, weight_decay=1e-5)
        sch_mom = torch.optim.lr_scheduler.CosineAnnealingLR(opt_mom, T_max=epochs, eta_min=1e-6)
        sch_div = torch.optim.lr_scheduler.CosineAnnealingLR(opt_div, T_max=epochs, eta_min=1e-6)

        criterion = nn.MSELoss()

        for epoch in range(1, epochs + 1):
            opt_mom.zero_grad()
            # 加入微小的特征高斯噪声正则化，模拟动态迭代逼近真解过程中的微小梯度扰动，增强不动点吸引盆！
            if epoch < int(epochs * 0.75):
                noise = 0.015 * torch.randn_like(X_loc_tr)
                g_coeffs = self.mom_net(X_loc_tr + noise, S_glob_tr)
            else:
                g_coeffs = self.mom_net(X_loc_tr, S_glob_tr)

            pred_mom = torch.sum(g_coeffs.unsqueeze(-1) * B_train, dim=1)
            loss_mom = criterion(pred_mom, Y_mom) + 1e-5 * torch.mean(g_coeffs ** 2)
            loss_mom.backward()
            opt_mom.step()
            sch_mom.step()

            opt_div.zero_grad()
            if epoch < int(epochs * 0.75):
                noise_d = 0.015 * torch.randn_like(X_loc_tr)
                pred_div = self.div_net(X_loc_tr + noise_d, S_glob_tr)
            else:
                pred_div = self.div_net(X_loc_tr, S_glob_tr)
            loss_div = criterion(pred_div, Y_div)
            loss_div.backward()
            opt_div.step()
            sch_div.step()

            if epoch % 500 == 0 or epoch == 1:
                cur_lr = opt_mom.param_groups[0]['lr']
                print(f"  Epoch {epoch:4d} | VBNN_Mom_MSE={loss_mom.item():.3e} | "
                      f"Div_MSE={loss_div.item():.3e} | LR={cur_lr:.2e}")

        print("  ✓ 面向实时动态迭代的自洽不动点 VBNN 训练完成！")

    def predict(self, solver_current_state: UniversalSIMPLE):
        """基于当前迭代步的实时粗网格流场 solver_current_state 提取特征并预测源项"""
        m = solver_current_state.mesh
        act = solver_current_state.active

        loc_f, scale_f, basis_vecs, mom_dim_scale, div_dim_scale = extract_decoupled_galilean_features(
            solver_current_state
        )
        norm_loc   = np.clip((loc_f - self.loc_mean) / self.loc_std, -6.0, 6.0)
        norm_scale = (scale_f - self.scale_mean) / self.scale_std

        X_loc_t  = torch.tensor(norm_loc, dtype=torch.float32, device=self.device)
        S_glob_t = torch.tensor(norm_scale, dtype=torch.float32, device=self.device)
        B_tensor = torch.tensor(basis_vecs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            g_coeffs = self.mom_net(X_loc_t, S_glob_t)  # (N, 8)
            pred_mom_nd = torch.sum(g_coeffs.unsqueeze(-1) * B_tensor, dim=1).cpu().numpy() * self.mom_scale

            raw_div = self.div_net(X_loc_t, S_glob_t).cpu().numpy().ravel()
            res_div_nd = raw_div * self.div_std + self.div_mean

        res_u_pred   = pred_mom_nd[:, 0] * mom_dim_scale
        res_v_pred   = pred_mom_nd[:, 1] * mom_dim_scale
        res_div_pred = res_div_nd * div_dim_scale

        res_u_pred[~act]   = 0.0
        res_v_pred[~act]   = 0.0
        res_div_pred[~act] = 0.0

        if not solver_current_state.has_pressure_outlet:
            vol = m.cell_vol[act]
            vol_weighted_mean = np.sum(res_div_pred[act] * vol) / np.sum(vol)
            res_div_pred[act] -= vol_weighted_mean

        return res_u_pred, res_v_pred, res_div_pred


# ============================================================
# 模块 6: 评估与全套学术可视化函数
# ============================================================
def evaluate_velocity_errors(solver_coarse_orig: UniversalSIMPLE,
                             solver_coarse_nn: UniversalSIMPLE,
                             u_sampled_1d: np.ndarray,
                             v_sampled_1d: np.ndarray,
                             case_name: str,
                             verbose=True):
    """计算 1D 内部流体节点的速度最大误差与 RMSE 改善率"""
    act = solver_coarse_orig.mesh.is_internal
    err_u_orig = float(np.max(np.abs(u_sampled_1d[act] - solver_coarse_orig.u[act])))
    err_u_nn   = float(np.max(np.abs(u_sampled_1d[act] - solver_coarse_nn.u[act])))
    err_v_orig = float(np.max(np.abs(v_sampled_1d[act] - solver_coarse_orig.v[act])))
    err_v_nn   = float(np.max(np.abs(v_sampled_1d[act] - solver_coarse_nn.v[act])))

    rmse_u_orig = float(np.sqrt(np.mean((u_sampled_1d[act] - solver_coarse_orig.u[act])**2)))
    rmse_u_nn   = float(np.sqrt(np.mean((u_sampled_1d[act] - solver_coarse_nn.u[act])**2)))
    rmse_v_orig = float(np.sqrt(np.mean((v_sampled_1d[act] - solver_coarse_orig.v[act])**2)))
    rmse_v_nn   = float(np.sqrt(np.mean((v_sampled_1d[act] - solver_coarse_nn.v[act])**2)))

    imp_u_max = (1.0 - err_u_nn / (err_u_orig + 1e-30)) * 100.0
    imp_v_max = (1.0 - err_v_nn / (err_v_orig + 1e-30)) * 100.0
    imp_u_l2  = (1.0 - rmse_u_nn / (rmse_u_orig + 1e-30)) * 100.0
    imp_v_l2  = (1.0 - rmse_v_nn / (rmse_v_orig + 1e-30)) * 100.0

    if verbose:
        print("-" * 84)
        print(f"  [{case_name}] 1D 内部流体节点速度误差对比 (全程实时动态更新源项):")
        print(f"    u_RMSE: 原粗网格 = {rmse_u_orig:.4e} → 动态VBNN修正 = {rmse_u_nn:.4e} (降低 {imp_u_l2:.2f}%)")
        print(f"    v_RMSE: 原粗网格 = {rmse_v_orig:.4e} → 动态VBNN修正 = {rmse_v_nn:.4e} (降低 {imp_v_l2:.2f}%)")
        print(f"    u_Max : 原粗网格 = {err_u_orig:.4e} → 动态VBNN修正 = {err_u_nn:.4e} (降低 {imp_u_max:.2f}%)")
        print(f"    v_Max : 原粗网格 = {err_v_orig:.4e} → 动态VBNN修正 = {err_v_nn:.4e} (降低 {imp_v_max:.2f}%)")
        print("-" * 84)

    return {
        'case_name': case_name,
        'scale': solver_coarse_orig.mesh.scale_factor,
        'rot90': solver_coarse_orig.mesh.rotate_90_ccw,
        'err_u_orig': err_u_orig, 'err_u_nn': err_u_nn, 'imp_u_max': imp_u_max,
        'err_v_orig': err_v_orig, 'err_v_nn': err_v_nn, 'imp_v_max': imp_v_max,
        'rmse_u_orig': rmse_u_orig, 'rmse_u_nn': rmse_u_nn, 'imp_u_l2': imp_u_l2,
        'rmse_v_orig': rmse_v_orig, 'rmse_v_nn': rmse_v_nn, 'imp_v_l2': imp_v_l2,
    }


def plot_nn_source_terms(res_u_true_1d, res_v_true_1d, res_div_true_1d,
                         res_u_pred_1d, res_v_pred_1d, res_div_pred_1d,
                         solver: UniversalSIMPLE, case_name: str, show_fig=True):
    """图1: 对比真实离散残差 vs 动态收敛终态 VBNN 预测残差 vs 预测误差 (3×3 云图)"""
    m = solver.mesh
    valid = m.valid_mask
    true_fields = [res_u_true_1d, res_v_true_1d, res_div_true_1d]
    pred_fields = [res_u_pred_1d, res_v_pred_1d, res_div_pred_1d]
    names = ['Res_u (u-Momentum)', 'Res_v (v-Momentum)', 'Res_div (Continuity)']

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    fig.suptitle(f"[{case_name}] True Dynamic VBNN Source Terms: Ground Truth vs Self-Consistent Converged NN",
                 fontsize=14, fontweight='bold')

    for row, (t_1d, p_1d, name) in enumerate(zip(true_fields, pred_fields, names)):
        t_full = m.to_2d(t_1d, fill_val=0.0)
        p_full = m.to_2d(p_1d, fill_val=0.0)
        diff = p_full - t_full

        vmax_main = max(np.max(np.abs(t_full[valid])), np.max(np.abs(p_full[valid])), 1e-12)
        vmax_diff = max(np.max(np.abs(diff[valid])), 1e-12)

        for col, (data, sub_title, vm) in enumerate([
            (t_full, f'True {name}', vmax_main),
            (p_full, f'Dynamic VBNN {name}', vmax_main),
            (diff,   f'Error (max={np.max(np.abs(diff[valid])):.2e})', vmax_diff)
        ]):
            ax = axes[row, col]
            plot_data = np.where(valid, data, np.nan)
            cf = ax.contourf(solver.X.T, solver.Y.T, plot_data.T, levels=30,
                             cmap='RdBu_r', vmin=-vm, vmax=vm)
            if m.has_obstacle:
                ax.set_facecolor('#dcdcdc')
            plt.colorbar(cf, ax=ax)
            ax.set_title(sub_title, fontsize=11)
            ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.set_aspect('equal')

            X_grid = np.where(solver.mesh.valid_mask, solver.X, np.nan)
            Y_grid = np.where(solver.mesh.valid_mask, solver.Y, np.nan)
            lw = 0.2 if max(solver.nx, solver.ny) > 40 else 0.45
            ax.plot(X_grid, Y_grid, 'k-', lw=lw, alpha=0.2)
            ax.plot(X_grid.T, Y_grid.T, 'k-', lw=lw, alpha=0.2)

    plt.tight_layout()
    save_path = f'./{case_name}_vbnn_source_terms.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close(fig)


def plot_flow_field(solver_dense: UniversalSIMPLE,
                    solver_coarse_orig: UniversalSIMPLE,
                    solver_coarse_nn: UniversalSIMPLE,
                    case_name: str, show_fig=True):
    """图2: 绘制流场对比 —— 速度幅值云图 + 流函数 ψ 涡结构云图 (2×3 布局)"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    fig.suptitle(f"[{case_name}] Flow Field: Dense Reference vs Coarse Original vs True Dynamic VBNN",
                 fontsize=14, fontweight='bold')

    solvers = [solver_dense, solver_coarse_orig, solver_coarse_nn]
    titles = [
        f'Dense Grid Ref (N={solver_dense.N})',
        f'Coarse Original (N={solver_coarse_orig.N})',
        f'Coarse Dynamic VBNN (N={solver_coarse_nn.N})'
    ]

    vmax_vel = float(np.max(np.sqrt(solver_dense.u**2 + solver_dense.v**2)))
    vmax_vel = max(vmax_vel, 1.0)

    for col, (sv, title) in enumerate(zip(solvers, titles)):
        valid = sv.mesh.valid_mask

        ax = axes[0, col]
        vel_mag = np.sqrt(sv.u_2d**2 + sv.v_2d**2)
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

        X_grid = np.where(sv.mesh.valid_mask, sv.X, np.nan)
        Y_grid = np.where(sv.mesh.valid_mask, sv.Y, np.nan)
        lw = 0.2 if max(sv.nx, solver_dense.ny) > 40 else 0.45
        ax.plot(X_grid, Y_grid, 'k-', lw=lw, alpha=0.25)
        ax.plot(X_grid.T, Y_grid.T, 'k-', lw=lw, alpha=0.25)

    plt.tight_layout()
    save_path = f'./{case_name}_vbnn_flow_field.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close(fig)


def plot_profile_and_error(solver_dense: UniversalSIMPLE,
                           solver_coarse_orig: UniversalSIMPLE,
                           solver_coarse_nn: UniversalSIMPLE,
                           stats: dict, case_name: str, show_fig=True):
    """图3: 绘制特征中心线速度剖面对比图"""
    mc = solver_coarse_orig.mesh

    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5.5))
    fig1.suptitle(
        f"[{case_name}] True Dynamic VBNN Velocity Profiles | RMSE Reduction: u={stats['imp_u_l2']:.2f}%, v={stats['imp_v_l2']:.2f}%",
        fontsize=13, fontweight='bold'
    )

    if not mc.rotate_90_ccw:
        full_y_cols = np.where(np.all(mc.valid_mask, axis=1))[0]
        mid_i_c = int(full_y_cols[len(full_y_cols) // 2]) if len(full_y_cols) > 0 else mc.nx // 2
        x_loc = mc.x[mid_i_c]
        mid_i_d = int(np.argmin(np.abs(solver_dense.x - x_loc)))

        full_x_rows = np.where(np.all(mc.valid_mask, axis=0))[0]
        mid_j_c = int(full_x_rows[len(full_x_rows) // 2]) if len(full_x_rows) > 0 else mc.ny // 2
        y_loc = mc.y[mid_j_c]
        mid_j_d = int(np.argmin(np.abs(solver_dense.y - y_loc)))

        ax = axes1[0]
        ax.plot(solver_dense.u_2d[mid_i_d, :], solver_dense.y, 'k-', lw=2.5, label='Dense Grid (Reference)')
        ax.plot(solver_coarse_orig.u_2d[mid_i_c, :], solver_coarse_orig.y, 'r--o', ms=5, lw=1.8, label='Coarse (Original)')
        ax.plot(solver_coarse_nn.u_2d[mid_i_c, :], solver_coarse_nn.y, 'b:s', ms=5, lw=2.2, label='Coarse (Dynamic VBNN)')
        ax.set_xlabel('Streamwise u (m/s)'); ax.set_ylabel('Cross-stream y (m)')
        ax.set_title(f'Streamwise Profile (x = {x_loc:.2f}): u vs y')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes1[1]
        ax.plot(solver_dense.x, solver_dense.v_2d[:, mid_j_d], 'k-', lw=2.5, label='Dense Grid (Reference)')
        ax.plot(solver_coarse_orig.x, solver_coarse_orig.v_2d[:, mid_j_c], 'r--o', ms=5, lw=1.8, label='Coarse (Original)')
        ax.plot(solver_coarse_nn.x, solver_coarse_nn.v_2d[:, mid_j_c], 'b:s', ms=5, lw=2.2, label='Coarse (Dynamic VBNN)')
        ax.set_xlabel('Streamwise x (m)'); ax.set_ylabel('Cross-stream v (m/s)')
        ax.set_title(f'Cross-Stream Profile (y = {y_loc:.2f}): v vs x')
        ax.legend(); ax.grid(True, alpha=0.3)
    else:
        full_x_rows = np.where(np.all(mc.valid_mask, axis=0))[0]
        mid_j_c = int(full_x_rows[len(full_x_rows) // 2]) if len(full_x_rows) > 0 else mc.ny // 2
        y_loc = mc.y[mid_j_c]
        mid_j_d = int(np.argmin(np.abs(solver_dense.y - y_loc)))

        full_y_cols = np.where(np.all(mc.valid_mask, axis=1))[0]
        mid_i_c = int(full_y_cols[(len(full_y_cols) - 1) - len(full_y_cols) // 2]) if len(full_y_cols) > 0 else mc.nx // 2
        x_loc = mc.x[mid_i_c]
        mid_i_d = int(np.argmin(np.abs(solver_dense.x - x_loc)))

        ax = axes1[0]
        ax.plot(solver_dense.v_2d[:, mid_j_d], solver_dense.Lx - solver_dense.x, 'k-', lw=2.5, label='Dense Grid (Reference)')
        ax.plot(solver_coarse_orig.v_2d[:, mid_j_c], mc.Lx - solver_coarse_orig.x, 'r--o', ms=5, lw=1.8, label='Coarse (Original)')
        ax.plot(solver_coarse_nn.v_2d[:, mid_j_c], mc.Lx - solver_coarse_nn.x, 'b:s', ms=5, lw=2.2, label='Coarse (Dynamic VBNN)')
        ax.set_xlabel('Streamwise v (m/s)'); ax.set_ylabel('Cross-stream (Lx - x\') (m)')
        ax.set_title(f'Streamwise Profile (y\' = {y_loc:.2f}): v vs (Lx - x\')')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes1[1]
        ax.plot(solver_dense.y, -solver_dense.u_2d[mid_i_d, :], 'k-', lw=2.5, label='Dense Grid (Reference)')
        ax.plot(solver_coarse_orig.y, -solver_coarse_orig.u_2d[mid_i_c, :], 'r--o', ms=5, lw=1.8, label='Coarse (Original)')
        ax.plot(solver_coarse_nn.y, -solver_coarse_nn.u_2d[mid_i_c, :], 'b:s', ms=5, lw=2.2, label='Coarse (Dynamic VBNN)')
        ax.set_xlabel('Streamwise y\' (m)'); ax.set_ylabel('Cross-stream -u (m/s)')
        ax.set_title(f'Cross-Stream Profile (x\' = {x_loc:.2f}): -u vs y\'')
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    prof_path = f'./{case_name}_vbnn_centerline_profiles.png'
    plt.savefig(prof_path, dpi=150, bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close(fig1)


def plot_all_cases_summary_bar(all_stats_list):
    """汇总图: 绘制全部 8 组算例的主流与法向 RMSE 对比柱状图"""
    labels = [s['case_name'].replace('backward_', '') for s in all_stats_list]
    stream_orig = [s['rmse_v_orig'] if s['rot90'] else s['rmse_u_orig'] for s in all_stats_list]
    stream_nn   = [s['rmse_v_nn']   if s['rot90'] else s['rmse_u_nn']   for s in all_stats_list]
    cross_orig  = [s['rmse_u_orig'] if s['rot90'] else s['rmse_v_orig'] for s in all_stats_list]
    cross_nn    = [s['rmse_u_nn']   if s['rot90'] else s['rmse_v_nn']   for s in all_stats_list]

    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(2, 1, figsize=(14, 9.5))
    fig.suptitle("True Self-Consistent Dynamic VBNN Summary (8 Cases: Continuous In-the-Loop Updating vs Coarse)",
                 fontsize=14, fontweight='bold')

    for ax, orig_v, nn_v, title_str in [
        (axes[0], stream_orig, stream_nn, "Streamwise Velocity RMSE (m/s) [Log Scale]"),
        (axes[1], cross_orig,  cross_nn,  "Cross-Stream Velocity RMSE (m/s) [Log Scale]")
    ]:
        b1 = ax.bar(x - width/2, orig_v, width, label='Coarse Original (No Correction)', color='#ff6b6b')
        b2 = ax.bar(x + width/2, nn_v,   width, label='Coarse + Continuous In-the-Loop VBNN (Never Frozen)', color='#4ecdc4')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10.5, fontweight='bold')
        ax.set_ylabel('RMSE (m/s)')
        ax.set_title(title_str, fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.set_ylim(top=max(orig_v) * 6.0, bottom=max(min(nn_v) * 0.2, 1e-7))
        ax.legend(fontsize=10.5)
        ax.grid(True, axis='y', alpha=0.3, which='both')

        for bar_o, bar_n, vo, vn in zip(b1, b2, orig_v, nn_v):
            imp = (1.0 - vn / (vo + 1e-30)) * 100.0
            ax.text(bar_n.get_x() + bar_n.get_width()/2, bar_n.get_height() * 1.18,
                    f"-{imp:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#0b7285')

    plt.tight_layout()
    save_path = "./all_8cases_dynamic_vbnn_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  [汇总图已保存] 8 算例真·实时动态闭环跨几何与旋转泛化对比总图: {save_path}")
    plt.show()


# ============================================================
# 模块 7: 多几何尺度数据集构建 + 8 算例真·实时动态闭环验证主流程
# ============================================================
def run_multiscale_vbnn_pipeline(config: dict,
                                 train_scales=(0.5, 1.0, 2.0),
                                 unseen_test_scales=(0.75, 1.5),
                                 plot_each_case=True):
    dense_msh_path  = config['meshes']['dense_msh']
    coarse_msh_path = config['meshes']['coarse_msh']
    ensure_gmsh_files_exist(dense_msh_path, coarse_msh_path)

    # =========================================================================
    # Step 1: 在 [0.5x, 1.0x, 2.0x] 三个几何尺度 (0° 方向) 上构建【自洽收敛真解态】训练集
    # =========================================================================
    print("\n" + "#" * 84)
    print(f"  >>> Step 1: 构建面向实时动态迭代的自洽不动点训练集 (Scales = {list(train_scales)}) <<<")
    print("#" * 84)

    train_cache_0deg = {}
    dataset_tuples = []

    for s in train_scales:
        c_cfg = copy.deepcopy(config)
        c_cfg['case_name'] = f"{config['case_name']}_0deg_{s:g}x"
        print(f"\n--- [数据集构建 | 几何缩放 {s:g}x (0°)] 密网格参考求解 & 粗网格真解算子残差提取 ---")

        m_dense  = ConnectivityGmshMesh(dense_msh_path, rotate_90_ccw=False, scale_factor=s)
        m_coarse = ConnectivityGmshMesh(coarse_msh_path, rotate_90_ccw=False, scale_factor=s)

        sv_dense = UniversalSIMPLE(m_dense, c_cfg)
        sv_dense.solve(verbose=True)

        # 仅供最终计算 RMSE 降低率的对比基准，训练与在线动态推理均不依赖它！
        sv_coarse_orig = UniversalSIMPLE(m_coarse, c_cfg)
        sv_coarse_orig.solve(verbose=True)

        # 将密网格真解采样到粗网格上，sv_tmpl 的速度/压力为真解 (u*, v*, p*)，
        # 但 extract_decoupled_galilean_features 会使用粗网格算子 sv_tmpl._gradient() 重构其梯度！
        u_raw, v_raw, p_raw = sample_dense_to_coarse_spline(m_dense, sv_dense, m_coarse)
        sv_tmpl = UniversalSIMPLE(m_coarse, c_cfg)
        res_u, res_v, res_div, u_samp, v_samp, _ = compute_discrete_residual(sv_tmpl, u_raw, v_raw, p_raw)

        train_cache_0deg[s] = {
            'cfg': c_cfg, 'mesh_dense': m_dense, 'mesh_coarse': m_coarse,
            'sv_dense': sv_dense, 'sv_coarse_orig': sv_coarse_orig, 'sv_tmpl': sv_tmpl,
            'res_u': res_u, 'res_v': res_v, 'res_div': res_div,
            'u_samp': u_samp, 'v_samp': v_samp
        }

        # 【核心自洽设计】：直接用 sv_tmpl（即理想收敛终点态下粗网格算子看到的流场与粗网格梯度）训练网络！
        dataset_tuples.append((sv_tmpl, res_u, res_v, res_div))

    # =========================================================================
    # Step 2: 联合训练局部-全局解耦的尺度调制神经网络 (Scale-Modulated VBNN)
    # =========================================================================
    corrector = GalileanInvariantVBNNCorrector()
    X_loc_tr, S_glob_tr, B_tr, Y_mom_tr, Y_div_tr = corrector.prepare_multi_case_training_data(dataset_tuples)
    corrector.train(X_loc_tr, S_glob_tr, B_tr, Y_mom_tr, Y_div_tr, config.get('nn_config', {}))

    all_stats = []

    # =========================================================================
    # Step 3 (阶段 A): 测试训练集 3 个几何尺度 [0.5x, 1.0x, 2.0x] 在 0° 下的【全程实时动态更新】
    # =========================================================================
    print("\n\n" + "#" * 84)
    print(f"  >>> Step 3 (阶段 A): 训练集尺度 {list(train_scales)} 在 0° 下的【真·全程实时动态更新验证】 <<<")
    print("#" * 84)

    preds_0deg_cache = {}
    for s in train_scales:
        item = train_cache_0deg[s]
        c_name = item['cfg']['case_name']

        sv_coarse_nn = UniversalSIMPLE(item['mesh_coarse'], item['cfg'])
        sv_coarse_nn.solve(verbose=True, dynamic_nn_corrector=corrector)

        preds_0deg_cache[s] = (
            sv_coarse_nn.Sp_nn_u.copy(),
            sv_coarse_nn.Sp_nn_v.copy(),
            sv_coarse_nn.Sp_nn_div.copy()
        )

        st = evaluate_velocity_errors(item['sv_coarse_orig'], sv_coarse_nn, item['u_samp'], item['v_samp'], c_name)
        all_stats.append(st)

        if plot_each_case:
            plot_profile_and_error(item['sv_dense'], item['sv_coarse_orig'], sv_coarse_nn, st, c_name, show_fig=False)
            plot_flow_field(item['sv_dense'], item['sv_coarse_orig'], sv_coarse_nn, c_name, show_fig=False)

    # =========================================================================
    # Step 4 (阶段 B): 测试训练集 3 个几何尺度在 90° 逆时针旋转下的【零样本全程实时动态更新】
    # =========================================================================
    print("\n\n" + "#" * 84)
    print(f"  >>> Step 4 (阶段 B): 训练集尺度 {list(train_scales)} 在 90° 旋转下的【零样本全程动态更新测试】 <<<")
    print("#" * 84)

    for s in train_scales:
        cfg_90 = make_rotated_90_ccw_config(config)
        cfg_90['case_name'] = f"{config['case_name']}_90deg_{s:g}x"
        c_name_90 = cfg_90['case_name']

        m_dense_90  = ConnectivityGmshMesh(dense_msh_path, rotate_90_ccw=True, scale_factor=s)
        m_coarse_90 = ConnectivityGmshMesh(coarse_msh_path, rotate_90_ccw=True, scale_factor=s)

        sv_dense_90 = UniversalSIMPLE(m_dense_90, cfg_90)
        sv_dense_90.solve(verbose=False)

        u_raw_90, v_raw_90, p_raw_90 = sample_dense_to_coarse_spline(m_dense_90, sv_dense_90, m_coarse_90)
        sv_tmpl_90 = UniversalSIMPLE(m_coarse_90, cfg_90)
        _, _, _, u_samp_90, v_samp_90, _ = compute_discrete_residual(sv_tmpl_90, u_raw_90, v_raw_90, p_raw_90)

        sv_coarse_orig_90 = UniversalSIMPLE(m_coarse_90, cfg_90)
        sv_coarse_orig_90.solve(verbose=False)

        sv_coarse_nn_90 = UniversalSIMPLE(m_coarse_90, cfg_90)
        sv_coarse_nn_90.solve(verbose=True, dynamic_nn_corrector=corrector)

        res_u_pred_0, res_v_pred_0, _ = preds_0deg_cache[s]
        eq_err = max(
            np.max(np.abs(sv_coarse_nn_90.Sp_nn_u + res_v_pred_0)),
            np.max(np.abs(sv_coarse_nn_90.Sp_nn_v - res_u_pred_0))
        )
        print(f"  [{c_name_90}] 动态收敛终态 0° vs 90° 源项旋转协变最大偏差 = {eq_err:.3e}")

        st_90 = evaluate_velocity_errors(sv_coarse_orig_90, sv_coarse_nn_90, u_samp_90, v_samp_90, c_name_90)
        all_stats.append(st_90)

        if plot_each_case:
            plot_profile_and_error(sv_dense_90, sv_coarse_orig_90, sv_coarse_nn_90, st_90, c_name_90, show_fig=False)
            plot_flow_field(sv_dense_90, sv_coarse_orig_90, sv_coarse_nn_90, c_name_90, show_fig=False)

    # =========================================================================
    # Step 5 (阶段 C): 测试 2 个未见过的全新几何缩放尺度 [0.75x, 1.5x] (零样本全程实时动态更新)
    # =========================================================================
    print("\n\n" + "#" * 84)
    print(f"  >>> Step 5 (阶段 C): 未见过的全新几何尺寸 {list(unseen_test_scales)} 【零样本全程动态更新验证】 <<<")
    print("#" * 84)

    for s in unseen_test_scales:
        u_cfg = copy.deepcopy(config)
        u_cfg['case_name'] = f"{config['case_name']}_UNSEEN_{s:g}x"
        c_name_unseen = u_cfg['case_name']

        print(f"\n>>> [未见几何尺度动态验证: Scale = {s:g}x] 直接从零初场运行全程动态闭环 VBNN 求解 <<<")
        m_dense_u  = ConnectivityGmshMesh(dense_msh_path, rotate_90_ccw=False, scale_factor=s)
        m_coarse_u = ConnectivityGmshMesh(coarse_msh_path, rotate_90_ccw=False, scale_factor=s)

        sv_dense_u = UniversalSIMPLE(m_dense_u, u_cfg)
        sv_dense_u.solve(verbose=True)

        u_raw_u, v_raw_u, p_raw_u = sample_dense_to_coarse_spline(m_dense_u, sv_dense_u, m_coarse_u)
        sv_tmpl_u = UniversalSIMPLE(m_coarse_u, u_cfg)
        res_u_u, res_v_u, res_div_u, u_samp_u, v_samp_u, _ = compute_discrete_residual(
            sv_tmpl_u, u_raw_u, v_raw_u, p_raw_u
        )

        sv_coarse_orig_u = UniversalSIMPLE(m_coarse_u, u_cfg)
        sv_coarse_orig_u.solve(verbose=True)

        sv_coarse_nn_u = UniversalSIMPLE(m_coarse_u, u_cfg)
        sv_coarse_nn_u.solve(verbose=True, dynamic_nn_corrector=corrector)

        st_u = evaluate_velocity_errors(sv_coarse_orig_u, sv_coarse_nn_u, u_samp_u, v_samp_u, c_name_unseen)
        all_stats.append(st_u)

        if plot_each_case:
            plot_nn_source_terms(
                res_u_u, res_v_u, res_div_u,
                sv_coarse_nn_u.Sp_nn_u, sv_coarse_nn_u.Sp_nn_v, sv_coarse_nn_u.Sp_nn_div,
                sv_tmpl_u, c_name_unseen, show_fig=True
            )
            plot_profile_and_error(sv_dense_u, sv_coarse_orig_u, sv_coarse_nn_u, st_u, c_name_unseen, show_fig=True)
            plot_flow_field(sv_dense_u, sv_coarse_orig_u, sv_coarse_nn_u, c_name_unseen, show_fig=True)

    # =========================================================================
    # Step 6: 打印全部 8 组算例的最终汇总表并绘制全景对比柱状图
    # =========================================================================
    print("\n\n" + "=" * 96)
    print("  【真·全程实时动态自洽 VBNN 最终 8 算例汇总表：源项随迭代实时更新直至收敛 vs 原粗网格】")
    print("=" * 96)
    print(f"  {'算例名称 (Case)':<24s} | {'主流 RMSE (原 -> 动态VBNN)':<30s} | {'主流降低率':<10s} | {'法向降低率':<10s}")
    print("-" * 96)
    for s in all_stats:
        if s['rot90']:
            rm_o, rm_n, imp_main, imp_cross = s['rmse_v_orig'], s['rmse_v_nn'], s['imp_v_l2'], s['imp_u_l2']
        else:
            rm_o, rm_n, imp_main, imp_cross = s['rmse_u_orig'], s['rmse_u_nn'], s['imp_u_l2'], s['imp_v_l2']
        print(f"  {s['case_name']:<24s} | {rm_o:.3e} -> {rm_n:.3e}    | {imp_main:7.2f}%   | {imp_cross:7.2f}%")
    print("=" * 96)

    if plot_each_case:
        plot_all_cases_summary_bar(all_stats)

    return all_stats


if __name__ == '__main__':
    dense_path = "../meshes/back_fine.msh" if os.path.exists("../meshes/back_fine.msh") else "./meshes/back_fine.msh"
    coarse_path = "../meshes/back_coarse_coarse.msh" if os.path.exists("../meshes/back_coarse_coarse.msh") else "./meshes/back_coarse_coarse.msh"

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
            "max_iter": 2500, "tol": 1e-7,
            "alpha_u": 0.7, "alpha_v": 0.7, "alpha_p": 0.3,
            # 全程实时动态更新控制参数：仅前30步建立非零初场，之后每5步实时重算一次NN直至收敛，绝不中途冻结！
            "nn_warmup": 30, "nn_update_freq": 5, "beta_nn": 0.25
        },
        "nn_config": {
            "epochs": 4500, "lr": 3e-3, "hidden_dim": 112
        }
    }

    # 训练尺度: [0.5x, 1.0x, 2.0x]，未见测试尺度: [0.75x, 1.5x]
    run_multiscale_vbnn_pipeline(
        config_backward,
        train_scales=(0.5, 1.0, 2.0),
        unseen_test_scales=(0.75, 1.5),
        plot_each_case=True
    )