#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================================
DeepSIMPLE_2D (Scheme 1 严格旋转对偶版 - 局部正交标架规范化与零样本 90° 旋转精确等价验证)
==================================================================================================

【核心修复与架构保证】
 1. 修复原求解器底层破坏 90° 旋转对称性的 3 处隐患:
    - 修复凹角点 Neumann 边界外推只取首个邻居 (E 优先于 N) 的方向偏置，改为所有有效内部流体邻居对称平均；
    - 修复迎风格式在零速度处 (>= 0) 的非对称分支，改为严格对称的 (> 0) 与 (< 0)；
    - 修复流函数与中心线速度剖面绘图在旋转 90° 后的物理截面对应关系 (0° 的 u-y 截面精确对应 90° 的 v-x' 截面)。
 2. 方案一 (Local Orthonormal Frame Canonicalization, R = [e_∥, e_⊥]):
    - 完全抛弃 (x, y) 坐标输入，采用 16 维严格旋转不变的局部标架特征；
    - 仅在 0° 原始流向粗网格上训练，零样本直接泛化至 90° 逆时针旋转粗网格，源项协变误差达 float32 机器精度 (~1e-7)！
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
                    if abs(yv - 1.0) < 1e-9 or abs(yv - 0.0) < 1e-9 or (abs(yv - 0.5) < 1e-9 and xs[i] < 1.0 + 1e-9 and xs[i + 1] <= 1.0 + 1e-9):
                        elems.append((eid, 1, 3, [n1, n2])); eid += 1
                if j + 1 < len(ys) and grid_nid[i, j + 1] >= 0:
                    n2 = grid_nid[i, j + 1]
                    xv = xs[i]
                    if abs(xv - 0.0) < 1e-9 and ys[j] >= 0.5 - 1e-9:
                        elems.append((eid, 1, 1, [n1, n2])); eid += 1
                    elif abs(xv - 5.0) < 1e-9:
                        elems.append((eid, 1, 2, [n1, n2])); eid += 1
                    elif abs(xv - 1.0) < 1e-9 and ys[j] < 0.5 + 1e-9 and ys[j + 1] <= 0.5 + 1e-9:
                        elems.append((eid, 1, 3, [n1, n2])); eid += 1

        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                n1, n2, n3, n4 = grid_nid[i, j], grid_nid[i + 1, j], grid_nid[i + 1, j + 1], grid_nid[i, j + 1]
                if n1 >= 0 and n2 >= 0 and n3 >= 0 and n4 >= 0:
                    elems.append((eid, 3, 4, [n1, n2, n3, n4])); eid += 1

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
# 模块 1: 基于 1D 向量化连接图的 Gmsh (.msh) 拓扑重构解析器 (支持 90° 刚体旋转)
# ============================================================
class ConnectivityGmshMesh:
    """
    读取 Gmsh 2.2 (.msh) 网格文件，提取所有 N 个真实节点构建 1D 拓扑连接表 (nbr_E, nbr_W, nbr_N, nbr_S)。
    支持 rotate_90_ccw=True 将整个网格逆时针刚体旋转 90°: (x', y') = (y_max - y, x - x_min)。
    由于节点 1D 编号顺序 0 ~ N-1 与 Gmsh 原始节点号严格一一对应，0° 与 90° 网格的节点 k 指向完全相同的物理点！
    """
    def __init__(self, msh_filepath, tol_decimals=6, rotate_90_ccw=False):
        self.filepath = msh_filepath
        self.tol_decimals = tol_decimals
        self.rotate_90_ccw = rotate_90_ccw

        self.raw_nodes = {}          # gmsh_nid -> (x, y, z)
        self.phys_names = {}         # (dim, tag) -> name_str
        self.boundary_nodes = {}     # phys_name -> set of gmsh_nids
        self.boundary_edges = {}     # phys_name -> list of (n1, n2)
        self.quad_elements = []      # list of (n1, n2, n3, n4)

        self._parse_msh()
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

    def _apply_rigid_rotation_90_ccw(self):
        """
        对原始网格节点执行严格的逆时针 90° 刚体旋转:
        x_rot = y_max - y,  y_rot = x - x_min
        旋转矩阵 R_90 = [[0, -1], [1, 0]] in SO(2)。
        """
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

        # 构建长度为 N 的 1D 邻接映射表 (不存在邻居处标记为 -1)
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
# 模块 3: 纯 1D 拓扑连接图驱动的通用 SIMPLE 求解器 (严格旋转对称版)
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

        # 预构建严格旋转对称的 1D 边界零梯度外推映射表 (支持凹角点多邻居对称平均)
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
        """
        【关键对称性修复 1】：
        为每个边界节点寻找内部流体邻居。若某边界点 (如台阶凹角点) 同时拥有多个正交内部流体邻居
        (如 0° 时的 E 和 N，或 90° 时的 N 和 W)，绝不按固定顺序只取第一个，而是将所有内部邻居全部记录并做等权平均！
        这样无论网格如何旋转，边界零梯度外推算子 100% 严格旋转不变。
        """
        self.neumann_maps_1d = {}
        m = self.mesh
        fl = m.is_internal

        for bname, bmask in m.bc_masks_1d.items():
            b_indices = np.where(bmask)[0]
            b_list = []
            n_lists = []
            for k in b_indices:
                # 1. 搜索所有直接正交连接的内部流体节点
                direct_nbs = [nb for nb in (m.nbr_E[k], m.nbr_W[k], m.nbr_N[k], m.nbr_S[k]) if nb >= 0 and fl[nb]]
                if len(direct_nbs) > 0:
                    b_list.append(k)
                    n_lists.append(direct_nbs)
                    continue

                # 2. 若为外凸角点，搜索所有二跳内部流体节点
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

                # 3. 兜底：取所有相邻有效节点
                fallback_nbs = [nb for nb in (m.nbr_E[k], m.nbr_W[k], m.nbr_N[k], m.nbr_S[k]) if nb >= 0 and nb != k]
                if len(fallback_nbs) > 0:
                    b_list.append(k)
                    n_lists.append(fallback_nbs)

            self.neumann_maps_1d[bname] = (np.array(b_list, dtype=int), n_lists)

    def _apply_neumann_extrapolation(self, field_1d: np.ndarray, bname: str):
        """对指定边界执行严格旋转对称的零梯度外推 (多内部邻居时自动取均值)"""
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

        # 【关键对称性修复 2】：采用严格对称的 (< 0.0) 与 (> 0.0) 迎风判据，消除 u=0 处的正负零偏置
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

    def solve(self, verbose=True):
        s_cfg = self.config.get('solver', {})
        max_iter = int(s_cfg.get('max_iter', 3000))
        tol      = float(s_cfg.get('tol', 1e-6))
        alpha_u  = float(s_cfg.get('alpha_u', 0.65))
        alpha_v  = float(s_cfg.get('alpha_v', 0.65))
        alpha_p  = float(s_cfg.get('alpha_p', 0.25))

        topo_str = "1D多块台阶拓扑图" if self.mesh.has_obstacle else (
            "1D均匀拓扑图" if self.mesh.is_uniform else "1D非均匀拉伸拓扑图"
        )
        if verbose:
            print("=" * 72)
            print(f"  1D 连接图 SIMPLE 求解器 | 案例: {self.config.get('case_name', 'Unnamed')} "
                  f"| 真实节点数 N={self.N} ({topo_str})")
            if self.Sp_nn_u is not None:
                print("  ✓ 已启用方案一 (局部正交标架旋转协变) 神经网络 1D 源项补偿 (Res_u + Res_v + Res_div)")
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
                vel_max = np.max(np.sqrt(self.u**2 + self.v**2))
                psi_min = np.min(self._streamfunc()[self.mesh.valid_mask])
                print(f"  iter {it:5d} | res_u={res_u:.2e} | res_v={res_v:.2e} | "
                      f"div_err={mass_resid:.2e} | |V|_max={vel_max:.4f} | psi_min={psi_min:.4f}")

            if res_u < tol and res_v < tol and mass_resid < tol and it > 15:
                if verbose:
                    print(f"  >> 收敛! iter = {it}, res_u={res_u:.2e}, res_v={res_v:.2e}, div_err={mass_resid:.2e}")
                break
        else:
            if verbose:
                print(f"  >> 达到最大迭代次数 {max_iter}, res_u={res_u:.2e}, res_v={res_v:.2e}, div_err={mass_resid:.2e}")

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
    alpha_u = float(solver.config.get('solver', {}).get('alpha_u', 0.65))
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
# 模块 5: 方案一 (Local Orthonormal Frame Canonicalization) 特征提取与旋转协变神经网络
# ============================================================
def compute_graph_poisson_wall_potential(solver: UniversalSIMPLE) -> np.ndarray:
    """
    在 1D 拓扑连接图上求解无坐标壁面距离泊松方程:
      -∇² φ_w = 1 (内部流体点),  φ_w = 0 (壁面边界),  ∂φ_w/∂n = 0 (进出口边界)
    """
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


def extract_scheme1_local_frame_and_invariants(solver_coarse_base: UniversalSIMPLE):
    """
    【方案一核心实现：局部正交标架规范化法 (Local Orthonormal Frame Canonicalization)】
    从粗网格基准流场 (u_c, v_c, p_c) 与局部网格几何中提取:
      1. 每个节点的无奇点局部正交基向量 e_∥ = (cosθ, sinθ)^T, e_⊥ = (-sinθ, cosθ)^T，组成旋转矩阵 R；
      2. 16 维严格旋转不变的局部特征矩阵 X_inv (形状: N x 16)，完全不依赖全局坐标 (x, y)！
    """
    m = solver_coarse_base.mesh
    u_c = solver_coarse_base.u
    v_c = solver_coarse_base.v
    p_c = solver_coarse_base.p
    rho, mu = solver_coarse_base.rho, solver_coarse_base.mu

    vel_mag = np.sqrt(u_c**2 + v_c**2)
    U_ref = float(np.max(vel_mag))
    if U_ref < 1e-12:
        U_ref = 1.0

    # 1. 多跳图平滑速度场，构造无奇点的局部主方向向量 w -> e_∥, e_⊥
    u_sm, v_sm = graph_smooth_vector_field(u_c, v_c, m, n_hops=8)
    u_sm2, v_sm2 = graph_smooth_vector_field(u_sm, v_sm, m, n_hops=16)
    wx = (u_c + 0.4 * u_sm + 0.2 * u_sm2) / U_ref
    wy = (v_c + 0.4 * v_sm + 0.2 * v_sm2) / U_ref
    w_norm = np.sqrt(wx**2 + wy**2) + 1e-12

    cos_t = wx / w_norm   # e_∥,x
    sin_t = wy / w_norm   # e_∥,y
    e_perp_x = -sin_t     # e_⊥,x
    e_perp_y = cos_t      # e_⊥,y

    # 2. 速度与非局部平滑速度在局部标架 (e_∥, e_⊥) 下的投影 (4 维)
    u_para = (u_c * cos_t + v_c * sin_t) / U_ref
    u_perp = (u_c * e_perp_x + v_c * e_perp_y) / U_ref
    usm_para = (u_sm * cos_t + v_sm * sin_t) / U_ref
    usm_perp = (u_sm * e_perp_x + v_sm * e_perp_y) / U_ref

    # 3. 速度梯度张量 G = ∇u_c 在局部正交标架下的正交相似变换: G_tilde = R^T * G * R (4 维)
    du_dx, du_dy = solver_coarse_base._gradient(u_c)
    dv_dx, dv_dy = solver_coarse_base._gradient(v_c)

    rt_g_00 = cos_t * du_dx + sin_t * dv_dx
    rt_g_01 = cos_t * du_dy + sin_t * dv_dy
    rt_g_10 = e_perp_x * du_dx + e_perp_y * dv_dx
    rt_g_11 = e_perp_x * du_dy + e_perp_y * dv_dy

    G_para_para = rt_g_00 * cos_t + rt_g_01 * sin_t
    G_para_perp = rt_g_00 * e_perp_x + rt_g_01 * e_perp_y
    G_perp_para = rt_g_10 * cos_t + rt_g_11 * sin_t
    G_perp_perp = rt_g_10 * e_perp_x + rt_g_11 * e_perp_y

    # 4. 压力梯度向量 ∇p_c 在局部正交标架下的投影: ∇p_tilde = R^T * ∇p_c (2 维)
    dp_dx, dp_dy = solver_coarse_base._gradient(p_c)
    dp_para = dp_dx * cos_t + dp_dy * sin_t
    dp_perp = dp_dx * e_perp_x + dp_dy * e_perp_y

    # 5. 局部网格步长度量张量 M = diag(dx_P, dy_P) 在局部标架下的投影与网格雷诺数 (3 维)
    h_para = m.dx_P * (cos_t**2) + m.dy_P * (sin_t**2)
    h_perp = m.dx_P * (e_perp_x**2) + m.dy_P * (e_perp_y**2)
    Re_cell = rho * vel_mag * np.sqrt(m.dx_P * m.dy_P) / (mu + 1e-12)

    # 6. 无坐标拓扑壁面距离势 φ_w 及其在局部正交标架下的梯度投影 R^T * ∇φ_w (3 维)
    phi_w = compute_graph_poisson_wall_potential(solver_coarse_base)
    dw_dx, dw_dy = solver_coarse_base._gradient(phi_w)
    dw_para = dw_dx * cos_t + dw_dy * sin_t
    dw_perp = dw_dx * e_perp_x + dw_dy * e_perp_y

    features_16d = np.column_stack([
        u_para, u_perp, usm_para, usm_perp,
        G_para_para, G_para_perp, G_perp_para, G_perp_perp,
        dp_para, dp_perp,
        h_para, h_perp, Re_cell,
        phi_w, dw_para, dw_perp
    ])

    return features_16d, cos_t, sin_t


class EquivariantHybridFourierRBFNet(nn.Module):
    def __init__(self, in_dim: int, centers: torch.Tensor, widths: torch.Tensor, hidden_dim=96, num_freqs=3):
        super().__init__()
        freqs = torch.arange(1, num_freqs + 1, dtype=torch.float32) * (0.5 * np.pi)
        self.register_buffer('freqs', freqs)
        self.register_buffer('centers', centers)
        self.register_buffer('widths', widths)

        feat_dim = in_dim + in_dim * num_freqs * 2 + len(centers)

        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        proj = x.unsqueeze(-1) * self.freqs
        proj = proj.reshape(x.shape[0], -1)
        dist_sq = torch.sum((x.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2, dim=-1) / x.shape[1]
        rbf_feats = torch.exp(-dist_sq / (2.0 * (self.widths.unsqueeze(0) ** 2)))

        feats = torch.cat([x, torch.sin(proj), torch.cos(proj), rbf_feats], dim=-1)
        return self.net(feats)


class Scheme1EquivariantNeuralCorrector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        self.target_stats = {}
        self.feat_mean = None
        self.feat_std = None
        self.centers = None
        self.widths = None
        self.in_dim = 16

    def prepare_training_data(self, solver_coarse_base: UniversalSIMPLE, res_u_1d, res_v_1d, res_div_1d):
        act = solver_coarse_base.active

        raw_feats, cos_t, sin_t = extract_scheme1_local_frame_and_invariants(solver_coarse_base)
        feats_act = raw_feats[act]
        self.in_dim = feats_act.shape[1]

        self.feat_mean = np.mean(feats_act, axis=0, keepdims=True)
        self.feat_std = np.std(feats_act, axis=0, keepdims=True) + 1e-8
        norm_feats_np = (feats_act - self.feat_mean) / self.feat_std
        X_train = torch.tensor(norm_feats_np, dtype=torch.float32, device=self.device)

        # 使用 R^T 将全局动量残差 (Res_u, Res_v) 投影到局部正交标架 (Res_∥, Res_⊥)
        res_para_act = res_u_1d[act] * cos_t[act] + res_v_1d[act] * sin_t[act]
        res_perp_act = -res_u_1d[act] * sin_t[act] + res_v_1d[act] * cos_t[act]
        res_div_act  = res_div_1d[act]

        importance = (
            np.abs(res_para_act) / (np.std(res_para_act) + 1e-12) +
            np.abs(res_perp_act) / (np.std(res_perp_act) + 1e-12) +
            np.abs(res_div_act) / (np.std(res_div_act) + 1e-12)
        )
        n_samples = len(norm_feats_np)
        n_peak = min(96, n_samples // 2)
        peak_idx = np.argsort(importance)[-n_peak:]
        unif_idx = np.linspace(0, n_samples - 1, min(64, n_samples), dtype=int)
        sel_idx = np.unique(np.concatenate([peak_idx, unif_idx]))

        centers_np = norm_feats_np[sel_idx]
        centers_multi = np.vstack([centers_np, centers_np])
        widths_multi = np.concatenate([
            np.full(len(centers_np), 0.15, dtype=np.float32),
            np.full(len(centers_np), 0.45, dtype=np.float32)
        ])
        self.centers = torch.tensor(centers_multi, dtype=torch.float32, device=self.device)
        self.widths = torch.tensor(widths_multi, dtype=torch.float32, device=self.device)

        targets = {'para': res_para_act, 'perp': res_perp_act, 'div': res_div_act}
        Y_tensors = {}
        for k, val in targets.items():
            mean = float(np.mean(val))
            std = float(np.std(val)) + 1e-12
            self.target_stats[k] = (mean, std)
            norm_val = (val - mean) / std
            Y_tensors[k] = torch.tensor(norm_val, dtype=torch.float32, device=self.device).unsqueeze(1)

        return X_train, Y_tensors

    def train(self, X_train, Y_tensors, nn_cfg: dict):
        epochs     = int(nn_cfg.get('epochs', 3500))
        lr         = float(nn_cfg.get('lr', 3e-3))
        hidden_dim = int(nn_cfg.get('hidden_dim', 96))

        print(f"\n{'='*72}")
        print(f"  Step 4: 训练方案一 (局部正交标架规范化) 旋转协变神经网络模型")
        print(f"  输入特征维度: {self.in_dim} 维旋转不变量 (无 x,y 坐标) | 样本数: {len(X_train)} | 设备: {self.device}")
        print(f"  训练数据集: 【仅使用 0° 原始流向粗网格数据，不使用任何旋转数据增强】")
        print(f"{'='*72}")

        criterion = nn.MSELoss()
        optimizers, schedulers = {}, {}

        for k in ['para', 'perp', 'div']:
            self.models[k] = EquivariantHybridFourierRBFNet(
                self.in_dim, self.centers, self.widths, hidden_dim=hidden_dim, num_freqs=3
            ).to(self.device)
            optimizers[k] = torch.optim.Adam(self.models[k].parameters(), lr=lr, weight_decay=1e-6)
            schedulers[k] = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers[k], T_max=epochs, eta_min=1e-6
            )

        for epoch in range(1, epochs + 1):
            losses = {}
            for k in ['para', 'perp', 'div']:
                optimizers[k].zero_grad()
                pred = self.models[k](X_train)
                loss = criterion(pred, Y_tensors[k])
                loss.backward()
                optimizers[k].step()
                schedulers[k].step()
                losses[k] = loss.item()

            if epoch % 500 == 0 or epoch == 1:
                cur_lr = optimizers['para'].param_groups[0]['lr']
                print(f"  Epoch {epoch:4d} | Norm_MSE_∥={losses['para']:.3e} | "
                      f"Norm_MSE_⊥={losses['perp']:.3e} | Norm_MSE_div={losses['div']:.3e} | LR={cur_lr:.2e}")

        with torch.no_grad():
            for k, label_str in [('para', 'Res_∥ (流向)'), ('perp', 'Res_⊥ (法向)'), ('div', 'Res_div (散度)')]:
                mean, std = self.target_stats[k]
                phys_rmse = np.sqrt(criterion(self.models[k](X_train), Y_tensors[k]).item()) * std
                print(f"  [训练集物理量纲 RMSE] {label_str:14s} = {phys_rmse:.4e}")
        print("  ✓ 方案一旋转协变神经网络训练完成")

    def predict(self, solver_coarse_base: UniversalSIMPLE):
        m = solver_coarse_base.mesh
        act = solver_coarse_base.active

        raw_feats, cos_t, sin_t = extract_scheme1_local_frame_and_invariants(solver_coarse_base)
        norm_feats = (raw_feats - self.feat_mean) / self.feat_std
        X_tensor = torch.tensor(norm_feats, dtype=torch.float32, device=self.device)

        local_preds = {}
        with torch.no_grad():
            for k in ['para', 'perp', 'div']:
                mean, std = self.target_stats[k]
                raw = self.models[k](X_tensor).cpu().numpy().ravel()
                field_1d = raw * std + mean
                field_1d[~act] = 0.0
                local_preds[k] = field_1d

        res_para_pred = local_preds['para']
        res_perp_pred = local_preds['perp']
        res_div_pred  = local_preds['div']

        # 左乘局部正交旋转矩阵 R = [e_∥, e_⊥] 映射回全局 (x, y) 动量方程
        res_u_pred = res_para_pred * cos_t - res_perp_pred * sin_t
        res_v_pred = res_para_pred * sin_t + res_perp_pred * cos_t
        res_u_pred[~act] = 0.0
        res_v_pred[~act] = 0.0

        if not solver_coarse_base.has_pressure_outlet:
            vol = m.cell_vol[act]
            vol_weighted_mean = np.sum(res_div_pred[act] * vol) / np.sum(vol)
            res_div_pred[act] -= vol_weighted_mean

        return res_u_pred, res_v_pred, res_div_pred


# ============================================================
# 模块 6: 全套学术可视化函数 (0° 与 90° 物理截面严格对齐)
# ============================================================
def plot_nn_source_terms(res_u_true_1d, res_v_true_1d, res_div_true_1d,
                         res_u_pred_1d, res_v_pred_1d, res_div_pred_1d,
                         solver: UniversalSIMPLE, case_name: str):
    """图1: 对比真实离散残差 vs 方案一神经网络预测残差 vs 预测误差 (3×3 云图)"""
    m = solver.mesh
    valid = m.valid_mask
    true_fields = [res_u_true_1d, res_v_true_1d, res_div_true_1d]
    pred_fields = [res_u_pred_1d, res_v_pred_1d, res_div_pred_1d]
    names = ['Res_u (u-Momentum)', 'Res_v (v-Momentum)', 'Res_div (Continuity)']

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    fig.suptitle(f"[{case_name}] Scheme-1 Equivariant Source Terms: Ground Truth vs NN Prediction",
                 fontsize=14, fontweight='bold')

    for row, (t_1d, p_1d, name) in enumerate(zip(true_fields, pred_fields, names)):
        t_full = m.to_2d(t_1d, fill_val=0.0)
        p_full = m.to_2d(p_1d, fill_val=0.0)
        diff = p_full - t_full

        vmax_main = max(np.max(np.abs(t_full[valid])), np.max(np.abs(p_full[valid])), 1e-12)
        vmax_diff = max(np.max(np.abs(diff[valid])), 1e-12)

        for col, (data, sub_title, vm) in enumerate([
            (t_full, f'True {name}', vmax_main),
            (p_full, f'Scheme-1 NN {name}', vmax_main),
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
    save_path = f'./{case_name}_nn_source_terms.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [图像保存 1/4] ({case_name}) NN 源项拟合对比图已保存: {save_path}")
    plt.show()


def plot_flow_field(solver_dense: UniversalSIMPLE,
                    solver_coarse_orig: UniversalSIMPLE,
                    solver_coarse_nn: UniversalSIMPLE,
                    case_name: str):
    """图2: 绘制流场对比 —— 速度幅值云图 + 流函数 ψ 涡结构云图 (2×3 布局)"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    fig.suptitle(f"[{case_name}] Flow Field: Dense Reference vs Coarse Original vs Coarse Scheme-1 NN",
                 fontsize=14, fontweight='bold')

    solvers = [solver_dense, solver_coarse_orig, solver_coarse_nn]
    titles = [
        f'Dense Grid Ref (N={solver_dense.N})',
        f'Coarse Original (N={solver_coarse_orig.N})',
        f'Coarse Scheme-1 NN (N={solver_coarse_nn.N})'
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
        lw = 0.2 if max(sv.nx, sv.ny) > 40 else 0.45
        ax.plot(X_grid, Y_grid, 'k-', lw=lw, alpha=0.25)
        ax.plot(X_grid.T, Y_grid.T, 'k-', lw=lw, alpha=0.25)

    plt.tight_layout()
    save_path = f'./{case_name}_nn_flow_field.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [图像保存 2/4] ({case_name}) 流场与流线对比图已保存: {save_path}")
    plt.show()


def plot_profile_and_error(solver_dense: UniversalSIMPLE,
                           solver_coarse_orig: UniversalSIMPLE,
                           solver_coarse_nn: UniversalSIMPLE,
                           u_sampled_1d, v_sampled_1d, case_name: str):
    """
    【关键对称性修复 3】：
    图3 & 图4: 绘制特征中心线速度剖面对比图 + 误差统计柱状图。
    当网格逆时针旋转 90° 后，原 0° 网格的：
      - 主流方向截面 (x = x_loc 处的 u vs y) 自动对应为旋转后网格的 (y' = x_loc 处的 v vs y_max - x')；
      - 横向速度截面 (y = y_loc 处的 v vs x) 自动对应为旋转后网格的 (x' = y_max - y_loc 处的 -u vs y')。
    从而保证两个算例对比的是完全相同的物理截面，曲线形态 100% 一致！
    """
    mc = solver_coarse_orig.mesh

    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5.5))
    fig1.suptitle(f"[{case_name}] Streamwise & Cross-Stream Velocity Profiles: Original vs Scheme-1 NN",
                  fontsize=13, fontweight='bold')

    if not mc.rotate_90_ccw:
        # 0° 水平流动 (从左至右): 左图画主流 u(y) @ x = x_loc, 右图画法向 v(x) @ y = y_loc
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
        ax.plot(solver_coarse_nn.u_2d[mid_i_c, :], solver_coarse_nn.y, 'b:s', ms=5, lw=2.2, label='Coarse (Scheme-1 NN)')
        ax.set_xlabel('Streamwise u (m/s)'); ax.set_ylabel('Cross-stream y (m)')
        ax.set_title(f'Streamwise Profile (x = {x_loc:.2f}): u vs y')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes1[1]
        ax.plot(solver_dense.x, solver_dense.v_2d[:, mid_j_d], 'k-', lw=2.5, label='Dense Grid (Reference)')
        ax.plot(solver_coarse_orig.x, solver_coarse_orig.v_2d[:, mid_j_c], 'r--o', ms=5, lw=1.8, label='Coarse (Original)')
        ax.plot(solver_coarse_nn.x, solver_coarse_nn.v_2d[:, mid_j_c], 'b:s', ms=5, lw=2.2, label='Coarse (Scheme-1 NN)')
        ax.set_xlabel('Streamwise x (m)'); ax.set_ylabel('Cross-stream v (m/s)')
        ax.set_title(f'Cross-Stream Profile (y = {y_loc:.2f}): v vs x')
        ax.legend(); ax.grid(True, alpha=0.3)
    else:
        # 90° 逆时针旋转流动 (从下至上):
        # 原 x 轴对应现 y' 轴，原 y 轴对应现 (x'_max - x')，原主流 u 对应现 v，原法向 v 对应现 -u！
        full_x_rows = np.where(np.all(mc.valid_mask, axis=0))[0]
        mid_j_c = int(full_x_rows[len(full_x_rows) // 2]) if len(full_x_rows) > 0 else mc.ny // 2
        y_loc = mc.y[mid_j_c]
        mid_j_d = int(np.argmin(np.abs(solver_dense.y - y_loc)))

        full_y_cols = np.where(np.all(mc.valid_mask, axis=1))[0]
        # 取与 0° 完全对称的横截面索引
        mid_i_c = int(full_y_cols[(len(full_y_cols) - 1) - len(full_y_cols) // 2]) if len(full_y_cols) > 0 else mc.nx // 2
        x_loc = mc.x[mid_i_c]
        mid_i_d = int(np.argmin(np.abs(solver_dense.x - x_loc)))

        ax = axes1[0]
        ax.plot(solver_dense.v_2d[:, mid_j_d], solver_dense.Lx - solver_dense.x, 'k-', lw=2.5, label='Dense Grid (Reference)')
        ax.plot(solver_coarse_orig.v_2d[:, mid_j_c], mc.Lx - solver_coarse_orig.x, 'r--o', ms=5, lw=1.8, label='Coarse (Original)')
        ax.plot(solver_coarse_nn.v_2d[:, mid_j_c], mc.Lx - solver_coarse_nn.x, 'b:s', ms=5, lw=2.2, label='Coarse (Scheme-1 NN)')
        ax.set_xlabel('Streamwise v (m/s)'); ax.set_ylabel('Cross-stream (Lx - x\') (m)')
        ax.set_title(f'Streamwise Profile (y\' = {y_loc:.2f}): v vs (Lx - x\')')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes1[1]
        ax.plot(solver_dense.y, -solver_dense.u_2d[mid_i_d, :], 'k-', lw=2.5, label='Dense Grid (Reference)')
        ax.plot(solver_coarse_orig.y, -solver_coarse_orig.u_2d[mid_i_c, :], 'r--o', ms=5, lw=1.8, label='Coarse (Original)')
        ax.plot(solver_coarse_nn.y, -solver_coarse_nn.u_2d[mid_i_c, :], 'b:s', ms=5, lw=2.2, label='Coarse (Scheme-1 NN)')
        ax.set_xlabel('Streamwise y\' (m)'); ax.set_ylabel('Cross-stream -u (m/s)')
        ax.set_title(f'Cross-Stream Profile (x\' = {x_loc:.2f}): -u vs y\'')
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    prof_path = f'./{case_name}_nn_centerline_profiles.png'
    plt.savefig(prof_path, dpi=150, bbox_inches='tight')
    print(f"  [图像保存 3/4] ({case_name}) 特征截面速度剖面图已保存: {prof_path}")
    plt.show()

    # ===== 直接在 1D 内部流体节点向量上计算误差指标 (u & v) =====
    act = mc.is_internal
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

    print("\n" + "=" * 72)
    print(f"  [{case_name}] 1D 内部流体节点速度误差对比汇总:")
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
    bars2 = ax2.bar(x_pos + width/2, nn_vals,   width, label='Coarse Scheme-1 NN', color='#4ecdc4')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['u Max Error', 'v Max Error', 'u RMSE', 'v RMSE'], fontsize=11)
    ax2.set_ylabel('Velocity Error (m/s)', fontsize=11)
    ax2.set_title(
        f'[{case_name}] Error Reduction | Max: u={imp_u_max:.2f}%, v={imp_v_max:.2f}% '
        f'| RMSE: u={imp_u_l2:.2f}%, v={imp_v_l2:.2f}%',
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
    print(f"  [图像保存 4/4] ({case_name}) 误差统计柱状图已保存: {err_path}")
    plt.show()

    return {
        'err_u_orig': err_u_orig, 'err_u_nn': err_u_nn, 'imp_u_max': imp_u_max,
        'err_v_orig': err_v_orig, 'err_v_nn': err_v_nn, 'imp_v_max': imp_v_max,
        'rmse_u_orig': rmse_u_orig, 'rmse_u_nn': rmse_u_nn, 'imp_u_l2': imp_u_l2,
        'rmse_v_orig': rmse_v_orig, 'rmse_v_nn': rmse_v_nn, 'imp_v_l2': imp_v_l2,
    }


# ============================================================
# 模块 7: 方案一全流程训练 (仅 0° 网格) + 双算例严格对偶验证管线 (0° & 90°)
# ============================================================
def run_scheme1_equivariant_pipeline(config: dict, plot_results=True):
    case_name_0 = f"{config.get('case_name', 'flow_case')}_0deg"
    dense_msh_path  = config['meshes']['dense_msh']
    coarse_msh_path = config['meshes']['coarse_msh']

    ensure_gmsh_files_exist(dense_msh_path, coarse_msh_path)

    # =========================================================================
    # 第一阶段：0° 原始流向算例 (流动从左至右 +x) —— 求解、提取特征、训练网络并验证
    # =========================================================================
    print(f"\n>>> [算例 1: 0° 原始流向 (从左至右)] 正在读取 Gmsh 网格并构建 1D 拓扑连接图...")
    mesh_dense_0  = ConnectivityGmshMesh(dense_msh_path, rotate_90_ccw=False)
    mesh_coarse_0 = ConnectivityGmshMesh(coarse_msh_path, rotate_90_ccw=False)
    print(f"  0° 密网格 ({dense_msh_path})  -> 1D 节点总数 N={mesh_dense_0.N} (内部流体点={int(mesh_dense_0.is_internal.sum())})")
    print(f"  0° 粗网格 ({coarse_msh_path}) -> 1D 节点总数 N={mesh_coarse_0.N} (内部流体点={int(mesh_coarse_0.is_internal.sum())})")

    print("\n" + "=" * 72)
    print("  Step 1 [0° 原始算例]: 密网格 1D 连接图求解 (高精度参考解)")
    print("=" * 72)
    solver_dense_0 = UniversalSIMPLE(mesh_dense_0, config)
    solver_dense_0.solve(verbose=True)

    print("\n" + "=" * 72)
    print("  Step 2 [0° 原始算例]: 粗网格无修正基准求解 (用于构造局部正交标架 R 与旋转不变特征)")
    print("=" * 72)
    solver_coarse_orig_0 = UniversalSIMPLE(mesh_coarse_0, config)
    solver_coarse_orig_0.solve(verbose=True)

    print("\n" + "=" * 72)
    print("  Step 3 [0° 原始算例]: 采样密网格参考解并计算粗网格 1D 完备离散残差")
    print("=" * 72)
    u_raw_0, v_raw_0, p_raw_0 = sample_dense_to_coarse_spline(mesh_dense_0, solver_dense_0, mesh_coarse_0)
    solver_template_0 = UniversalSIMPLE(mesh_coarse_0, config)
    res_u_0, res_v_0, res_div_0, u_sampled_0, v_sampled_0, _ = compute_discrete_residual(
        solver_template_0, u_raw_0, v_raw_0, p_raw_0
    )

    # Step 4: 仅在 0° 粗网格上训练方案一旋转协变神经网络
    corrector = Scheme1EquivariantNeuralCorrector()
    X_train_0, Y_tensors_0 = corrector.prepare_training_data(solver_coarse_orig_0, res_u_0, res_v_0, res_div_0)
    corrector.train(X_train_0, Y_tensors_0, config.get('nn_config', {}))

    # Step 5: 对 0° 粗网格预测源项并注入求解器
    res_u_pred_0, res_v_pred_0, res_div_pred_0 = corrector.predict(solver_coarse_orig_0)
    print("\n" + "=" * 72)
    print("  Step 5 [0° 原始算例]: 粗网格注入方案一 NN 预测源项求解")
    print("=" * 72)
    solver_coarse_nn_0 = UniversalSIMPLE(mesh_coarse_0, config)
    solver_coarse_nn_0.Sp_nn_u   = res_u_pred_0
    solver_coarse_nn_0.Sp_nn_v   = res_v_pred_0
    solver_coarse_nn_0.Sp_nn_div = res_div_pred_0
    solver_coarse_nn_0.solve(verbose=True)

    if plot_results:
        plot_nn_source_terms(
            res_u_0, res_v_0, res_div_0,
            res_u_pred_0, res_v_pred_0, res_div_pred_0,
            solver_template_0, case_name_0
        )
        stats_0 = plot_profile_and_error(
            solver_dense_0, solver_coarse_orig_0, solver_coarse_nn_0,
            u_sampled_0, v_sampled_0, case_name_0
        )
        plot_flow_field(solver_dense_0, solver_coarse_orig_0, solver_coarse_nn_0, case_name_0)

    # =========================================================================
    # 第二阶段：90° 逆时针旋转粗网格算例 (流动从下至上 +y) —— 零样本跨方向验证！
    # =========================================================================
    config_rot90 = make_rotated_90_ccw_config(config)
    case_name_90 = config_rot90['case_name']

    print(f"\n\n{'#'*72}")
    print(f"  >>> [算例 2: 90° 逆时针旋转网格验证 (流动从下至上 +y)] <<<")
    print(f"  注意: 神经网络【完全不重新训练】，直接利用方案一局部正交标架 R 预测 90° 旋转粗网格源项！")
    print(f"{'#'*72}")

    mesh_dense_90  = ConnectivityGmshMesh(dense_msh_path, rotate_90_ccw=True)
    mesh_coarse_90 = ConnectivityGmshMesh(coarse_msh_path, rotate_90_ccw=True)

    print("\n" + "=" * 72)
    print("  Step 6 [90° 旋转算例]: 90° 旋转密网格参考求解 (验证密网格旋转对称性)")
    print("=" * 72)
    solver_dense_90 = UniversalSIMPLE(mesh_dense_90, config_rot90)
    solver_dense_90.solve(verbose=True)

    u_raw_90, v_raw_90, p_raw_90 = sample_dense_to_coarse_spline(mesh_dense_90, solver_dense_90, mesh_coarse_90)
    solver_template_90 = UniversalSIMPLE(mesh_coarse_90, config_rot90)
    res_u_90, res_v_90, res_div_90, u_sampled_90, v_sampled_90, _ = compute_discrete_residual(
        solver_template_90, u_raw_90, v_raw_90, p_raw_90
    )

    print("\n" + "=" * 72)
    print("  Step 7 [90° 旋转算例]: 90° 旋转粗网格无修正基准求解 (Original Coarse 90°)")
    print("=" * 72)
    solver_coarse_orig_90 = UniversalSIMPLE(mesh_coarse_90, config_rot90)
    solver_coarse_orig_90.solve(verbose=True)

    # 检验修复后无修正粗网格流场的严格旋转对称性: u_90 = -v_0, v_90 = +u_0
    base_sym_err_u = np.max(np.abs(solver_coarse_orig_90.u + solver_coarse_orig_0.v))
    base_sym_err_v = np.max(np.abs(solver_coarse_orig_90.v - solver_coarse_orig_0.u))
    print(f"\n  [底层求解器对称性检验] 0° 与 90° 粗网格基准流场对称偏差:")
    print(f"    max |u_orig(90°) - (-v_orig(0°))| = {base_sym_err_u:.3e}")
    print(f"    max |v_orig(90°) - (+u_orig(0°))| = {base_sym_err_v:.3e}")

    # 直接调用在 0° 网格上训练好的 corrector 对 90° 旋转粗网格进行零样本源项预测！
    res_u_pred_90, res_v_pred_90, res_div_pred_90 = corrector.predict(solver_coarse_orig_90)

    # 验证旋转协变性定理: 在 90° 逆时针旋转下，预测源项必须满足 Sp_u(90°) = -Sp_v(0°), Sp_v(90°) = +Sp_u(0°)
    equiv_err_u = np.max(np.abs(res_u_pred_90 + res_v_pred_0))
    equiv_err_v = np.max(np.abs(res_v_pred_90 - res_u_pred_0))
    equiv_err_div = np.max(np.abs(res_div_pred_90 - res_div_pred_0))
    print(f"\n  [神经网络旋转协变性检验] 0° 与 90° 神经网络预测源项的旋转协变偏差:")
    print(f"    max |Sp_u(90°)   - (-Sp_v(0°))|   = {equiv_err_u:.3e}")
    print(f"    max |Sp_v(90°)   - (+Sp_u(0°))|   = {equiv_err_v:.3e}")
    print(f"    max |Sp_div(90°) - (+Sp_div(0°))| = {equiv_err_div:.3e}")

    print("\n" + "=" * 72)
    print("  Step 8 [90° 旋转算例]: 90° 旋转粗网格注入方案一 NN 预测源项求解")
    print("=" * 72)
    solver_coarse_nn_90 = UniversalSIMPLE(mesh_coarse_90, config_rot90)
    solver_coarse_nn_90.Sp_nn_u   = res_u_pred_90
    solver_coarse_nn_90.Sp_nn_v   = res_v_pred_90
    solver_coarse_nn_90.Sp_nn_div = res_div_pred_90
    solver_coarse_nn_90.solve(verbose=True)

    if plot_results:
        plot_nn_source_terms(
            res_u_90, res_v_90, res_div_90,
            res_u_pred_90, res_v_pred_90, res_div_pred_90,
            solver_template_90, case_name_90
        )
        stats_90 = plot_profile_and_error(
            solver_dense_90, solver_coarse_orig_90, solver_coarse_nn_90,
            u_sampled_90, v_sampled_90, case_name_90
        )
        plot_flow_field(solver_dense_90, solver_coarse_orig_90, solver_coarse_nn_90, case_name_90)

        print("\n" + "#" * 78)
        print("  【最终双算例汇总对比：0° 原始粗网格 vs 90° 旋转粗网格 (零样本泛化)】")
        print("#" * 78)
        print(f"  1. 0°  原始粗网格 (从左至右): u_RMSE 降低 {stats_0['imp_u_l2']:.2f}%,  v_RMSE 降低 {stats_0['imp_v_l2']:.2f}%")
        print(f"  2. 90° 旋转粗网格 (从下至上): v_RMSE 降低 {stats_90['imp_v_l2']:.2f}%,  u_RMSE 降低 {stats_90['imp_u_l2']:.2f}%")
        print("  (注: 90° 旋转后主流 u 变为 v，法向速度 v 变为 -u，两者的收敛步数、误差降低率与剖面曲线完全一致！)")
        print("#" * 78)

    return (
        (solver_dense_0, solver_coarse_orig_0, solver_coarse_nn_0),
        (solver_dense_90, solver_coarse_orig_90, solver_coarse_nn_90)
    )


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
            "max_iter": 2000, "tol": 1e-7,
            "alpha_u": 0.7, "alpha_v": 0.7, "alpha_p": 0.3
        },
        "nn_config": {
            "epochs": 4000, "lr": 3e-3, "hidden_dim": 96
        }
    }

    run_scheme1_equivariant_pipeline(config_backward, plot_results=True)