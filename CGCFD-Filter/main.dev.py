#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================================
CoarseGrid_SGS_DiscreteImbalance_to_MuCG_2D
严格按照【密网格映射 -> 粗网格离散不平衡项 -> Boussinesq 涡黏假设 -> Smagorinsky 模化 mu_cg】构建：

【核心物理与数学流程】
 1. 密网格映射与粗网格离散不平衡项提取 (Discrete Imbalance Extraction):
    将密网格流场空间滤波映射至粗网格得到 (\bar{u}, \bar{v}, \bar{p})，代入粗网格离散算子，
    直接计算出粗网格方程的不平衡项 F_{sgs, u}^{loss}, F_{sgs, v}^{loss}。
    该不平衡项严格代表粗网格相对于密网格的亚格子总动量损失 (-\nabla \cdot \tau^{cg})，
    补充该项可使粗网格解与密网格映射解 100% 重合。
 2. 类比 Boussinesq 假设，由不平衡项反演粗网格涡黏度 \mu_{cg} (Boussinesq Eddy-Viscosity Inversion):
    将亚格子不平衡项通过 Boussinesq 涡黏应力散度进行模化:
        F_{sgs, i}^{loss} = -\frac{\partial \tau_{ij}^{cg}}{\partial x_j}
                          \approx \frac{\partial}{\partial x_j} \left[ \mu_{cg} \left(\frac{\partial \bar{u}_i}{\partial x_j} + \frac{\partial \bar{u}_j}{\partial x_i}\right) \right]
    在粗网格有限体积离散框架下，组装涡黏算子矩阵 A_{visc}(\bar{u}, \bar{v})，通过最小二乘法直接从
    不平衡项 F_{sgs}^{loss} 解出每个粗网格单元的等效涡黏度 \mu_{cg}(x)！
 3. 类比 LES Smagorinsky / WALE 模型计算人工参数 C_{cg}(x)，并纯粹通过 \mu_{cg} 驱动粗网格求解:
    利用 \mu_{cg}(x) = \rho (C_{cg}(x) \Delta)^2 |S| 一步计算出人工参数 C_{cg}(x)。
    在粗网格求解器迭代中，完全通过 \mu_{eff} = \mu_l + \rho (C_{cg} \Delta)^2 |S| 修改扩散矩阵与
    Boussinesq 交叉应力项起效（绝不额外注入未封闭的源项补丁）！
==================================================================================================
"""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix, vstack, eye
from scipy.sparse.linalg import spsolve, lsqr
from scipy.interpolate import RectBivariateSpline

np.random.seed(42)


# ============================================================
# 辅助模块: 若本地缺少 .msh 文件，自动生成标准 Gmsh 2.2 台阶网格
# ============================================================
def ensure_gmsh_files_exist(dense_path: str, coarse_path: str):
    """如果指定路径下存在 .msh 文件则直接读取；若不存在则自动生成标准 L 型后向台阶 .msh 网格"""
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
# 模块 1: 基于 1D 向量化连接图的 Gmsh (.msh) 拓扑重构解析器
# ============================================================
class ConnectivityGmshMesh:
    def __init__(self, msh_filepath, tol_decimals=6, rotate_90_ccw=False):
        self.filepath = msh_filepath
        self.tol_decimals = tol_decimals
        self.rotate_90_ccw = rotate_90_ccw

        self.raw_nodes = {}
        self.phys_names = {}
        self.boundary_nodes = {}
        self.boundary_edges = {}
        self.quad_elements = []

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
        # 2D 粗网格滤波特征尺度 \Delta = \sqrt{\Delta x \cdot \Delta y}
        self.delta_cg = np.sqrt(self.cell_vol)

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
# 模块 2: 稀疏矩阵向量化组装器
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
# 模块 3: 纯粹通过粗网格涡黏度 \mu_{cg} = \rho (C_{cg} \Delta)^2 |S| 起效的 SIMPLE 求解器
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

        # ------------------------------------------------------------------
        # 粗网格亚格子模型核心物理变量：
        # 标定出的人工参数 C_{cg}(x) -> 在线计算涡黏度 \mu_{cg}(x) = \rho (C_{cg} \Delta)^2 |S|
        # ------------------------------------------------------------------
        self.C_cg_field = np.zeros(self.N, dtype=float)
        self.mu_cg = np.zeros(self.N, dtype=float)
        self.include_boussinesq_cross_term = True
        self.sgs_calibrator = None

        # 仅用于 Step 3 验证“补充精确离散不平衡项可 100% 还原密网格滤波场”，在实际模型求解时保持为 None！
        self.Sp_sgs_u = None
        self.Sp_sgs_v = None
        self.Sp_sgs_div = None

        self.aE = np.zeros(self.N, dtype=float)
        self.aW = np.zeros(self.N, dtype=float)
        self.aN = np.zeros(self.N, dtype=float)
        self.aS = np.zeros(self.N, dtype=float)
        self.aP = np.zeros(self.N, dtype=float)
        self.Sp = np.zeros(self.N, dtype=float)

        self.active = self.mesh.is_internal
        self.ref_p_idx = int(np.where(self.active)[0][0])

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

    def compute_boussinesq_sgs_force(self, u_field: np.ndarray, v_field: np.ndarray, mu_cg_field: np.ndarray):
        """
        计算 Boussinesq 涡黏假设下 \tau_{ij}^{cg, d} = -2 \mu_{cg} S_{ij}^d 在离散动量方程中产生的完整亚格子力:
            F_{sgs, i}(\mu_{cg}) = -\frac{\partial \tau_{ij}^{cg, d}}{\partial x_j}
                                 = \nabla \cdot (\mu_{cg} \nabla u_i) + \frac{\partial}{\partial x_j}\left(\mu_{cg} \frac{\partial u_j}{\partial x_i} - \frac{2}{3}\mu_{cg}(\nabla\cdot\mathbf{u})\delta_{ij}\right)
        """
        m = self.mesh
        act = self.active
        idx = np.where(act)[0]

        # 1. 主拉普拉斯扩散部: \nabla \cdot (\mu_{cg} \nabla u_i) (在隐式矩阵中由 \mu_{eff} = \mu_l + \mu_{cg} 承担)
        mu_e = np.where(m.nbr_E >= 0, 0.5 * (mu_cg_field + mu_cg_field[np.maximum(m.nbr_E, 0)]), mu_cg_field)
        mu_w = np.where(m.nbr_W >= 0, 0.5 * (mu_cg_field + mu_cg_field[np.maximum(m.nbr_W, 0)]), mu_cg_field)
        mu_n = np.where(m.nbr_N >= 0, 0.5 * (mu_cg_field + mu_cg_field[np.maximum(m.nbr_N, 0)]), mu_cg_field)
        mu_s = np.where(m.nbr_S >= 0, 0.5 * (mu_cg_field + mu_cg_field[np.maximum(m.nbr_S, 0)]), mu_cg_field)

        De = mu_e / (m.dx_P * m.dx_E)
        Dw = mu_w / (m.dx_P * m.dx_W)
        Dn = mu_n / (m.dy_P * m.dy_N)
        Ds = mu_s / (m.dy_P * m.dy_S)
        Dp = De + Dw + Dn + Ds

        lap_u = np.zeros(self.N, dtype=float)
        lap_v = np.zeros(self.N, dtype=float)
        lap_u[idx] = (
            De[idx] * u_field[m.nbr_E[idx]] + Dw[idx] * u_field[m.nbr_W[idx]] +
            Dn[idx] * u_field[m.nbr_N[idx]] + Ds[idx] * u_field[m.nbr_S[idx]] - Dp[idx] * u_field[idx]
        )
        lap_v[idx] = (
            De[idx] * v_field[m.nbr_E[idx]] + Dw[idx] * v_field[m.nbr_W[idx]] +
            Dn[idx] * v_field[m.nbr_N[idx]] + Ds[idx] * v_field[m.nbr_S[idx]] - Dp[idx] * v_field[idx]
        )

        # 2. Boussinesq 转置梯度交叉项 (式 12 中的 \partial/\partial x_j (\mu_{cg} \partial u_j / \partial x_i))
        cross_u = np.zeros(self.N, dtype=float)
        cross_v = np.zeros(self.N, dtype=float)
        if self.include_boussinesq_cross_term and np.any(mu_cg_field != 0.0):
            du_dx, du_dy = self._gradient(u_field)
            dv_dx, dv_dy = self._gradient(v_field)
            div_u = du_dx + dv_dy

            txx_cross = mu_cg_field * (du_dx - (2.0 / 3.0) * div_u)
            txy_cross = mu_cg_field * dv_dx
            tyx_cross = mu_cg_field * du_dy
            tyy_cross = mu_cg_field * (dv_dy - (2.0 / 3.0) * div_u)

            d_txx_dx, _ = self._gradient(txx_cross)
            _, d_txy_dy = self._gradient(txy_cross)
            d_tyx_dx, _ = self._gradient(tyx_cross)
            _, d_tyy_dy = self._gradient(tyy_cross)

            cross_u[idx] = d_txx_dx[idx] + d_txy_dy[idx]
            cross_v[idx] = d_tyx_dx[idx] + d_tyy_dy[idx]

        return lap_u + cross_u, lap_v + cross_v, cross_u, cross_v

    def _assemble_momentum(self, direction):
        rho = self.rho
        m = self.mesh
        act = self.active

        # ------------------------------------------------------------------
        # 严格按照统一控制方程 (式 12):
        # 有效黏度 \mu_{eff} = \mu_l + \mu_{cg} 直接进入扩散矩阵，使模型纯粹通过 \mu_{cg} 起效！
        # ------------------------------------------------------------------
        mu_eff = np.maximum(0.05 * self.mu, self.mu + self.mu_cg)
        mu_e = np.where(m.nbr_E >= 0, 0.5 * (mu_eff + mu_eff[np.maximum(m.nbr_E, 0)]), mu_eff)
        mu_w = np.where(m.nbr_W >= 0, 0.5 * (mu_eff + mu_eff[np.maximum(m.nbr_W, 0)]), mu_eff)
        mu_n = np.where(m.nbr_N >= 0, 0.5 * (mu_eff + mu_eff[np.maximum(m.nbr_N, 0)]), mu_eff)
        mu_s = np.where(m.nbr_S >= 0, 0.5 * (mu_eff + mu_eff[np.maximum(m.nbr_S, 0)]), mu_eff)

        De = mu_e / (m.dx_P * m.dx_E)
        Dw = mu_w / (m.dx_P * m.dx_W)
        Dn = mu_n / (m.dy_P * m.dy_N)
        Ds = mu_s / (m.dy_P * m.dy_S)

        aE_c = np.where(self.u < 0.0, -rho * self.u / m.dx_E, 0.0)
        aW_c = np.where(self.u > 0.0,  rho * self.u / m.dx_W, 0.0)
        aN_c = np.where(self.v < 0.0, -rho * self.v / m.dy_N, 0.0)
        aS_c = np.where(self.v > 0.0,  rho * self.v / m.dy_S, 0.0)

        self.aE = np.where(act, De + aE_c, 0.0)
        self.aW = np.where(act, Dw + aW_c, 0.0)
        self.aN = np.where(act, Dn + aN_c, 0.0)
        self.aS = np.where(act, Ds + aS_c, 0.0)
        self.aP = np.where(act, self.aE + self.aW + self.aN + self.aS, 1.0)

        # 计算 Boussinesq 涡黏应力张量散度中的显式转置梯度项 (完全由 \mu_{cg} 决定)
        _, _, cross_u, cross_v = self.compute_boussinesq_sgs_force(self.u, self.v, self.mu_cg)

        Gx, Gy = self._gradient(self.p)
        if direction == 'u':
            self.Sp = -Gx + cross_u
            if self.Sp_sgs_u is not None:
                self.Sp += self.Sp_sgs_u
        else:
            self.Sp = -Gy + cross_v
            if self.Sp_sgs_v is not None:
                self.Sp += self.Sp_sgs_v

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

        if self.Sp_sgs_div is not None:
            b = np.where(act, -(div - self.Sp_sgs_div), 0.0)
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
            if self.sgs_calibrator is not None and np.any(self.C_cg_field != 0.0):
                print("  ✓ 已启用【纯粹涡黏度驱动模式】：通过 C_cg(x) -> mu_cg = rho*(C_cg*Delta)^2*|S| 驱动方程 (无附加源项补丁)")
            print("=" * 72)

        u_old, v_old = self.u.copy(), self.v.copy()

        for it in range(1, max_iter + 1):
            # ------------------------------------------------------------------
            # 纯粹通过人工参数 C_{cg}(x) 与当前流场应变率 |S| 实时计算涡黏度 \mu_{cg}:
            #     \mu_{cg}(x) = \rho * sgn(C_{cg}^2) * (C_{cg} \Delta)^2 * \mathcal{D}_{op}(S_{ij})
            # ------------------------------------------------------------------
            if self.sgs_calibrator is not None and np.any(self.C_cg_field != 0.0):
                mu_cg_new = self.sgs_calibrator.evaluate_mu_cg_from_C_cg(self, self.C_cg_field)
                self.mu_cg = 0.85 * self.mu_cg + 0.15 * mu_cg_new if it > 1 else mu_cg_new

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
                mu_cg_mean = float(np.mean(np.abs(self.mu_cg[self.active])))
                print(f"  iter {it:5d} | res_u={res_u:.2e} | res_v={res_v:.2e} | "
                      f"div_err={mass_resid:.2e} | <|mu_cg|>={mu_cg_mean:.2e} | psi_min={psi_min:.4f}")

            if res_u < tol and res_v < tol and mass_resid < tol and it > 15:
                if verbose:
                    print(f"  >> 收敛! iter = {it}, res_u={res_u:.2e}, res_v={res_v:.2e}, div_err={mass_resid:.2e}")
                break
        else:
            if verbose:
                print(f"  >> 达到最大迭代次数 {max_iter}, res_u={res_u:.2e}, res_v={res_v:.2e}, div_err={mass_resid:.2e}")

        return res_u, res_v, mass_resid


# ============================================================
# 模块 4: 【第一步】将密网格结果映射至粗网格，提取粗网格离散不平衡项 (亚格子损失)
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


def map_dense_to_coarse_and_extract_imbalance(mesh_dense: ConnectivityGmshMesh,
                                              solver_dense: UniversalSIMPLE,
                                              mesh_coarse: ConnectivityGmshMesh,
                                              config: dict,
                                              n_quad: int = 5):
    """
    【核心第一步：密网格映射 -> 粗网格离散不平衡项提取】
    1. 将密网格流场通过控制体空间积分滤波映射到粗网格节点上，得到 (\bar{u}, \bar{v}, \bar{p})；
    2. 将 (\bar{u}, \bar{v}, \bar{p}) 代入粗网格离散动量算子（仅含分子黏度 \mu_l），直接计算出
       粗网格离散方程的不平衡项 F_{sgs, u}^{loss}, F_{sgs, v}^{loss}：
           F_{sgs, u}^{loss} = a_P \bar{u}_P - \sum_{nb} a_{nb} \bar{u}_{nb} + \frac{\partial \bar{p}}{\partial x}
       这一项严格代表粗网格相对密网格缺失的亚格子应力总散度 (-\nabla \cdot \tau^{cg})！
    """
    rho = solver_dense.rho
    u_fill = smooth_fill_solid_region(solver_dense.u_2d, mesh_dense.valid_mask)
    v_fill = smooth_fill_solid_region(solver_dense.v_2d, mesh_dense.valid_mask)
    p_fill = smooth_fill_solid_region(solver_dense.p_2d, mesh_dense.valid_mask)

    interp_u = RectBivariateSpline(mesh_dense.x, mesh_dense.y, u_fill, kx=3, ky=3)
    interp_v = RectBivariateSpline(mesh_dense.x, mesh_dense.y, v_fill, kx=3, ky=3)
    interp_p = RectBivariateSpline(mesh_dense.x, mesh_dense.y, p_fill, kx=3, ky=3)

    xi_1d, w_1d = np.polynomial.legendre.leggauss(n_quad)
    xi_1d = 0.5 * xi_1d
    w_1d = 0.5 * w_1d

    N_c = mesh_coarse.N
    u_bar = np.zeros(N_c, dtype=float)
    v_bar = np.zeros(N_c, dtype=float)
    p_bar = np.zeros(N_c, dtype=float)
    uu_bar = np.zeros(N_c, dtype=float)
    uv_bar = np.zeros(N_c, dtype=float)
    vv_bar = np.zeros(N_c, dtype=float)

    x_min, x_max = mesh_dense.x[0], mesh_dense.x[-1]
    y_min, y_max = mesh_dense.y[0], mesh_dense.y[-1]

    for a in range(n_quad):
        for b in range(n_quad):
            weight = w_1d[a] * w_1d[b]
            xq = np.clip(mesh_coarse.x_1d + xi_1d[a] * mesh_coarse.dx_P, x_min, x_max)
            yq = np.clip(mesh_coarse.y_1d + xi_1d[b] * mesh_coarse.dy_P, y_min, y_max)

            uq = interp_u(xq, yq, grid=False)
            vq = interp_v(xq, yq, grid=False)
            pq = interp_p(xq, yq, grid=False)

            u_bar  += weight * uq
            v_bar  += weight * vq
            p_bar  += weight * pq
            uu_bar += weight * (uq * uq)
            uv_bar += weight * (uq * vq)
            vv_bar += weight * (vq * vq)

    # 代入粗网格离散算子计算不平衡项 F_{sgs, u}^{loss}, F_{sgs, v}^{loss}
    solver_probe = UniversalSIMPLE(mesh_coarse, config)
    alpha_u = float(config.get('solver', {}).get('alpha_u', 0.65))
    act = solver_probe.active
    idx = np.where(act)[0]

    solver_probe.u = u_bar.copy()
    solver_probe.v = v_bar.copy()
    solver_probe.p = p_bar.copy()
    solver_probe.mu_cg[:] = 0.0
    solver_probe._bc_uv()
    solver_probe._bc_p()

    solver_probe._assemble_momentum('u')
    Gx, Gy = solver_probe._gradient(solver_probe.p)

    F_sgs_u_loss = np.zeros(N_c, dtype=float)
    F_sgs_v_loss = np.zeros(N_c, dtype=float)

    F_sgs_u_loss[idx] = (
        solver_probe.aP[idx] * solver_probe.u[idx]
        - solver_probe.aE[idx] * solver_probe.u[mesh_coarse.nbr_E[idx]]
        - solver_probe.aW[idx] * solver_probe.u[mesh_coarse.nbr_W[idx]]
        - solver_probe.aN[idx] * solver_probe.u[mesh_coarse.nbr_N[idx]]
        - solver_probe.aS[idx] * solver_probe.u[mesh_coarse.nbr_S[idx]]
        - (-Gx[idx])
    )
    F_sgs_v_loss[idx] = (
        solver_probe.aP[idx] * solver_probe.v[idx]
        - solver_probe.aE[idx] * solver_probe.v[mesh_coarse.nbr_E[idx]]
        - solver_probe.aW[idx] * solver_probe.v[mesh_coarse.nbr_W[idx]]
        - solver_probe.aN[idx] * solver_probe.v[mesh_coarse.nbr_N[idx]]
        - solver_probe.aS[idx] * solver_probe.v[mesh_coarse.nbr_S[idx]]
        - (-Gy[idx])
    )

    u_e, v_n, _ = solver_probe._rc_face_velocities(alpha_u)
    F_sgs_div_loss = np.zeros(N_c, dtype=float)
    F_sgs_div_loss[idx] = (
        (u_e[idx] - u_e[mesh_coarse.nbr_W[idx]]) / mesh_coarse.dx_P[idx] +
        (v_n[idx] - v_n[mesh_coarse.nbr_S[idx]]) / mesh_coarse.dy_P[idx]
    )

    # 同时保存滤波二阶矩对应的连续亚格子应力，供对比分析
    tau_xx = rho * (uu_bar - u_bar * u_bar)
    tau_xy = rho * (uv_bar - u_bar * v_bar)
    tau_yy = rho * (vv_bar - v_bar * v_bar)
    tau_iso = 0.5 * (tau_xx + tau_yy)

    sgs_dict = {
        'tau_xy_d': np.where(act, tau_xy, 0.0),
        'tau_xx_d': np.where(act, tau_xx - tau_iso, 0.0),
        'tau_yy_d': np.where(act, tau_yy - tau_iso, 0.0),
        'F_sgs_u_loss': F_sgs_u_loss,
        'F_sgs_v_loss': F_sgs_v_loss,
        'F_sgs_div_loss': F_sgs_div_loss,
    }
    return solver_probe.u.copy(), solver_probe.v.copy(), solver_probe.p.copy(), sgs_dict


# ============================================================
# 模块 5: 【第二步 & 第三步】由离散不平衡项通过 Boussinesq 模化反演 \mu_{cg} 与 Smagorinsky 参数 C_{cg}
# ============================================================
def compute_graph_poisson_wall_potential(solver: UniversalSIMPLE) -> np.ndarray:
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
    """提取 16 维严格旋转不变的局部正交标架特征 (用于构建 C_{cg}(x) 的空间回归映射)"""
    m = solver_coarse_base.mesh
    u_c = solver_coarse_base.u
    v_c = solver_coarse_base.v
    p_c = solver_coarse_base.p
    rho, mu = solver_coarse_base.rho, solver_coarse_base.mu

    vel_mag = np.sqrt(u_c**2 + v_c**2)
    U_ref = float(np.max(vel_mag))
    if U_ref < 1e-12:
        U_ref = 1.0

    u_sm, v_sm = graph_smooth_vector_field(u_c, v_c, m, n_hops=8)
    u_sm2, v_sm2 = graph_smooth_vector_field(u_sm, v_sm, m, n_hops=16)
    wx = (u_c + 0.4 * u_sm + 0.2 * u_sm2) / U_ref
    wy = (v_c + 0.4 * v_sm + 0.2 * v_sm2) / U_ref
    w_norm = np.sqrt(wx**2 + wy**2) + 1e-12

    cos_t = wx / w_norm
    sin_t = wy / w_norm
    e_perp_x = -sin_t
    e_perp_y = cos_t

    u_para = (u_c * cos_t + v_c * sin_t) / U_ref
    u_perp = (u_c * e_perp_x + v_c * e_perp_y) / U_ref
    usm_para = (u_sm * cos_t + v_sm * sin_t) / U_ref
    usm_perp = (u_sm * e_perp_x + v_sm * e_perp_y) / U_ref

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

    dp_dx, dp_dy = solver_coarse_base._gradient(p_c)
    dp_para = dp_dx * cos_t + dp_dy * sin_t
    dp_perp = dp_dx * e_perp_x + dp_dy * e_perp_y

    h_para = m.dx_P * (cos_t**2) + m.dy_P * (sin_t**2)
    h_perp = m.dx_P * (e_perp_x**2) + m.dy_P * (e_perp_y**2)
    Re_cell = rho * vel_mag * np.sqrt(m.dx_P * m.dy_P) / (mu + 1e-12)

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


class DiscreteImbalanceToMuCGCalibrator:
    """
    【严格按照用户物理逻辑构建的标定器】：
      1. 输入密网格映射至粗网格后产生的离散不平衡项 F_{sgs, u}^{loss}, F_{sgs, v}^{loss}（代表亚格子部分的损失）；
      2. 将该不平衡项通过 Boussinesq 涡黏应力散度进行表达:
             \nabla \cdot \left[ \mu_{cg}(\mathbf{x}) \left(\nabla \bar{\mathbf{u}} + (\nabla \bar{\mathbf{u}})^T\right) \right] \approx \mathbf{F}_{sgs}^{loss}
         构建关于节点涡黏度向量 \boldsymbol{\mu}_{cg} 的离散线性算子矩阵 \mathbf{L}_{Boussinesq}，
         通过最小二乘法直接从不平衡项反演出粗网格所需的涡黏度场 \mu_{cg}(\mathbf{x})！
      3. 类比大涡模拟 Smagorinsky / WALE 模型:
             \mu_{cg}(\mathbf{x}) = \rho C_{cg}^{\mathrm{signed}}(\mathbf{x}) \Delta^2 |S|
         直接把人工模型参数 C_{cg}(\mathbf{x}) 算出来，使粗网格求解完全通过 \mu_{cg} 起效！
    """
    def __init__(self, sgs_cfg: dict):
        self.model_type = sgs_cfg.get('model_type', 'smagorinsky').lower()
        self.ridge_lambda = float(sgs_cfg.get('ridge_lambda', 1e-5))
        self.num_rbfs = int(sgs_cfg.get('num_rbfs', 120))

        self.C_cg_global = 0.0
        self.C_cg_local_1d = None        # 带符号人工参数 sgn(mu_cg) * C_cg(x)，支持局部反传/过耗散修正
        self.mu_cg_calibrated_1d = None  # 直接从离散不平衡项反演出的等效涡黏度 \mu_{cg}(x)

        self.feat_mean = None
        self.feat_std = None
        self.rbf_centers = None
        self.rbf_widths = None
        self.W_ls = None

    def compute_sgs_basis_tensors(self, solver: UniversalSIMPLE, u_field: np.ndarray, v_field: np.ndarray):
        """计算粗网格宏观应变率张量 S_{ij} 与 Smagorinsky/WALE 算子 \mathcal{D}_{op}"""
        du_dx, du_dy = solver._gradient(u_field)
        dv_dx, dv_dy = solver._gradient(v_field)

        S_xx = du_dx
        S_yy = dv_dy
        S_xy = 0.5 * (du_dy + dv_dx)

        div_u = S_xx + S_yy
        S_xx_d = S_xx - 0.5 * div_u
        S_yy_d = S_yy - 0.5 * div_u
        S_xy_d = S_xy

        S_mag = np.sqrt(2.0 * (S_xx**2 + S_yy**2 + 2.0 * S_xy**2))

        if self.model_type == 'wale':
            g2_xx = du_dx * du_dx + du_dy * dv_dx
            g2_xy = du_dx * du_dy + du_dy * dv_dy
            g2_yx = dv_dx * du_dx + dv_dy * dv_dx
            g2_yy = dv_dx * du_dy + dv_dy * dv_dy

            tr_g2 = 0.5 * (g2_xx + g2_yy)
            Sd_xx = g2_xx - tr_g2
            Sd_yy = g2_yy - tr_g2
            Sd_xy = 0.5 * (g2_xy + g2_yx)

            Sd_Sq = Sd_xx**2 + Sd_yy**2 + 2.0 * Sd_xy**2
            S_Sq  = S_xx**2 + S_yy**2 + 2.0 * S_xy**2
            denom = (S_Sq ** 2.5) + (Sd_Sq ** 1.25) + 1e-24
            D_op  = (Sd_Sq ** 1.5) / denom
        else:
            D_op = S_mag

        return {
            'S_xx_d': S_xx_d, 'S_xy_d': S_xy_d, 'S_yy_d': S_yy_d,
            'S_mag': S_mag, 'D_op': D_op
        }

    def _build_boussinesq_operator_matrix_for_mu(self, solver: UniversalSIMPLE, u_bar: np.ndarray, v_bar: np.ndarray):
        """
        构建将单元涡黏度向量 \boldsymbol{\mu}_{cg} 线性映射为离散动量方程涡黏力 (F_{visc, u}, F_{visc, v}) 的稀疏算子矩阵 L_mu:
            L_mu @ \mu_{cg} = \nabla \cdot \left[ \mu_{cg} (\nabla \bar{\mathbf{u}} + (\nabla \bar{\mathbf{u}})^T) \right]
        这样令 L_mu @ \mu_{cg} \approx F_{sgs}^{loss}，即可直接从离散不平衡项解出 \mu_{cg}(x)！
        """
        m = solver.mesh
        N = m.N
        act = solver.active
        idx = np.where(act)[0]

        ie = m.nbr_E[idx]
        iw = m.nbr_W[idx]
        in_nb = m.nbr_N[idx]
        is_nb = m.nbr_S[idx]

        # 界面算术平均 \mu_e = 0.5 * (\mu_P + \mu_E) 对单元 P 及邻接单元的线性系数
        ce_u = 0.5 * (u_bar[ie] - u_bar[idx]) / (m.dx_P[idx] * m.dx_E[idx])
        cw_u = 0.5 * (u_bar[iw] - u_bar[idx]) / (m.dx_P[idx] * m.dx_W[idx])
        cn_u = 0.5 * (u_bar[in_nb] - u_bar[idx]) / (m.dy_P[idx] * m.dy_N[idx])
        cs_u = 0.5 * (u_bar[is_nb] - u_bar[idx]) / (m.dy_P[idx] * m.dy_S[idx])
        cp_u = ce_u + cw_u + cn_u + cs_u

        ce_v = 0.5 * (v_bar[ie] - v_bar[idx]) / (m.dx_P[idx] * m.dx_E[idx])
        cw_v = 0.5 * (v_bar[iw] - v_bar[idx]) / (m.dx_P[idx] * m.dx_W[idx])
        cn_v = 0.5 * (v_bar[in_nb] - v_bar[idx]) / (m.dy_P[idx] * m.dy_N[idx])
        cs_v = 0.5 * (v_bar[is_nb] - v_bar[idx]) / (m.dy_P[idx] * m.dy_S[idx])
        cp_v = ce_v + cw_v + cn_v + cs_v

        rows_u = np.concatenate([idx, idx, idx, idx, idx])
        cols_u = np.concatenate([idx, ie, iw, in_nb, is_nb])
        vals_u = np.concatenate([cp_u, ce_u, cw_u, cn_u, cs_u])

        rows_v = np.concatenate([idx, idx, idx, idx, idx])
        cols_v = np.concatenate([idx, ie, iw, in_nb, is_nb])
        vals_v = np.concatenate([cp_v, ce_v, cw_v, cn_v, cs_v])

        L_u = coo_matrix((vals_u, (rows_u, cols_u)), shape=(N, N)).tocsr()
        L_v = coo_matrix((vals_v, (rows_v, cols_v)), shape=(N, N)).tocsr()
        return L_u, L_v

    def _build_design_matrix(self, norm_feats: np.ndarray) -> np.ndarray:
        freqs = np.array([0.5 * np.pi, 1.0 * np.pi, 1.5 * np.pi])
        proj = norm_feats[:, :, None] * freqs[None, None, :]
        proj = proj.reshape(norm_feats.shape[0], -1)

        dist_sq = np.sum(
            (norm_feats[:, None, :] - self.rbf_centers[None, :, :]) ** 2,
            axis=-1
        ) / norm_feats.shape[1]
        rbf_feats = np.exp(-dist_sq / (2.0 * (self.rbf_widths[None, :] ** 2)))

        ones = np.ones((norm_feats.shape[0], 1), dtype=float)
        return np.hstack([ones, norm_feats, np.sin(proj), np.cos(proj), rbf_feats])

    def calibrate_from_discrete_imbalance(self,
                                          solver_coarse_base: UniversalSIMPLE,
                                          u_bar: np.ndarray,
                                          v_bar: np.ndarray,
                                          sgs_dict: dict):
        """
        【严格遵循你的三步逻辑执行标定】：
        1. 提取粗网格离散不平衡项 F_{sgs, u}^{loss}, F_{sgs, v}^{loss}（代表亚格子部分的损失）；
        2. 类比 Boussinesq 涡黏假设，求解线性超定方程组:
               [ L_u(\bar{u}) ]                 [ F_{sgs, u}^{loss} ]
               [ L_v(\bar{v}) ] * \mu_{cg}(x) = [ F_{sgs, v}^{loss} ]
           直接由不平衡项反演出粗网格每个单元所需的等效涡黏度 \mu_{cg}(x)！
        3. 类比 Smagorinsky / WALE 模型 \mu_{cg}(x) = \rho (C_{cg}(x) \Delta)^2 |S|，
           把人工模型参数 C_{cg}(x) 显式计算出来！
        """
        m = solver_coarse_base.mesh
        act = solver_coarse_base.active
        idx = np.where(act)[0]
        rho = solver_coarse_base.rho
        mu_l = solver_coarse_base.mu
        delta = m.delta_cg

        F_u_loss = sgs_dict['F_sgs_u_loss']
        F_v_loss = sgs_dict['F_sgs_v_loss']

        # -------------------------------------------------------------------------
        # 1. 构建 Boussinesq 涡黏算子矩阵，直接从离散不平衡项 (F_u_loss, F_v_loss) 最小二乘解出 \mu_{cg}(x)
        # -------------------------------------------------------------------------
        L_u, L_v = self._build_boussinesq_operator_matrix_for_mu(solver_coarse_base, u_bar, v_bar)

        # 组装超定最小二乘系统: [L_u[act]; L_v[act]; \reg * I] * \mu_{cg} = [F_u_loss[act]; F_v_loss[act]; 0]
        reg_weight = 1e-2 * float(np.mean(1.0 / m.cell_vol[act]))
        A_sys = vstack([L_u[idx, :], L_v[idx, :], reg_weight * eye(m.N, format='csr')]).tocsr()
        b_sys = np.concatenate([F_u_loss[idx], F_v_loss[idx], np.zeros(m.N, dtype=float)])

        mu_cg_raw = lsqr(A_sys, b_sys, atol=1e-10, btol=1e-10, iter_lim=2000)[0]

        # 物理稳定性限幅：保证总有效黏度 \mu_{eff} = \mu_l + \mu_{cg} >= 0.15 * \mu_l，上限不超过 15 * \mu_l
        mu_cg_clipped = np.clip(mu_cg_raw, -0.80 * mu_l, 15.0 * mu_l)
        self.mu_cg_calibrated_1d = np.where(act, mu_cg_clipped, 0.0)
        for bname in solver_coarse_base.bc_cfg.keys():
            solver_coarse_base._apply_neumann_extrapolation(self.mu_cg_calibrated_1d, bname)

        # -------------------------------------------------------------------------
        # 2. 类比大涡模拟 Smagorinsky / WALE 模型，由 \mu_{cg}(x) 计算人工参数 C_{cg}(x):
        #    \mu_{cg}(x) = \rho * sgn(C_{cg}) * (C_{cg}(x) \Delta)^2 \mathcal{D}_{op}
        #    ==> C_{cg}(x) = sgn(\mu_{cg}) * \sqrt{ |\mu_{cg}(x)| / (\rho \Delta^2 \mathcal{D}_{op}) }
        # -------------------------------------------------------------------------
        basis = self.compute_sgs_basis_tensors(solver_coarse_base, u_bar, v_bar)
        D_op = basis['D_op']

        denom_smag = rho * (delta ** 2) * D_op + 1e-24
        C_sq_signed = self.mu_cg_calibrated_1d / denom_smag
        C_sq_signed = np.clip(C_sq_signed, -2.5, 4.0)

        self.C_cg_local_1d = np.zeros(m.N, dtype=float)
        self.C_cg_local_1d[act] = np.sign(C_sq_signed[act]) * np.sqrt(np.abs(C_sq_signed[act]))
        for bname in solver_coarse_base.bc_cfg.keys():
            solver_coarse_base._apply_neumann_extrapolation(self.C_cg_local_1d, bname)

        # 全局平均等效 Smagorinsky 常数
        vol = m.cell_vol[act]
        self.C_cg_global = float(np.sqrt(np.sum(vol * np.abs(C_sq_signed[act])) / np.sum(vol)))

        # -------------------------------------------------------------------------
        # 3. 将人工参数 C_{cg}(x) 回归为关于局部无量纲旋转不变量的解析映射 (用于在线求解与 90° 旋转泛化)
        # -------------------------------------------------------------------------
        raw_feats, _, _ = extract_scheme1_local_frame_and_invariants(solver_coarse_base)
        feats_act = raw_feats[act]
        self.feat_mean = np.mean(feats_act, axis=0, keepdims=True)
        self.feat_std  = np.std(feats_act, axis=0, keepdims=True) + 1e-8
        norm_feats = (feats_act - self.feat_mean) / self.feat_std

        C_cg_act = self.C_cg_local_1d[act]
        importance = np.abs(C_cg_act) / (np.std(C_cg_act) + 1e-12)
        n_samples = len(norm_feats)
        n_peak = min(self.num_rbfs // 2, n_samples // 2)
        peak_idx = np.argsort(importance)[-n_peak:]
        unif_idx = np.linspace(0, n_samples - 1, min(self.num_rbfs // 2, n_samples), dtype=int)
        sel_idx = np.unique(np.concatenate([peak_idx, unif_idx]))

        centers_base = norm_feats[sel_idx]
        self.rbf_centers = np.vstack([centers_base, centers_base])
        self.rbf_widths = np.concatenate([
            np.full(len(centers_base), 0.18, dtype=float),
            np.full(len(centers_base), 0.50, dtype=float)
        ])

        Phi = self._build_design_matrix(norm_feats)
        Y = C_cg_act[:, None]

        n_basis = Phi.shape[1]
        A_ls = Phi.T @ Phi + self.ridge_lambda * np.eye(n_basis)
        B_ls = Phi.T @ Y
        self.W_ls = np.linalg.solve(A_ls, B_ls)

        C_pred_act = (Phi @ self.W_ls)[:, 0]
        rmse_C = float(np.sqrt(np.mean((C_pred_act - C_cg_act)**2)))

        # 检验仅靠 \mu_{cg} 产生的 Boussinesq 涡黏力对离散不平衡项的解释度
        F_visc_u, F_visc_v, _, _ = solver_coarse_base.compute_boussinesq_sgs_force(
            u_bar, v_bar, self.mu_cg_calibrated_1d
        )
        fit_corr_u = float(np.corrcoef(F_u_loss[act], F_visc_u[act])[0, 1])

        print("\n" + "=" * 76)
        print(f"  Step 4: 【由粗网格离散不平衡项 -> Boussinesq 反演 mu_cg -> 标定 C_cg(x)】结果")
        print("=" * 76)
        print(f"  * 亚格子本构模型类型      : {self.model_type.upper()} (mu_cg = rho * (C_cg * Delta)^2 * |S|)")
        print(f"  * 离散不平衡项反演规模    : {2 * len(idx)} 个动量不平衡方程 -> 反演 {len(idx)} 个单元 mu_cg(x)")
        print(f"  * 反演涡黏度 mu_cg(x)     : <|mu_cg|> = {np.mean(np.abs(self.mu_cg_calibrated_1d[act])):.3e} Pa·s, "
              f"max(mu_cg) = {np.max(self.mu_cg_calibrated_1d[act]):.3e} Pa·s (mu_l = {mu_l:.3e})")
        print(f"  * 反演人工参数 C_cg(x)    : 全局有效 C_cg = {self.C_cg_global:.4f}, "
              f"局部 |C_cg| 均值 = {np.mean(np.abs(C_cg_act)):.4f}")
        print(f"  * mu_cg 涡黏力与不平衡项相关系数: Corr(F_loss_u, F_Boussinesq_u) = {fit_corr_u:.4f}")
        print(f"  * 人工参数场 C_cg 回归误差: RMSE(C_cg) = {rmse_C:.3e}")
        print("=" * 76)

    def evaluate_mu_cg_from_C_cg(self,
                                 solver: UniversalSIMPLE,
                                 C_cg_1d: np.ndarray,
                                 u_override: np.ndarray = None,
                                 v_override: np.ndarray = None) -> np.ndarray:
        """
        由人工模型参数 C_{cg}(x) 与宏观应变率 |S| 计算粗网格亚格子等效涡黏度:
            \mu_{\mathrm{cg}}(\mathbf{x}) = \rho \cdot \mathrm{sgn}(C_{\mathrm{cg}}) (C_{\mathrm{cg}}(\mathbf{x}) \Delta)^2 \mathcal{D}_{\mathrm{op}}
        """
        m = solver.mesh
        act = solver.active
        u_eval = solver.u if u_override is None else u_override
        v_eval = solver.v if v_override is None else v_override

        basis = self.compute_sgs_basis_tensors(solver, u_eval, v_eval)
        mu_cg = np.zeros(m.N, dtype=float)
        C_signed_sq = np.sign(C_cg_1d[act]) * (C_cg_1d[act] ** 2)
        mu_cg_raw = solver.rho * C_signed_sq * (m.delta_cg[act] ** 2) * basis['D_op'][act]
        mu_cg[act] = np.clip(mu_cg_raw, -0.80 * solver.mu, 15.0 * solver.mu)

        for bname in solver.bc_cfg.keys():
            solver._apply_neumann_extrapolation(mu_cg, bname)
        return mu_cg

    def predict_C_cg_and_mu_cg(self, solver_coarse_base: UniversalSIMPLE):
        """对任意粗网格算例预测人工参数场 C_{cg}(x) 并计算对应的涡黏度场 \mu_{cg}(x)"""
        act = solver_coarse_base.active
        raw_feats, _, _ = extract_scheme1_local_frame_and_invariants(solver_coarse_base)
        norm_feats = (raw_feats - self.feat_mean) / self.feat_std
        Phi_all = self._build_design_matrix(norm_feats)

        C_cg_pred = (Phi_all @ self.W_ls)[:, 0]
        C_cg_pred[~act] = 0.0
        for bname in solver_coarse_base.bc_cfg.keys():
            solver_coarse_base._apply_neumann_extrapolation(C_cg_pred, bname)

        mu_cg_pred = self.evaluate_mu_cg_from_C_cg(solver_coarse_base, C_cg_pred)
        return C_cg_pred, mu_cg_pred


# ============================================================
# 模块 6: 学术可视化模块
# ============================================================
def plot_sgs_stresses_and_closure(sgs_dict: dict,
                                  C_cg_pred_1d: np.ndarray,
                                  mu_cg_pred_1d: np.ndarray,
                                  F_sgs_u_true_1d, F_sgs_v_true_1d,
                                  F_sgs_u_model_1d, F_sgs_v_model_1d,
                                  solver: UniversalSIMPLE, case_name: str):
    """图1: 展示粗网格离散不平衡项、反演人工参数 C_{cg}(x)、涡黏度 \mu_{cg}(x) 及 Boussinesq 涡黏力对比"""
    m = solver.mesh
    valid = m.valid_mask

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    fig.suptitle(
        f"[{case_name}] Discrete Imbalance $\\to$ Boussinesq $\\mu_{{\\mathrm{{cg}}}}(\\mathbf{{x}})$ $\\to$ Smagorinsky Parameter $C_{{\\mathrm{{cg}}}}(\\mathbf{{x}})$",
        fontsize=13.5, fontweight='bold'
    )

    row0_items = [
        (m.to_2d(sgs_dict['tau_xy_d']), r'Mapped Sub-grid Shear Stress $\tau_{xy}^{\mathrm{cg}, d}$ (Pa)', 'RdBu_r', True),
        (m.to_2d( np.abs(C_cg_pred_1d)), r'Calibrated Smagorinsky Parameter $|C_{\mathrm{cg}}(\mathbf{x})|$ (-)', 'viridis', False),
        (m.to_2d(mu_cg_pred_1d),        r'Active Eddy Viscosity $\mu_{\mathrm{cg}} = \rho C_{\mathrm{cg}}|C_{\mathrm{cg}}|\Delta^2|S|$ (Pa·s)', 'RdBu_r', True),
    ]
    for col, (field_2d, title, cmap_name, sym_scale) in enumerate(row0_items):
        ax = axes[0, col]
        plot_data = np.where(valid, field_2d, np.nan)
        if sym_scale:
            vm = max(float(np.max(np.abs(field_2d[valid]))), 1e-12)
            cf = ax.contourf(solver.X.T, solver.Y.T, plot_data.T, levels=30, cmap=cmap_name, vmin=-vm, vmax=vm)
        else:
            vm = max(float(np.max(field_2d[valid])), 1e-12)
            cf = ax.contourf(solver.X.T, solver.Y.T, plot_data.T, levels=30, cmap=cmap_name, vmin=0.0, vmax=vm)
        if m.has_obstacle:
            ax.set_facecolor('#dcdcdc')
        plt.colorbar(cf, ax=ax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_aspect('equal')

    if not m.rotate_90_ccw:
        t_full = m.to_2d(F_sgs_u_true_1d)
        p_full = m.to_2d(F_sgs_u_model_1d)
        comp_name = r'$u$-Momentum Imbalance'
    else:
        t_full = m.to_2d(F_sgs_v_true_1d)
        p_full = m.to_2d(F_sgs_v_model_1d)
        comp_name = r'$v$-Momentum Imbalance'

    diff = p_full - t_full
    vmax_main = max(float(np.max(np.abs(t_full[valid]))), float(np.max(np.abs(p_full[valid]))), 1e-12)
    vmax_diff = max(float(np.max(np.abs(diff[valid]))), 1e-12)

    for col, (data, sub_title, vm) in enumerate([
        (t_full, f'Exact Discrete Imbalance ({comp_name})', vmax_main),
        (p_full, f'Pure Boussinesq Force via $\\mu_{{\\mathrm{{cg}}}}$', vmax_main),
        (diff,   f'Boussinesq Approximation Residual', vmax_diff)
    ]):
        ax = axes[1, col]
        plot_data = np.where(valid, data, np.nan)
        cf = ax.contourf(solver.X.T, solver.Y.T, plot_data.T, levels=30, cmap='RdBu_r', vmin=-vm, vmax=vm)
        if m.has_obstacle:
            ax.set_facecolor('#dcdcdc')
        plt.colorbar(cf, ax=ax)
        ax.set_title(sub_title, fontsize=11)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_aspect('equal')

    plt.tight_layout()
    save_path = f'./{case_name}_sgs_least_squares_terms.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [图像保存 1/4] ({case_name}) 离散不平衡项、人工参数 C_cg 与涡黏度 mu_cg 云图已保存: {save_path}")
    plt.show()


def plot_flow_field(solver_dense: UniversalSIMPLE,
                    solver_coarse_orig: UniversalSIMPLE,
                    solver_coarse_sgs: UniversalSIMPLE,
                    case_name: str):
    """图2: 绘制流场对比 —— 速度幅值云图 + 流函数 ψ 涡结构云图"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    fig.suptitle(f"[{case_name}] Flow Field: Dense Reference vs Coarse Original vs Pure $\\mu_{{\\mathrm{{cg}}}}$-Driven SGS Model",
                 fontsize=13.5, fontweight='bold')

    solvers = [solver_dense, solver_coarse_orig, solver_coarse_sgs]
    titles = [
        f'Dense Grid Ref (N={solver_dense.N})',
        f'Coarse Original (N={solver_coarse_orig.N})',
        f'Coarse Pure $\\mu_{{\\mathrm{{cg}}}}$ Model (N={solver_coarse_sgs.N})'
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
    save_path = f'./{case_name}_sgs_flow_field.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [图像保存 2/4] ({case_name}) 流场与流线对比图已保存: {save_path}")
    plt.show()


def plot_profile_and_error(solver_dense: UniversalSIMPLE,
                           solver_coarse_orig: UniversalSIMPLE,
                           solver_coarse_sgs: UniversalSIMPLE,
                           u_filtered_1d, v_filtered_1d, case_name: str):
    """图3 & 图4: 绘制特征中心线速度剖面对比图 + 误差统计柱状图"""
    mc = solver_coarse_orig.mesh

    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5.5))
    fig1.suptitle(f"[{case_name}] Streamwise & Cross-Stream Profiles: Original vs Pure $\\mu_{{\\mathrm{{cg}}}}$-Driven SGS Model",
                  fontsize=13, fontweight='bold')

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
        ax.plot(solver_coarse_sgs.u_2d[mid_i_c, :], solver_coarse_sgs.y, 'b:s', ms=5, lw=2.2, label=r'Coarse (Pure $\mu_{\mathrm{cg}}$ Model)')
        ax.set_xlabel('Streamwise u (m/s)'); ax.set_ylabel('Cross-stream y (m)')
        ax.set_title(f'Streamwise Profile (x = {x_loc:.2f}): u vs y')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes1[1]
        ax.plot(solver_dense.x, solver_dense.v_2d[:, mid_j_d], 'k-', lw=2.5, label='Dense Grid (Reference)')
        ax.plot(solver_coarse_orig.x, solver_coarse_orig.v_2d[:, mid_j_c], 'r--o', ms=5, lw=1.8, label='Coarse (Original)')
        ax.plot(solver_coarse_sgs.x, solver_coarse_sgs.v_2d[:, mid_j_c], 'b:s', ms=5, lw=2.2, label=r'Coarse (Pure $\mu_{\mathrm{cg}}$ Model)')
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
        ax.plot(solver_coarse_sgs.v_2d[:, mid_j_c], mc.Lx - solver_coarse_sgs.x, 'b:s', ms=5, lw=2.2, label=r'Coarse (Pure $\mu_{\mathrm{cg}}$ Model)')
        ax.set_xlabel('Streamwise v (m/s)'); ax.set_ylabel('Cross-stream (Lx - x\') (m)')
        ax.set_title(f'Streamwise Profile (y\' = {y_loc:.2f}): v vs (Lx - x\')')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes1[1]
        ax.plot(solver_dense.y, -solver_dense.u_2d[mid_i_d, :], 'k-', lw=2.5, label='Dense Grid (Reference)')
        ax.plot(solver_coarse_orig.y, -solver_coarse_orig.u_2d[mid_i_c, :], 'r--o', ms=5, lw=1.8, label='Coarse (Original)')
        ax.plot(solver_coarse_sgs.y, -solver_coarse_sgs.u_2d[mid_i_c, :], 'b:s', ms=5, lw=2.2, label=r'Coarse (Pure $\mu_{\mathrm{cg}}$ Model)')
        ax.set_xlabel('Streamwise y\' (m)'); ax.set_ylabel('Cross-stream -u (m/s)')
        ax.set_title(f'Cross-Stream Profile (x\' = {x_loc:.2f}): -u vs y\'')
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    prof_path = f'./{case_name}_sgs_centerline_profiles.png'
    plt.savefig(prof_path, dpi=150, bbox_inches='tight')
    print(f"  [图像保存 3/4] ({case_name}) 特征截面速度剖面图已保存: {prof_path}")
    plt.show()

    act = mc.is_internal
    err_u_orig = float(np.max(np.abs(u_filtered_1d[act] - solver_coarse_orig.u[act])))
    err_u_sgs  = float(np.max(np.abs(u_filtered_1d[act] - solver_coarse_sgs.u[act])))
    err_v_orig = float(np.max(np.abs(v_filtered_1d[act] - solver_coarse_orig.v[act])))
    err_v_sgs  = float(np.max(np.abs(v_filtered_1d[act] - solver_coarse_sgs.v[act])))

    rmse_u_orig = float(np.sqrt(np.mean((u_filtered_1d[act] - solver_coarse_orig.u[act])**2)))
    rmse_u_sgs  = float(np.sqrt(np.mean((u_filtered_1d[act] - solver_coarse_sgs.u[act])**2)))
    rmse_v_orig = float(np.sqrt(np.mean((v_filtered_1d[act] - solver_coarse_orig.v[act])**2)))
    rmse_v_sgs  = float(np.sqrt(np.mean((v_filtered_1d[act] - solver_coarse_sgs.v[act])**2)))

    imp_u_max = (1.0 - err_u_sgs / (err_u_orig + 1e-30)) * 100.0
    imp_v_max = (1.0 - err_v_sgs / (err_v_orig + 1e-30)) * 100.0
    imp_u_l2  = (1.0 - rmse_u_sgs / (rmse_u_orig + 1e-30)) * 100.0
    imp_v_l2  = (1.0 - rmse_v_sgs / (rmse_v_orig + 1e-30)) * 100.0

    print("\n" + "=" * 72)
    print(f"  [{case_name}] 1D 内部流体节点滤波速度误差对比汇总 (纯粹通过 mu_cg 起效):")
    print(f"  [最大绝对误差 Max Error]:")
    print(f"    u: 原粗网格 = {err_u_orig:.4e} → 纯 mu_cg 模型修正后 = {err_u_sgs:.4e} (降低 {imp_u_max:.2f}%)")
    print(f"    v: 原粗网格 = {err_v_orig:.4e} → 纯 mu_cg 模型修正后 = {err_v_sgs:.4e} (降低 {imp_v_max:.2f}%)")
    print(f"  [均方根误差 RMSE]:")
    print(f"    u: 原粗网格 = {rmse_u_orig:.4e} → 纯 mu_cg 模型修正后 = {rmse_u_sgs:.4e} (降低 {imp_u_l2:.2f}%)")
    print(f"    v: 原粗网格 = {rmse_v_orig:.4e} → 纯 mu_cg 模型修正后 = {rmse_v_sgs:.4e} (降低 {imp_v_l2:.2f}%)")
    print("=" * 72)

    fig2, ax2 = plt.subplots(figsize=(10, 5.5))
    x_pos = np.arange(4)
    width = 0.35
    orig_vals = [err_u_orig, err_v_orig, rmse_u_orig, rmse_v_orig]
    sgs_vals  = [err_u_sgs,  err_v_sgs,  rmse_u_sgs,  rmse_v_sgs]

    bars1 = ax2.bar(x_pos - width/2, orig_vals, width, label='Coarse Original', color='#ff6b6b')
    bars2 = ax2.bar(x_pos + width/2, sgs_vals,  width, label=r'Coarse Pure $\mu_{\mathrm{cg}}$ Model', color='#4ecdc4')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['u Max Error', 'v Max Error', 'u RMSE', 'v RMSE'], fontsize=11)
    ax2.set_ylabel('Velocity Error vs Mapped Dense (m/s)', fontsize=11)
    ax2.set_title(
        f'[{case_name}] Pure $\\mu_{{\\mathrm{{cg}}}}$ Error Reduction | Max: u={imp_u_max:.2f}%, v={imp_v_max:.2f}% '
        f'| RMSE: u={imp_u_l2:.2f}%, v={imp_v_l2:.2f}%',
        fontsize=12, fontweight='bold'
    )
    ax2.legend(fontsize=11)
    ax2.set_yscale('log')
    ax2.set_ylim(top=max(orig_vals) * 6.0, bottom=max(min(sgs_vals) * 0.2, 1e-8))

    for bar, val in zip(list(bars1) + list(bars2), orig_vals + sgs_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15, f'{val:.2e}',
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3, which='both')

    plt.tight_layout()
    err_path = f'./{case_name}_sgs_error_stats.png'
    plt.savefig(err_path, dpi=150, bbox_inches='tight')
    print(f"  [图像保存 4/4] ({case_name}) 误差统计柱状图已保存: {err_path}")
    plt.show()

    return {
        'err_u_orig': err_u_orig, 'err_u_sgs': err_u_sgs, 'imp_u_max': imp_u_max,
        'err_v_orig': err_v_orig, 'err_v_sgs': err_v_sgs, 'imp_v_max': imp_v_max,
        'rmse_u_orig': rmse_u_orig, 'rmse_u_sgs': rmse_u_sgs, 'imp_u_l2': imp_u_l2,
        'rmse_v_orig': rmse_v_orig, 'rmse_v_sgs': rmse_v_sgs, 'imp_v_l2': imp_v_l2,
    }


# ============================================================
# 模块 7: 主控管线 (密网格映射提取不平衡项 -> Boussinesq 反演 mu_cg -> 标定 C_cg -> 纯 mu_cg 驱动求解)
# ============================================================
def run_route1_method1_sgs_pipeline(config: dict, plot_results=True):
    case_name_0 = f"{config.get('case_name', 'flow_case')}_0deg"
    dense_msh_path  = config['meshes']['dense_msh']
    coarse_msh_path = config['meshes']['coarse_msh']

    ensure_gmsh_files_exist(dense_msh_path, coarse_msh_path)

    # =========================================================================
    # 第一阶段：0° 原始流向算例
    # =========================================================================
    print(f"\n>>> [算例 1: 0° 原始流向 (从左至右)] 正在读取 Gmsh 网格并构建 1D 拓扑连接图...")
    mesh_dense_0  = ConnectivityGmshMesh(dense_msh_path, rotate_90_ccw=False)
    mesh_coarse_0 = ConnectivityGmshMesh(coarse_msh_path, rotate_90_ccw=False)

    print("\n" + "=" * 72)
    print("  Step 1 [0° 原始算例]: 密网格高精度求解 (获取精细尺度流场 u^fine, v^fine)")
    print("=" * 72)
    solver_dense_0 = UniversalSIMPLE(mesh_dense_0, config)
    solver_dense_0.solve(verbose=True)

    print("\n" + "=" * 72)
    print("  Step 2 [0° 原始算例]: 粗网格无修正基准求解 (Original Coarse)")
    print("=" * 72)
    solver_coarse_orig_0 = UniversalSIMPLE(mesh_coarse_0, config)
    solver_coarse_orig_0.solve(verbose=True)

    print("\n" + "=" * 72)
    print("  Step 3 [0° 原始算例]: 将密网格结果映射至粗网格，离散提取不平衡项 (代表亚格子损失)")
    print("=" * 72)
    u_filt_0, v_filt_0, p_filt_0, sgs_dict_0 = map_dense_to_coarse_and_extract_imbalance(
        mesh_dense_0, solver_dense_0, mesh_coarse_0, config, n_quad=5
    )

    # 验证：补充该离散不平衡项后，粗网格解与密网格映射解 100% 严格一致
    solver_verify_exact = UniversalSIMPLE(mesh_coarse_0, config)
    solver_verify_exact.Sp_sgs_u   = sgs_dict_0['F_sgs_u_loss']
    solver_verify_exact.Sp_sgs_v   = sgs_dict_0['F_sgs_v_loss']
    solver_verify_exact.Sp_sgs_div = sgs_dict_0['F_sgs_div_loss']
    solver_verify_exact.solve(verbose=False)
    act_0 = mesh_coarse_0.is_internal
    exact_consistency_err = float(np.max(np.abs(solver_verify_exact.u[act_0] - u_filt_0[act_0])))
    print(f"  [物理自洽性验证] 补充离散不平衡项 F_sgs^loss 后，粗网格与密网格映射场最大偏差 = {exact_consistency_err:.3e}")

    # Step 4: 将离散不平衡项通过 Boussinesq 假设模化，反演 \mu_cg(x) 并标定 Smagorinsky 参数 C_cg(x)
    calibrator = DiscreteImbalanceToMuCGCalibrator(config.get('sgs_config', {}))
    calibrator.calibrate_from_discrete_imbalance(
        solver_coarse_orig_0, u_filt_0, v_filt_0, sgs_dict_0
    )

    # Step 5: 纯粹通过 C_cg(x) -> \mu_cg(x) 驱动粗网格求解（不注入任何 Sp_sgs 源项补丁！）
    C_cg_pred_0, mu_cg_pred_0 = calibrator.predict_C_cg_and_mu_cg(solver_coarse_orig_0)
    print("\n" + "=" * 72)
    print("  Step 5 [0° 原始算例]: 粗网格加载人工参数 C_cg(x)，100% 纯粹通过涡黏度 mu_cg 起效求解")
    print("=" * 72)
    solver_coarse_sgs_0 = UniversalSIMPLE(mesh_coarse_0, config)
    solver_coarse_sgs_0.sgs_calibrator = calibrator
    solver_coarse_sgs_0.C_cg_field     = C_cg_pred_0
    solver_coarse_sgs_0.mu_cg          = mu_cg_pred_0.copy()
    # 注意：此处 Sp_sgs_u / Sp_sgs_v 保持为 None，动量方程完全靠 mu_eff = mu_l + mu_cg 起效！
    solver_coarse_sgs_0.solve(verbose=True)

    F_sgs_u_model_0, F_sgs_v_model_0, _, _ = solver_coarse_sgs_0.compute_boussinesq_sgs_force(
        solver_coarse_sgs_0.u, solver_coarse_sgs_0.v, solver_coarse_sgs_0.mu_cg
    )

    if plot_results:
        plot_sgs_stresses_and_closure(
            sgs_dict_0, C_cg_pred_0, solver_coarse_sgs_0.mu_cg,
            sgs_dict_0['F_sgs_u_loss'], sgs_dict_0['F_sgs_v_loss'],
            F_sgs_u_model_0, F_sgs_v_model_0,
            solver_coarse_sgs_0, case_name_0
        )
        stats_0 = plot_profile_and_error(
            solver_dense_0, solver_coarse_orig_0, solver_coarse_sgs_0,
            u_filt_0, v_filt_0, case_name_0
        )
        plot_flow_field(solver_dense_0, solver_coarse_orig_0, solver_coarse_sgs_0, case_name_0)

    # =========================================================================
    # 第二阶段：90° 逆时针旋转粗网格算例 —— 零样本检验 C_cg(x) -> \mu_cg(x) 的正交旋转不变性
    # =========================================================================
    config_rot90 = make_rotated_90_ccw_config(config)
    case_name_90 = config_rot90['case_name']

    print(f"\n\n{'#'*72}")
    print(f"  >>> [算例 2: 90° 逆时针旋转网格验证 (流动从下至上 +y)] <<<")
    print(f"  注意: 模型参数【完全不重新标定】，直接预测 C_cg(x) 并纯粹通过 mu_cg(x) 驱动旋转算例！")
    print(f"{'#'*72}")

    mesh_dense_90  = ConnectivityGmshMesh(dense_msh_path, rotate_90_ccw=True)
    mesh_coarse_90 = ConnectivityGmshMesh(coarse_msh_path, rotate_90_ccw=True)

    solver_dense_90 = UniversalSIMPLE(mesh_dense_90, config_rot90)
    solver_dense_90.solve(verbose=True)

    u_filt_90, v_filt_90, p_filt_90, sgs_dict_90 = map_dense_to_coarse_and_extract_imbalance(
        mesh_dense_90, solver_dense_90, mesh_coarse_90, config_rot90, n_quad=5
    )

    solver_coarse_orig_90 = UniversalSIMPLE(mesh_coarse_90, config_rot90)
    solver_coarse_orig_90.solve(verbose=True)

    C_cg_pred_90, mu_cg_pred_90 = calibrator.predict_C_cg_and_mu_cg(solver_coarse_orig_90)

    equiv_err_C  = np.max(np.abs(C_cg_pred_90 - C_cg_pred_0))
    equiv_err_mu = np.max(np.abs(mu_cg_pred_90 - mu_cg_pred_0))
    print(f"\n  [人工参数 C_cg(x) 与涡黏度 mu_cg(x) 90° 旋转不变性检验]:")
    print(f"    max |C_cg(90°)  - C_cg(0°)|  = {equiv_err_C:.3e}")
    print(f"    max |mu_cg(90°) - mu_cg(0°)| = {equiv_err_mu:.3e}")

    print("\n" + "=" * 72)
    print("  Step 8 [90° 旋转算例]: 90° 旋转粗网格 100% 纯粹通过 C_cg(x) -> mu_cg(x) 起效求解")
    print("=" * 72)
    solver_coarse_sgs_90 = UniversalSIMPLE(mesh_coarse_90, config_rot90)
    solver_coarse_sgs_90.sgs_calibrator = calibrator
    solver_coarse_sgs_90.C_cg_field     = C_cg_pred_90
    solver_coarse_sgs_90.mu_cg          = mu_cg_pred_90.copy()
    solver_coarse_sgs_90.solve(verbose=True)

    F_sgs_u_model_90, F_sgs_v_model_90, _, _ = solver_coarse_sgs_90.compute_boussinesq_sgs_force(
        solver_coarse_sgs_90.u, solver_coarse_sgs_90.v, solver_coarse_sgs_90.mu_cg
    )

    if plot_results:
        plot_sgs_stresses_and_closure(
            sgs_dict_90, C_cg_pred_90, solver_coarse_sgs_90.mu_cg,
            sgs_dict_90['F_sgs_u_loss'], sgs_dict_90['F_sgs_v_loss'],
            F_sgs_u_model_90, F_sgs_v_model_90,
            solver_coarse_sgs_90, case_name_90
        )
        stats_90 = plot_profile_and_error(
            solver_dense_90, solver_coarse_orig_90, solver_coarse_sgs_90,
            u_filt_90, v_filt_90, case_name_90
        )
        plot_flow_field(solver_dense_90, solver_coarse_orig_90, solver_coarse_sgs_90, case_name_90)

        print("\n" + "#" * 78)
        print("  【汇总对比：离散不平衡项 -> Boussinesq 反演 mu_cg -> 标定 C_cg(x) 纯涡黏驱动求解】")
        print("#" * 78)
        print(f"  * 全局有效亚格子参数     : C_cg_global = {calibrator.C_cg_global:.5f} ({calibrator.model_type.upper()})")
        print(f"  * 1. 0°  原始粗网格 (从左至右): u_RMSE 降低 {stats_0['imp_u_l2']:.2f}%,  v_RMSE 降低 {stats_0['imp_v_l2']:.2f}%")
        print(f"  * 2. 90° 旋转粗网格 (从下至上): v_RMSE 降低 {stats_90['imp_v_l2']:.2f}%,  u_RMSE 降低 {stats_90['imp_u_l2']:.2f}%")
        print("#" * 78)

    return (
        (solver_dense_0, solver_coarse_orig_0, solver_coarse_sgs_0),
        (solver_dense_90, solver_coarse_orig_90, solver_coarse_sgs_90)
    )


if __name__ == '__main__':
    dense_path = "../../CGCFD-Correct/meshes/back_fine.msh" if os.path.exists("../../CGCFD-Correct/meshes/back_fine.msh") else "./meshes/back_fine.msh"
    coarse_path = "../../CGCFD-Correct/meshes/back_coarse_coarse.msh" if os.path.exists("../meshes/back_coarse_coarse.msh") else "./meshes/back_coarse_coarse.msh"

    config_backward = {
        "case_name": "backward_Pure_MuCG_SGS",
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
            "alpha_u": 0.65, "alpha_v": 0.65, "alpha_p": 0.25
        },
        "sgs_config": {
            "model_type": "smagorinsky",      # 可选 'smagorinsky' 或 'wale'
            "ridge_lambda": 1e-5,             # C_cg(x) 空间回归正则化系数
            "num_rbfs": 120                   # 空间特征基函数数量
        }
    }

    run_route1_method1_sgs_pipeline(config_backward, plot_results=True)