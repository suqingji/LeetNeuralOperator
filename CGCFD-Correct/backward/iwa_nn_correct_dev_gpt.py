#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSIMPLE-Unstructured: cell-centred 2-D SIMPLE + Fourier/RBF residual correction.

Mesh format
-----------
- ASCII Gmsh MSH 2.x (normally 2.2).
- Triangles/quads, selected quadratic triangles/quads (corner nodes only), and Gmsh type-34 polygons.
- Boundary line elements must carry physical tags whose names match boundary_conditions.
- Unknowns live at cell centroids. The solver stores no bounding-box ghost/solid cells.

Numerics
--------
- Cell-face connectivity (owner/neighbour) drives all fluxes and matrix assembly.
- First-order upwind convection, implicit orthogonal diffusion with deferred non-orthogonal
  correction, least-squares cell gradients, Rhie-Chow face mass flux, SIMPLE pressure correction.
- The NN remains a per-case/per-coarse-mesh residual surrogate, not a geometry-general model.

This is a research/educational implementation. Validate conservation, mesh sensitivity,
non-orthogonal convergence and boundary conditions for each production case before relying on it.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
import torch
import torch.nn as nn


torch.manual_seed(42)
np.random.seed(42)


class UnstructuredGmshMesh:
    """Read an ASCII Gmsh MSH 2.x mesh and build a cell-face connectivity graph."""

    # Gmsh type -> number of corner nodes used for the 2-D cell geometry.
    CELL_CORNER_COUNT = {
        2: 3,    # triangle
        3: 4,    # quadrangle
        9: 3,    # 2nd-order triangle
        10: 4,   # 2nd-order quadrangle
        16: 4,   # 2nd-order quadrangle (8 nodes)
        20: 3,   # higher-order triangle
        21: 3,
        22: 3,
        23: 3,
        24: 3,
        25: 3,
        34: -1,   # arbitrary polygon (Gmsh polygon element)
    }
    LINE_TYPES = {1: 2, 8: 2, 26: 2, 27: 2, 28: 2}  # endpoint node count used for edge key

    def __init__(self, filepath: str, geometry_tol: float = 1e-12):
        self.filepath = filepath
        self.geometry_tol = float(geometry_tol)
        self.raw_nodes: Dict[int, Tuple[float, float, float]] = {}
        self.physical_names: Dict[Tuple[int, int], str] = {}
        self._raw_boundary_edges: List[Tuple[Tuple[int, int], str]] = []
        self._raw_cells: List[Tuple[int, ...]] = []
        self._parse_msh2()
        self._build_connectivity()

    def _parse_msh2(self) -> None:
        if not os.path.isfile(self.filepath):
            raise FileNotFoundError(f"找不到 Gmsh 网格文件: {self.filepath}")
        with open(self.filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Require MSH 2.x ASCII so the element parser cannot silently misread MSH 4.x blocks.
        version = None
        for i, line in enumerate(lines):
            if line == "$MeshFormat":
                if i + 1 < len(lines):
                    parts = lines[i + 1].split()
                    version = parts[0] if parts else None
                    if len(parts) >= 2 and parts[1] != "0":
                        raise ValueError("当前实现只支持 ASCII Gmsh 文件（MeshFormat 的 file-type 应为 0）。")
                break
        if version is not None and not version.startswith("2."):
            raise ValueError(f"当前实现需要 Gmsh MSH 2.x，检测到版本 {version}。请从 Gmsh 导出 MSH 2.2 ASCII。")

        # Physical names
        i = 0
        while i < len(lines):
            if lines[i] == "$PhysicalNames":
                n = int(lines[i + 1])
                for j in range(n):
                    parts = lines[i + 2 + j].split(maxsplit=2)
                    dim, tag = int(parts[0]), int(parts[1])
                    name = parts[2].strip().strip('"') if len(parts) > 2 else f"tag_{tag}"
                    self.physical_names[(dim, tag)] = name
                break
            i += 1

        # Nodes
        try:
            ni = lines.index("$Nodes")
        except ValueError as exc:
            raise ValueError("Gmsh 文件中未找到 $Nodes。") from exc
        num_nodes = int(lines[ni + 1])
        for line in lines[ni + 2:ni + 2 + num_nodes]:
            p = line.split()
            nid = int(p[0])
            self.raw_nodes[nid] = (float(p[1]), float(p[2]), float(p[3]))

        # Elements
        try:
            ei = lines.index("$Elements")
        except ValueError as exc:
            raise ValueError("Gmsh 文件中未找到 $Elements。") from exc
        num_elements = int(lines[ei + 1])
        for line in lines[ei + 2:ei + 2 + num_elements]:
            p = [int(v) for v in line.split()]
            if len(p) < 3:
                continue
            elem_type, n_tags = p[1], p[2]
            tags = p[3:3 + n_tags]
            conn = p[3 + n_tags:]
            physical_tag = tags[0] if tags else 0

            if elem_type in self.LINE_TYPES:
                if len(conn) < 2:
                    continue
                name = self.physical_names.get((1, physical_tag), f"tag_{physical_tag}")
                self._raw_boundary_edges.append(((conn[0], conn[1]), name))
            elif elem_type in self.CELL_CORNER_COUNT:
                ncorner = self.CELL_CORNER_COUNT[elem_type]
                if elem_type == 34:
                    # Gmsh polygon records contain the ordered boundary-node list.
                    if len(conn) < 3:
                        raise ValueError(f"多边形单元至少需要 3 个顶点：{line}")
                    self._raw_cells.append(tuple(conn))
                else:
                    if len(conn) < ncorner:
                        raise ValueError(f"单元类型 {elem_type} 的节点数不足：{line}")
                    self._raw_cells.append(tuple(conn[:ncorner]))

        if not self.raw_nodes:
            raise ValueError("网格中没有读到节点。")
        if not self._raw_cells:
            raise ValueError("没有找到支持的二维单元（线性/部分高阶三角形或四边形）。")
        if not self._raw_boundary_edges:
            raise ValueError("没有找到 Gmsh 物理边界线单元；请为入口、出口、壁面创建 Physical Curve。")

    @staticmethod
    def _polygon_geometry(xy: np.ndarray) -> Tuple[np.ndarray, float]:
        x, y = xy[:, 0], xy[:, 1]
        xn, yn = np.roll(x, -1), np.roll(y, -1)
        cross = x * yn - xn * y
        signed_area = 0.5 * np.sum(cross)
        if abs(signed_area) < 1e-30:
            raise ValueError("发现面积接近零的二维单元。")
        cx = np.sum((x + xn) * cross) / (6.0 * signed_area)
        cy = np.sum((y + yn) * cross) / (6.0 * signed_area)
        return np.array([cx, cy], dtype=float), float(signed_area)

    def _build_connectivity(self) -> None:
        node_xy = {nid: np.asarray(xyz[:2], dtype=float) for nid, xyz in self.raw_nodes.items()}
        boundary_names: Dict[Tuple[int, int], str] = {}
        for edge, name in self._raw_boundary_edges:
            key = tuple(sorted(edge))
            if key in boundary_names and boundary_names[key] != name:
                raise ValueError(f"同一边界边被赋予多个物理名称：{boundary_names[key]} / {name}")
            boundary_names[key] = name

        self.cell_nodes: List[Tuple[int, ...]] = []
        self.polygons: List[np.ndarray] = []
        centers, areas = [], []
        for cell_nodes_in in self._raw_cells:
            ids = list(cell_nodes_in)
            xy = np.asarray([node_xy[nid] for nid in ids], dtype=float)
            center, signed_area = self._polygon_geometry(xy)
            if signed_area < 0.0:
                ids.reverse()
                xy = xy[::-1].copy()
                center, signed_area = self._polygon_geometry(xy)
            self.cell_nodes.append(tuple(ids))
            self.polygons.append(xy)
            centers.append(center)
            areas.append(signed_area)

        self.centers = np.asarray(centers, dtype=float)
        self.volumes = np.asarray(areas, dtype=float)  # 2-D cell area (unit depth)
        self.n_cells = len(self.cell_nodes)
        self.bounds = (
            float(min(x[0] for x in node_xy.values())),
            float(max(x[0] for x in node_xy.values())),
            float(min(x[1] for x in node_xy.values())),
            float(max(x[1] for x in node_xy.values())),
        )

        face_lookup: Dict[Tuple[int, int], int] = {}
        face_nodes: List[Tuple[int, int]] = []
        owners: List[int] = []
        neighbors: List[int] = []
        cell_face_ids: List[List[int]] = [[] for _ in range(self.n_cells)]
        cell_face_signs: List[List[int]] = [[] for _ in range(self.n_cells)]

        for ci, ids_t in enumerate(self.cell_nodes):
            ids = list(ids_t)  # guaranteed CCW
            for k in range(len(ids)):
                a, b = ids[k], ids[(k + 1) % len(ids)]
                key = tuple(sorted((a, b)))
                if key not in face_lookup:
                    fi = len(face_nodes)
                    face_lookup[key] = fi
                    face_nodes.append((a, b))  # oriented outward from this (CCW) owner
                    owners.append(ci)
                    neighbors.append(-1)
                    cell_face_ids[ci].append(fi)
                    cell_face_signs[ci].append(+1)
                else:
                    fi = face_lookup[key]
                    if neighbors[fi] != -1:
                        raise ValueError(f"非流形网格：边 {key} 被三个或更多单元共享。")
                    # An interior edge shared by CCW cells must have opposite directions.
                    old_a, old_b = face_nodes[fi]
                    if (a, b) != (old_b, old_a):
                        raise ValueError(f"单元方向/拓扑不一致：共享边 {key} 未以相反方向出现。")
                    neighbors[fi] = ci
                    cell_face_ids[ci].append(fi)
                    cell_face_signs[ci].append(-1)

        self.face_nodes = np.asarray(face_nodes, dtype=int)
        self.owner = np.asarray(owners, dtype=int)
        self.neighbor = np.asarray(neighbors, dtype=int)
        self.n_faces = len(self.owner)
        self.cell_face_ids = [np.asarray(a, dtype=int) for a in cell_face_ids]
        self.cell_face_signs = [np.asarray(a, dtype=float) for a in cell_face_signs]

        p0 = np.asarray([node_xy[int(n)] for n in self.face_nodes[:, 0]])
        p1 = np.asarray([node_xy[int(n)] for n in self.face_nodes[:, 1]])
        self.face_centers = 0.5 * (p0 + p1)
        edge_vec = p1 - p0
        # For a CCW cell, the right-hand normal (dy, -dx) points outward.
        self.face_area_vectors = np.column_stack((edge_vec[:, 1], -edge_vec[:, 0]))
        self.face_areas = np.linalg.norm(self.face_area_vectors, axis=1)
        if np.any(self.face_areas <= self.geometry_tol):
            raise ValueError("发现长度接近零的网格边。")
        self.face_normals = self.face_area_vectors / self.face_areas[:, None]

        self.face_boundary_names: List[Optional[str]] = [None] * self.n_faces
        self.boundary_faces = np.where(self.neighbor < 0)[0]
        for fi in self.boundary_faces:
            key = tuple(sorted((int(self.face_nodes[fi, 0]), int(self.face_nodes[fi, 1]))))
            self.face_boundary_names[fi] = boundary_names.get(key, "unassigned")
        self.boundary_names = sorted({self.face_boundary_names[fi] for fi in self.boundary_faces})
        unassigned = [int(fi) for fi in self.boundary_faces if self.face_boundary_names[fi] == "unassigned"]
        if unassigned:
            warnings.warn(
                f"{len(unassigned)} 个外边界面没有匹配到 Physical Curve；这些边将命名为 'unassigned'。"
            )

        # Owner -> neighbour projected distance. Boundary distance is cell centre -> face centre.
        self.face_dproj = np.empty(self.n_faces, dtype=float)
        internal = self.neighbor >= 0
        o, n = self.owner[internal], self.neighbor[internal]
        d = self.centers[n] - self.centers[o]
        self.face_dproj[internal] = np.einsum("ij,ij->i", d, self.face_normals[internal])
        bnd = ~internal
        db = self.face_centers[bnd] - self.centers[self.owner[bnd]]
        self.face_dproj[bnd] = np.einsum("ij,ij->i", db, self.face_normals[bnd])
        if np.any(self.face_dproj <= self.geometry_tol):
            bad = np.where(self.face_dproj <= self.geometry_tol)[0][:10]
            raise ValueError(
                f"存在面法向投影距离非正/过小的单元对（{bad.tolist()}）。请检查单元质量、方向或网格退化。"
            )

        self.neighbors = [set() for _ in range(self.n_cells)]
        for o, n in zip(self.owner[internal], self.neighbor[internal]):
            self.neighbors[int(o)].add(int(n))
            self.neighbors[int(n)].add(int(o))
        print(
            f"[Mesh] {os.path.basename(self.filepath)}: cells={self.n_cells}, faces={self.n_faces}, "
            f"internal_faces={int(np.sum(internal))}, boundary_faces={len(self.boundary_faces)}, "
            f"boundary_names={self.boundary_names}"
        )


class UnstructuredSIMPLE:
    """Cell-centred, face-connectivity-based 2-D incompressible SIMPLE solver."""

    def __init__(self, mesh: UnstructuredGmshMesh, config: dict):
        self.mesh = mesh
        self.config = config
        fluid = config.get("fluid", {})
        self.rho = float(fluid.get("rho", 1.0))
        self.mu = float(fluid.get("mu", 0.01))
        if self.rho <= 0.0 or self.mu <= 0.0:
            raise ValueError("rho 和 mu 必须为正数。")

        raw_bc = config.get("boundary_conditions", {})
        raw_bc_lc = {str(k).lower(): dict(v) for k, v in raw_bc.items()}
        self.bc_cfg: Dict[str, dict] = {}
        for name in mesh.boundary_names:
            if name.lower() not in raw_bc_lc:
                raise KeyError(
                    f"网格物理边界 '{name}' 没有在 boundary_conditions 中定义；"
                    f"网格边界为 {mesh.boundary_names}，配置边界为 {list(raw_bc)}。"
                )
            bc = raw_bc_lc[name.lower()]
            kind = str(bc.get("type", "")).lower().replace("-", "_")
            aliases = {"velocity_inlet": "inlet", "fixed_velocity": "inlet", "outlet": "pressure_outlet"}
            kind = aliases.get(kind, kind)
            if kind not in {"wall", "inlet", "pressure_outlet"}:
                raise ValueError(f"边界 '{name}' 的 type='{kind}' 不支持；支持 wall/inlet/pressure_outlet。")
            bc["type"] = kind
            self.bc_cfg[name] = bc

        self.n = mesh.n_cells
        init = config.get("initial_conditions", {})
        self.u = np.full(self.n, float(init.get("u", 0.0)))
        self.v = np.full(self.n, float(init.get("v", 0.0)))
        self.p = np.full(self.n, float(init.get("p", 0.0)))
        self.nn_u: Optional[np.ndarray] = None
        self.nn_v: Optional[np.ndarray] = None
        self.nn_div: Optional[np.ndarray] = None
        self.last_residuals = None
        self.ref_cell = 0
        self.has_pressure_outlet = any(bc["type"] == "pressure_outlet" for bc in self.bc_cfg.values())
        self.last_diag_u_relaxed = np.ones(self.n)
        self.last_diag_v_relaxed = np.ones(self.n)

    def _face_bc(self, fi: int) -> dict:
        name = self.mesh.face_boundary_names[fi]
        if name not in self.bc_cfg:
            raise KeyError(f"边界面 {fi} 的物理名称 '{name}' 未定义。")
        return self.bc_cfg[name]

    def _boundary_values(self, variable: str) -> np.ndarray:
        """Face-centred Dirichlet values; NaN denotes a homogeneous-Neumann boundary."""
        vals = np.full(self.mesh.n_faces, np.nan, dtype=float)
        for fi in self.mesh.boundary_faces:
            bc = self._face_bc(int(fi))
            kind = bc["type"]
            if variable == "p" and kind == "pressure_outlet":
                vals[fi] = float(bc.get("p", 0.0))
            elif variable == "pp" and kind == "pressure_outlet":
                vals[fi] = 0.0
            elif variable in ("u", "v") and kind in ("wall", "inlet"):
                vals[fi] = float(bc.get(variable, 0.0))
        return vals

    def _gradient(self, phi: np.ndarray, face_dirichlet: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted least-squares gradient on each cell, optionally including Dirichlet faces."""
        m = self.mesh
        ncell = m.n_cells
        Mxx = np.zeros(ncell); Mxy = np.zeros(ncell); Myy = np.zeros(ncell)
        bx = np.zeros(ncell); by = np.zeros(ncell)
        internal = m.neighbor >= 0
        fi = np.where(internal)[0]
        o, nb = m.owner[fi], m.neighbor[fi]
        d = m.centers[nb] - m.centers[o]
        dp = phi[nb] - phi[o]
        r2 = np.maximum(np.einsum("ij,ij->i", d, d), 1e-30)
        w = 1.0 / r2
        xx, xy, yy = w * d[:, 0]**2, w * d[:, 0] * d[:, 1], w * d[:, 1]**2
        gx_b, gy_b = w * d[:, 0] * dp, w * d[:, 1] * dp
        np.add.at(Mxx, o, xx); np.add.at(Mxy, o, xy); np.add.at(Myy, o, yy)
        np.add.at(bx, o, gx_b); np.add.at(by, o, gy_b)
        np.add.at(Mxx, nb, xx); np.add.at(Mxy, nb, xy); np.add.at(Myy, nb, yy)
        np.add.at(bx, nb, gx_b); np.add.at(by, nb, gy_b)

        if face_dirichlet is not None:
            bfaces = m.boundary_faces
            use = np.isfinite(face_dirichlet[bfaces])
            fb = bfaces[use]
            if len(fb):
                ob = m.owner[fb]
                db = m.face_centers[fb] - m.centers[ob]
                deltap = face_dirichlet[fb] - phi[ob]
                rb2 = np.maximum(np.einsum("ij,ij->i", db, db), 1e-30)
                wb = 1.0 / rb2
                xx, xy, yy = wb * db[:, 0]**2, wb * db[:, 0] * db[:, 1], wb * db[:, 1]**2
                np.add.at(Mxx, ob, xx); np.add.at(Mxy, ob, xy); np.add.at(Myy, ob, yy)
                np.add.at(bx, ob, wb * db[:, 0] * deltap)
                np.add.at(by, ob, wb * db[:, 1] * deltap)

        det = Mxx * Myy - Mxy * Mxy
        scale = np.maximum(Mxx + Myy, 1.0)
        good = det > (1e-13 * scale * scale)
        gx = np.zeros(ncell); gy = np.zeros(ncell)
        gx[good] = (Myy[good] * bx[good] - Mxy[good] * by[good]) / det[good]
        gy[good] = (Mxx[good] * by[good] - Mxy[good] * bx[good]) / det[good]
        # Rare nearly rank-deficient cells: local least-squares fallback using neighbour differences.
        for ci in np.where(~good)[0]:
            rows, rhs = [], []
            for cj in m.neighbors[ci]:
                dd = m.centers[cj] - m.centers[ci]
                rows.append(dd)
                rhs.append(phi[cj] - phi[ci])
            if face_dirichlet is not None:
                for f in m.cell_face_ids[ci]:
                    val = face_dirichlet[f]
                    if m.neighbor[f] < 0 and np.isfinite(val):
                        rows.append(m.face_centers[f] - m.centers[ci])
                        rhs.append(val - phi[ci])
            if len(rows) >= 2:
                try:
                    sol, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
                    gx[ci], gy[ci] = sol
                except np.linalg.LinAlgError:
                    pass
        return gx, gy

    def _base_mass_flux(self) -> np.ndarray:
        """Upwind coefficient flux without Rhie-Chow pressure interpolation."""
        m = self.mesh
        flux = np.zeros(m.n_faces)
        internal = m.neighbor >= 0
        fi = np.where(internal)[0]
        o, nb = m.owner[fi], m.neighbor[fi]
        uf = 0.5 * (self.u[o] + self.u[nb])
        vf = 0.5 * (self.v[o] + self.v[nb])
        flux[fi] = self.rho * (uf * m.face_area_vectors[fi, 0] + vf * m.face_area_vectors[fi, 1])
        for fi in m.boundary_faces:
            bc = self._face_bc(int(fi))
            kind = bc["type"]
            if kind == "wall":
                flux[fi] = 0.0
            elif kind == "inlet":
                flux[fi] = self.rho * (
                    float(bc.get("u", 0.0)) * m.face_area_vectors[fi, 0]
                    + float(bc.get("v", 0.0)) * m.face_area_vectors[fi, 1]
                )
            else:  # pressure outlet, zero-gradient velocity or configured backflow
                c = m.owner[fi]
                flux[fi] = self.rho * (
                    self.u[c] * m.face_area_vectors[fi, 0]
                    + self.v[c] * m.face_area_vectors[fi, 1]
                )
        return flux

    def _rhie_chow_mass_flux(self, rAU_u: np.ndarray, rAU_v: np.ndarray) -> np.ndarray:
        m = self.mesh
        gradpx, gradpy = self._gradient(self.p, self._boundary_values("p"))
        flux = np.zeros(m.n_faces)
        internal = m.neighbor >= 0
        fi = np.where(internal)[0]
        o, nb = m.owner[fi], m.neighbor[fi]
        d = m.centers[nb] - m.centers[o]
        d2 = np.maximum(np.einsum("ij,ij->i", d, d), 1e-30)
        dp = self.p[nb] - self.p[o]
        gx_bar = 0.5 * (gradpx[o] + gradpx[nb])
        gy_bar = 0.5 * (gradpy[o] + gradpy[nb])
        corrx = gx_bar - dp * d[:, 0] / d2
        corry = gy_bar - dp * d[:, 1] / d2
        rux = 0.5 * (rAU_u[o] + rAU_u[nb])
        rvy = 0.5 * (rAU_v[o] + rAU_v[nb])
        uf = 0.5 * (self.u[o] + self.u[nb]) + rux * corrx
        vf = 0.5 * (self.v[o] + self.v[nb]) + rvy * corry
        flux[fi] = self.rho * (
            uf * m.face_area_vectors[fi, 0] + vf * m.face_area_vectors[fi, 1]
        )

        for fi in m.boundary_faces:
            bc = self._face_bc(int(fi))
            kind = bc["type"]
            if kind == "wall":
                flux[fi] = 0.0
            elif kind == "inlet":
                flux[fi] = self.rho * (
                    float(bc.get("u", 0.0)) * m.face_area_vectors[fi, 0]
                    + float(bc.get("v", 0.0)) * m.face_area_vectors[fi, 1]
                )
            else:  # pressure outlet: use prescribed face pressure in the RC correction
                c = m.owner[fi]
                d = m.face_centers[fi] - m.centers[c]
                d2 = max(float(np.dot(d, d)), 1e-30)
                dp = float(bc.get("p", 0.0)) - self.p[c]
                corrx = gradpx[c] - dp * d[0] / d2
                corry = gradpy[c] - dp * d[1] / d2
                uf = self.u[c] + rAU_u[c] * corrx
                vf = self.v[c] + rAU_v[c] * corry
                flux[fi] = self.rho * (
                    uf * m.face_area_vectors[fi, 0] + vf * m.face_area_vectors[fi, 1]
                )
        return flux

    def _net_cell_flux(self, face_flux: np.ndarray) -> np.ndarray:
        m = self.mesh
        net = np.zeros(m.n_cells)
        internal = m.neighbor >= 0
        fi = np.where(internal)[0]
        np.add.at(net, m.owner[fi], face_flux[fi])
        np.add.at(net, m.neighbor[fi], -face_flux[fi])
        bf = m.boundary_faces
        np.add.at(net, m.owner[bf], face_flux[bf])
        return net

    def _build_matrix(self, diagonal: np.ndarray, face_rows_o: np.ndarray,
                      face_cols_o: np.ndarray, face_vals_o: np.ndarray,
                      face_rows_n: Optional[np.ndarray] = None,
                      face_cols_n: Optional[np.ndarray] = None,
                      face_vals_n: Optional[np.ndarray] = None):
        n = len(diagonal)
        rows = [np.arange(n, dtype=int), np.asarray(face_rows_o, dtype=int)]
        cols = [np.arange(n, dtype=int), np.asarray(face_cols_o, dtype=int)]
        vals = [np.asarray(diagonal, dtype=float), np.asarray(face_vals_o, dtype=float)]
        if face_rows_n is not None and len(face_rows_n):
            rows.append(np.asarray(face_rows_n, dtype=int))
            cols.append(np.asarray(face_cols_n, dtype=int))
            vals.append(np.asarray(face_vals_n, dtype=float))
        A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
        A.sum_duplicates()
        return A

    def _build_momentum(self, direction: str, relaxed: bool = True,
                        include_nn: bool = True):
        """Assemble one momentum equation. Returns A, RHS, raw diagonal and relaxed diagonal."""
        if direction not in ("u", "v"):
            raise ValueError("direction must be 'u' or 'v'")
        m = self.mesh
        phi = self.u if direction == "u" else self.v
        phi_old = phi.copy()
        alpha = float(self.config.get("solver", {}).get("alpha_u" if direction == "u" else "alpha_v", 0.65))
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha_{direction} 必须位于 (0, 1]。")

        gradpx, gradpy = self._gradient(self.p, self._boundary_values("p"))
        gradphix, gradphiy = self._gradient(phi, self._boundary_values(direction))
        rhs = -m.volumes * (gradpx if direction == "u" else gradpy)
        diagonal = np.zeros(self.n)
        flux = self._base_mass_flux()

        internal = m.neighbor >= 0
        fi = np.where(internal)[0]
        o, nb = m.owner[fi], m.neighbor[fi]
        d = m.centers[nb] - m.centers[o]
        area = m.face_areas[fi]
        dproj = m.face_dproj[fi]
        D = self.mu * area / dproj
        sf = m.face_area_vectors[fi]
        s_nonorth = sf - (area / dproj)[:, None] * d
        gx_face = 0.5 * (gradphix[o] + gradphix[nb])
        gy_face = 0.5 * (gradphiy[o] + gradphiy[nb])
        Q = self.mu * (gx_face * s_nonorth[:, 0] + gy_face * s_nonorth[:, 1])
        np.add.at(rhs, o, Q)
        np.add.at(rhs, nb, -Q)

        F = flux[fi]
        np.add.at(diagonal, o, D + np.maximum(F, 0.0))
        np.add.at(diagonal, nb, D + np.maximum(-F, 0.0))
        off_o = -D + np.minimum(F, 0.0)  # row owner, column neighbour
        off_n = -D - np.maximum(F, 0.0)  # row neighbour, column owner

        for f in m.boundary_faces:
            fi0 = int(f)
            c = int(m.owner[fi0])
            bc = self._face_bc(fi0)
            kind = bc["type"]
            Ff = float(flux[fi0])
            if kind in ("wall", "inlet"):
                value = float(bc.get(direction, 0.0))
                dvec = m.face_centers[fi0] - m.centers[c]
                avec = m.face_areas[fi0]
                dproj_b = float(m.face_dproj[fi0])
                Db = self.mu * avec / dproj_b
                snon = m.face_area_vectors[fi0] - (avec / dproj_b) * dvec
                Qb = self.mu * (gradphix[c] * snon[0] + gradphiy[c] * snon[1])
                diagonal[c] += Db
                rhs[c] += Db * value + Qb
                if Ff >= 0.0:
                    diagonal[c] += Ff
                else:
                    rhs[c] += -Ff * value
            else:  # pressure outlet; velocity diffusion is zero-gradient
                if Ff >= 0.0:
                    diagonal[c] += Ff
                else:
                    backflow = float(bc.get("u_backflow" if direction == "u" else "v_backflow", 0.0))
                    rhs[c] += -Ff * backflow

        if include_nn:
            nn_source = self.nn_u if direction == "u" else self.nn_v
            if nn_source is not None:
                rhs += np.asarray(nn_source, dtype=float)

        if relaxed:
            diag_matrix = diagonal / alpha
            rhs = rhs + ((1.0 - alpha) / alpha) * diagonal * phi_old
        else:
            diag_matrix = diagonal.copy()

        if np.any(diag_matrix <= 0.0):
            bad = np.where(diag_matrix <= 0.0)[0][:10]
            raise RuntimeError(f"动量方程存在非正对角系数，单元索引示例: {bad.tolist()}。请检查网格和边界条件。")

        A = self._build_matrix(
            diag_matrix,
            o, nb, off_o,
            nb, o, off_n,
        )
        return A, rhs, diagonal, diag_matrix

    def _solve_pressure_correction(self, rAU_u: np.ndarray, rAU_v: np.ndarray):
        m = self.mesh
        face_flux = self._rhie_chow_mass_flux(rAU_u, rAU_v)
        net = self._net_cell_flux(face_flux)
        source_div = np.zeros(self.n) if self.nn_div is None else np.asarray(self.nn_div, dtype=float)
        rhs = -net + source_div * m.volumes

        diagonal = np.zeros(self.n)
        internal = m.neighbor >= 0
        fi = np.where(internal)[0]
        o, nb = m.owner[fi], m.neighbor[fi]
        nx, ny = m.face_normals[fi, 0], m.face_normals[fi, 1]
        rux_f = 0.5 * (rAU_u[o] + rAU_u[nb])
        rvy_f = 0.5 * (rAU_v[o] + rAU_v[nb])
        rAU_n = rux_f * nx**2 + rvy_f * ny**2
        C = self.rho * m.face_areas[fi] * rAU_n / m.face_dproj[fi]
        np.add.at(diagonal, o, C)
        np.add.at(diagonal, nb, C)
        off_o, off_n = -C, -C

        rows_o, cols_o, vals_o = [o], [nb], [off_o]
        rows_n, cols_n, vals_n = [nb], [o], [off_n]
        has_pressure_outlet = False
        for f in m.boundary_faces:
            fi0 = int(f)
            bc = self._face_bc(fi0)
            if bc["type"] != "pressure_outlet":
                continue
            has_pressure_outlet = True
            c = int(m.owner[fi0])
            nx_b, ny_b = m.face_normals[fi0]
            rAU_n_b = rAU_u[c] * nx_b**2 + rAU_v[c] * ny_b**2
            Cb = self.rho * m.face_areas[fi0] * rAU_n_b / m.face_dproj[fi0]
            diagonal[c] += Cb

        A = self._build_matrix(
            diagonal,
            np.concatenate(rows_o), np.concatenate(cols_o), np.concatenate(vals_o),
            np.concatenate(rows_n), np.concatenate(cols_n), np.concatenate(vals_n),
        )
        if not has_pressure_outlet:
            # Remove the pressure null space with a reference-cell constraint.
            A = A.tolil()
            A[self.ref_cell, :] = 0.0
            A[self.ref_cell, self.ref_cell] = 1.0
            rhs[self.ref_cell] = 0.0
            A = A.tocsr()

        pp = spsolve(A, rhs)
        if not np.all(np.isfinite(pp)):
            raise RuntimeError("压力修正方程出现 NaN/Inf；请检查边界条件、网格质量和矩阵系数。")
        return pp, net, face_flux

    def _correct_fields(self, pp: np.ndarray, rAU_u: np.ndarray, rAU_v: np.ndarray, alpha_p: float) -> None:
        gpx, gpy = self._gradient(pp, self._boundary_values("pp"))
        self.u -= rAU_u * gpx
        self.v -= rAU_v * gpy
        self.p += alpha_p * pp
        if not any(bc["type"] == "pressure_outlet" for bc in self.bc_cfg.values()):
            self.p -= self.p[self.ref_cell]

    @staticmethod
    def _relative_change(new: np.ndarray, old: np.ndarray) -> float:
        delta = float(np.mean(np.abs(new - old)))
        scale = float(np.max(np.abs(old)))
        if scale < 1e-30:
            return delta
        return delta / scale

    def solve(self, verbose: bool = True):
        scfg = self.config.get("solver", {})
        max_iter = int(scfg.get("max_iter", 2000))
        tol = float(scfg.get("tol", 1e-6))
        alpha_p = float(scfg.get("alpha_p", 0.25))
        if not (0.0 < alpha_p <= 1.0):
            raise ValueError("alpha_p 必须位于 (0, 1]。")
        log_every = int(scfg.get("log_every", 100))

        if verbose:
            print("=" * 76)
            print(f"Unstructured SIMPLE | case={self.config.get('case_name', 'Unnamed')} | cells={self.n}")
            print(f"Boundary groups: {self.mesh.boundary_names}; NN correction={self.nn_u is not None}")
            print("=" * 76)

        res_u = res_v = mass_resid = np.inf
        for it in range(1, max_iter + 1):
            u_old, v_old = self.u.copy(), self.v.copy()

            A_u, b_u, raw_u, diag_u_rel = self._build_momentum("u", relaxed=True, include_nn=True)
            self.u = spsolve(A_u, b_u)
            self.last_diag_u_relaxed = diag_u_rel.copy()

            A_v, b_v, raw_v, diag_v_rel = self._build_momentum("v", relaxed=True, include_nn=True)
            self.v = spsolve(A_v, b_v)
            self.last_diag_v_relaxed = diag_v_rel.copy()

            rAU_u = self.mesh.volumes / np.maximum(diag_u_rel, 1e-30)
            rAU_v = self.mesh.volumes / np.maximum(diag_v_rel, 1e-30)
            pp, _, _ = self._solve_pressure_correction(rAU_u, rAU_v)
            self._correct_fields(pp, rAU_u, rAU_v, alpha_p)

            res_u = self._relative_change(self.u, u_old)
            res_v = self._relative_change(self.v, v_old)
            # Check the continuity equation with the same RC face flux used by pressure correction.
            face_flux = self._rhie_chow_mass_flux(rAU_u, rAU_v)
            net = self._net_cell_flux(face_flux)
            source_div = np.zeros(self.n) if self.nn_div is None else np.asarray(self.nn_div, dtype=float)
            mass_resid = float(np.max(np.abs(net / self.mesh.volumes - source_div)))

            if verbose and (it == 1 or it % log_every == 0):
                speed_max = float(np.max(np.hypot(self.u, self.v)))
                print(
                    f"iter {it:5d} | dU={res_u:.3e} | dV={res_v:.3e} | "
                    f"continuity={mass_resid:.3e} | Umax={speed_max:.5g}"
                )
            if it >= 10 and res_u < tol and res_v < tol and mass_resid < tol:
                if verbose:
                    print(f"Converged at iter {it}: dU={res_u:.3e}, dV={res_v:.3e}, continuity={mass_resid:.3e}")
                break
        else:
            if verbose:
                print(f"Reached max_iter={max_iter}: dU={res_u:.3e}, dV={res_v:.3e}, continuity={mass_resid:.3e}")

        self.last_residuals = (res_u, res_v, mass_resid)
        return self.last_residuals


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test; polygon connectivity is taken from the mesh itself."""
    x, y = float(point[0]), float(point[1])
    inside = False
    nvert = len(polygon)
    for i in range(nvert):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % nvert]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def sample_dense_to_coarse(mesh_dense: UnstructuredGmshMesh,
                           solver_dense: UnstructuredSIMPLE,
                           mesh_coarse: UnstructuredGmshMesh):
    """Topology-aware dense-to-coarse mapping using containing-cell linear reconstruction.

    Unlike global scattered interpolation, this routine first locates each coarse centroid in
    an actual dense cell polygon. It therefore does not deliberately triangulate across a solid
    obstacle or a disconnected fluid region. A nearest-cell fallback is retained for unmatched
    points (e.g. slightly nonmatching geometries), and is reported to the user.
    """
    points = mesh_dense.centers
    query = mesh_coarse.centers
    tree = cKDTree(points)
    grad_u = solver_dense._gradient(solver_dense.u, solver_dense._boundary_values("u"))
    grad_v = solver_dense._gradient(solver_dense.v, solver_dense._boundary_values("v"))
    grad_p = solver_dense._gradient(solver_dense.p, solver_dense._boundary_values("p"))
    fields = (solver_dense.u, solver_dense.v, solver_dense.p)
    gradients = (grad_u, grad_v, grad_p)
    sampled = [np.empty(len(query), dtype=float) for _ in fields]
    fallback_count = 0
    k0 = min(32, len(points))
    k1 = min(128, len(points))

    for qi, point in enumerate(query):
        _, cand = tree.query(point, k=k0)
        cand = np.atleast_1d(cand).astype(int)
        hit = None
        for ci in cand:
            if _point_in_polygon(point, mesh_dense.polygons[int(ci)]):
                hit = int(ci)
                break
        if hit is None and k1 > k0:
            _, cand2 = tree.query(point, k=k1)
            for ci in np.atleast_1d(cand2).astype(int):
                if _point_in_polygon(point, mesh_dense.polygons[int(ci)]):
                    hit = int(ci)
                    break
        if hit is None:
            # The meshes should describe the same fluid domain. This branch is a defensive
            # fallback and should be inspected if it affects a substantial fraction of cells.
            _, hit = tree.query(point, k=1)
            hit = int(hit)
            fallback_count += 1
        delta = point - mesh_dense.centers[hit]
        for k, (values, grad) in enumerate(zip(fields, gradients)):
            sampled[k][qi] = values[hit] + grad[0][hit] * delta[0] + grad[1][hit] * delta[1]

    if fallback_count:
        warnings.warn(
            f"{fallback_count}/{len(query)} coarse cell centres were not located inside any of the "
            "candidate dense-cell polygons; nearest-cell reconstruction was used. Check that both "
            "meshes cover the same fluid domain."
        )
    return tuple(sampled)


def compute_discrete_residual(solver: UnstructuredSIMPLE,
                              u_sample: np.ndarray,
                              v_sample: np.ndarray,
                              p_sample: np.ndarray):
    """Evaluate coarse-grid momentum and continuity residuals at an interpolated reference field."""
    solver.u = np.asarray(u_sample, dtype=float).copy()
    solver.v = np.asarray(v_sample, dtype=float).copy()
    solver.p = np.asarray(p_sample, dtype=float).copy()
    if not (len(solver.u) == len(solver.v) == len(solver.p) == solver.n):
        raise ValueError("采样场长度必须等于粗网格单元数。")

    A_u, b_u, raw_u, _ = solver._build_momentum("u", relaxed=False, include_nn=False)
    res_u = np.asarray(A_u @ solver.u - b_u).ravel()
    A_v, b_v, raw_v, _ = solver._build_momentum("v", relaxed=False, include_nn=False)
    res_v = np.asarray(A_v @ solver.v - b_v).ravel()

    alpha_u = float(solver.config.get("solver", {}).get("alpha_u", 0.65))
    alpha_v = float(solver.config.get("solver", {}).get("alpha_v", 0.65))
    rAU_u = alpha_u * solver.mesh.volumes / np.maximum(raw_u, 1e-30)
    rAU_v = alpha_v * solver.mesh.volumes / np.maximum(raw_v, 1e-30)
    face_flux = solver._rhie_chow_mass_flux(rAU_u, rAU_v)
    res_div = solver._net_cell_flux(face_flux) / solver.mesh.volumes
    return res_u, res_v, res_div, solver.u.copy(), solver.v.copy(), solver.p.copy()


class HybridFourierRBFNet(nn.Module):
    """Global Fourier features plus local Gaussian RBF centres."""

    def __init__(self, centers: torch.Tensor, widths: torch.Tensor,
                 hidden_dim: int = 96, num_freqs: int = 4):
        super().__init__()
        freqs = torch.arange(1, num_freqs + 1, dtype=torch.float32) * np.pi
        self.register_buffer("freqs", freqs)
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)
        feature_dim = 2 + 2 * num_freqs * 2 + len(centers)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        proj = (x.unsqueeze(-1) * self.freqs).reshape(x.shape[0], -1)
        dist_sq = torch.sum((x.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2, dim=-1)
        rbf = torch.exp(-dist_sq / (2.0 * self.widths.unsqueeze(0).clamp_min(1e-6) ** 2))
        features = torch.cat([x, torch.sin(proj), torch.cos(proj), rbf], dim=-1)
        return self.net(features)


class UniversalNeuralCorrector:
    """Fit three coordinate-conditioned residual fields for one coarse mesh/case."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, nn.Module] = {}
        self.stats: Dict[str, Tuple[float, float]] = {}
        self.centers: Optional[torch.Tensor] = None
        self.widths: Optional[torch.Tensor] = None
        self.bounds = None
        self.has_pressure_outlet = True

    @staticmethod
    def _normalised_coordinates(mesh: UnstructuredGmshMesh):
        xmin, xmax, ymin, ymax = mesh.bounds
        lx, ly = max(xmax - xmin, 1e-12), max(ymax - ymin, 1e-12)
        coords = np.column_stack(((mesh.centers[:, 0] - xmin) / lx,
                                  (mesh.centers[:, 1] - ymin) / ly))
        return coords, (xmin, xmax, ymin, ymax)

    def prepare_training_data(self, solver: UnstructuredSIMPLE,
                              res_u: np.ndarray, res_v: np.ndarray, res_div: np.ndarray):
        coords_np, self.bounds = self._normalised_coordinates(solver.mesh)
        self.has_pressure_outlet = solver.has_pressure_outlet
        X = torch.tensor(coords_np, dtype=torch.float32, device=self.device)
        targets = {"u": np.asarray(res_u), "v": np.asarray(res_v), "div": np.asarray(res_div)}
        if any(len(v) != solver.n for v in targets.values()):
            raise ValueError("NN 残差目标的长度必须等于粗网格单元数。")

        importance = sum(np.abs(a) / (np.std(a) + 1e-12) for a in targets.values())
        n_samples = len(coords_np)
        n_peak = min(96, max(1, n_samples // 2))
        peak_idx = np.argsort(importance)[-n_peak:]
        n_uniform = min(64, n_samples)
        uniform_idx = np.linspace(0, n_samples - 1, n_uniform, dtype=int)
        selected = np.unique(np.concatenate((peak_idx, uniform_idx)))
        centres_once = coords_np[selected]
        centres_twice = np.vstack((centres_once, centres_once))
        widths = np.concatenate((np.full(len(centres_once), 0.025),
                                 np.full(len(centres_once), 0.080))).astype(np.float32)
        self.centers = torch.tensor(centres_twice, dtype=torch.float32, device=self.device)
        self.widths = torch.tensor(widths, dtype=torch.float32, device=self.device)

        y_tensors = {}
        for key, values in targets.items():
            mean = float(np.mean(values))
            std = float(np.std(values)) + 1e-12
            self.stats[key] = (mean, std)
            y_tensors[key] = torch.tensor((values - mean) / std, dtype=torch.float32,
                                          device=self.device).reshape(-1, 1)
        return X, y_tensors

    def train(self, X_train: torch.Tensor, y_tensors: dict, nn_cfg: dict):
        epochs = int(nn_cfg.get("epochs", 1500))
        lr = float(nn_cfg.get("lr", 3e-3))
        hidden_dim = int(nn_cfg.get("hidden_dim", 96))
        batch_size = int(nn_cfg.get("batch_size", min(8192, len(X_train))))
        batch_size = max(1, min(batch_size, len(X_train)))
        criterion = nn.MSELoss()
        if self.centers is None or self.widths is None:
            raise RuntimeError("请先调用 prepare_training_data。")
        if len(X_train) == 0:
            raise ValueError("训练样本为空。")

        if len(X_train) > batch_size:
            print(f"[NN] samples={len(X_train)}, mini_batch={batch_size}, device={self.device}")
        else:
            print(f"[NN] samples={len(X_train)}, full_batch, device={self.device}")

        for key in ("u", "v", "div"):
            model = HybridFourierRBFNet(self.centers, self.widths,
                                        hidden_dim=hidden_dim, num_freqs=4).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)
            for epoch in range(1, epochs + 1):
                if len(X_train) > batch_size:
                    order = torch.randperm(len(X_train), device=self.device)
                else:
                    order = None
                running_loss, seen = 0.0, 0
                starts = range(0, len(X_train), batch_size)
                for start in starts:
                    ids = order[start:start + batch_size] if order is not None else slice(start, start + batch_size)
                    xb, yb = X_train[ids], y_tensors[key][ids]
                    optimizer.zero_grad(set_to_none=True)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                    running_loss += float(loss.item()) * len(xb)
                    seen += len(xb)
                scheduler.step()
                if epoch == 1 or epoch % 500 == 0 or epoch == epochs:
                    print(f"[NN:{key}] epoch={epoch:4d}/{epochs}, norm_MSE={running_loss/max(seen,1):.4e}, "
                          f"lr={optimizer.param_groups[0]['lr']:.2e}")
            model.eval()
            self.models[key] = model

    def predict(self, mesh: UnstructuredGmshMesh):
        coords_np, _ = self._normalised_coordinates(mesh)
        X = torch.tensor(coords_np, dtype=torch.float32, device=self.device)
        predictions = {}
        with torch.no_grad():
            for key in ("u", "v", "div"):
                if key not in self.models:
                    raise RuntimeError("神经网络尚未训练。")
                mean, std = self.stats[key]
                predictions[key] = self.models[key](X).cpu().numpy().reshape(-1) * std + mean
        # A closed domain has no net volumetric source; remove a volume-weighted mean drift.
        if not self.has_pressure_outlet:
            vol = mesh.volumes
            predictions["div"] -= np.sum(predictions["div"] * vol) / np.sum(vol)
        return predictions["u"], predictions["v"], predictions["div"]


def _plot_cell_field(ax, mesh: UnstructuredGmshMesh, values: np.ndarray, title: str,
                     cmap: str = "viridis", vmin=None, vmax=None):
    pc = PolyCollection(mesh.polygons, array=np.asarray(values), cmap=cmap,
                        edgecolors="none", linewidths=0.0)
    if vmin is not None or vmax is not None:
        pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    xmin, xmax, ymin, ymax = mesh.bounds
    padx = 0.015 * max(xmax - xmin, 1e-12)
    pady = 0.015 * max(ymax - ymin, 1e-12)
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
    return pc


def plot_nn_source_terms(res_u, res_v, res_div, pred_u, pred_v, pred_div,
                         mesh: UnstructuredGmshMesh, case_name: str):
    truth = [res_u, res_v, res_div]
    pred = [pred_u, pred_v, pred_div]
    names = ["Res_u (momentum)", "Res_v (momentum)", "Res_div (continuity)"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.suptitle(f"{case_name}: residual truth / NN / error", fontsize=13, fontweight="bold")
    for r, (t, p, name) in enumerate(zip(truth, pred, names)):
        diff = p - t
        vmax = max(float(np.max(np.abs(t))), float(np.max(np.abs(p))), 1e-14)
        emax = max(float(np.max(np.abs(diff))), 1e-14)
        _plot_cell_field(axes[r, 0], mesh, t, f"True {name}", "RdBu_r", -vmax, vmax)
        _plot_cell_field(axes[r, 1], mesh, p, f"NN predicted {name}", "RdBu_r", -vmax, vmax)
        _plot_cell_field(axes[r, 2], mesh, diff, f"Error (max={emax:.2e})", "RdBu_r", -emax, emax)
    fig.tight_layout()
    path = f"./{case_name}_nn_source_terms.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"[Plot] saved {path}")
    plt.show()


def plot_flow_field(mesh_dense, solver_dense, mesh_coarse, solver_coarse_orig,
                    solver_coarse_nn, case_name: str):
    items = [(mesh_dense, solver_dense, f"Dense reference ({mesh_dense.n_cells} cells)"),
             (mesh_coarse, solver_coarse_orig, f"Coarse original ({mesh_coarse.n_cells} cells)"),
             (mesh_coarse, solver_coarse_nn, f"Coarse NN-corrected ({mesh_coarse.n_cells} cells)")]
    speed_max = max(float(np.max(np.hypot(solver_dense.u, solver_dense.v))), 1e-12)
    pmin = min(float(np.min(solver_dense.p)), float(np.min(solver_coarse_orig.p)), float(np.min(solver_coarse_nn.p)))
    pmax = max(float(np.max(solver_dense.p)), float(np.max(solver_coarse_orig.p)), float(np.max(solver_coarse_nn.p)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    fig.suptitle(f"{case_name}: dense / coarse baseline / coarse NN-corrected", fontsize=13, fontweight="bold")
    for j, (mesh, sol, title) in enumerate(items):
        _plot_cell_field(axes[0, j], mesh, np.hypot(sol.u, sol.v), f"Speed | {title}", "viridis", 0.0, speed_max)
        _plot_cell_field(axes[1, j], mesh, sol.p, f"Pressure | {title}", "coolwarm", pmin, pmax)
    fig.tight_layout()
    path = f"./{case_name}_flow_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"[Plot] saved {path}")
    plt.show()


def _profile_band(mesh, coordinate_axis: int, value: float, fraction: float = 0.025):
    span = (mesh.bounds[3] - mesh.bounds[2]) if coordinate_axis == 1 else (mesh.bounds[1] - mesh.bounds[0])
    delta = np.abs(mesh.centers[:, coordinate_axis] - value)
    limit = max(fraction * span, 1e-12)
    ids = np.where(delta <= limit)[0]
    if len(ids) < min(5, mesh.n_cells):
        ids = np.argsort(delta)[:min(5, mesh.n_cells)]
    return ids


def plot_profile_and_error(mesh_dense, solver_dense, mesh_coarse, solver_coarse_orig,
                           solver_coarse_nn, u_sampled, v_sampled, case_name: str):
    xmin, xmax, ymin, ymax = mesh_dense.bounds
    ymid, xmid = 0.5 * (ymin + ymax), 0.5 * (xmin + xmax)
    ids_d = _profile_band(mesh_dense, 1, ymid)
    ids_o = _profile_band(mesh_coarse, 1, ymid)
    ids_c = _profile_band(mesh_coarse, 0, xmid)
    ids_dv = _profile_band(mesh_dense, 0, xmid)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].scatter(mesh_dense.centers[ids_d, 0], solver_dense.u[ids_d], s=10, label="Dense reference")
    axes[0, 0].scatter(mesh_coarse.centers[ids_o, 0], solver_coarse_orig.u[ids_o], s=18, marker="x", label="Coarse original")
    axes[0, 0].scatter(mesh_coarse.centers[ids_o, 0], solver_coarse_nn.u[ids_o], s=16, marker="+", label="Coarse NN")
    axes[0, 0].set_title(f"u(x) near y={ymid:.4g}")
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("u"); axes[0, 0].legend(fontsize=8)

    axes[0, 1].scatter(mesh_dense.centers[ids_dv, 1], solver_dense.v[ids_dv], s=10, label="Dense reference")
    axes[0, 1].scatter(mesh_coarse.centers[ids_c, 1], solver_coarse_orig.v[ids_c], s=18, marker="x", label="Coarse original")
    axes[0, 1].scatter(mesh_coarse.centers[ids_c, 1], solver_coarse_nn.v[ids_c], s=16, marker="+", label="Coarse NN")
    axes[0, 1].set_title(f"v(y) near x={xmid:.4g}")
    axes[0, 1].set_xlabel("y"); axes[0, 1].set_ylabel("v"); axes[0, 1].legend(fontsize=8)

    err_u_orig = solver_coarse_orig.u - u_sampled
    err_u_nn = solver_coarse_nn.u - u_sampled
    err_v_orig = solver_coarse_orig.v - v_sampled
    err_v_nn = solver_coarse_nn.v - v_sampled
    labels = ["u max", "u RMSE", "v max", "v RMSE"]
    orig = [np.max(np.abs(err_u_orig)), np.sqrt(np.mean(err_u_orig**2)),
            np.max(np.abs(err_v_orig)), np.sqrt(np.mean(err_v_orig**2))]
    corrected = [np.max(np.abs(err_u_nn)), np.sqrt(np.mean(err_u_nn**2)),
                 np.max(np.abs(err_v_nn)), np.sqrt(np.mean(err_v_nn**2))]
    x = np.arange(len(labels)); w = 0.36
    # A small positive floor keeps the logarithmic axis valid for exact-zero errors.
    axes[1, 0].bar(x - w / 2, np.maximum(orig, 1e-30), w, label="Coarse original")
    axes[1, 0].bar(x + w / 2, np.maximum(corrected, 1e-30), w, label="Coarse NN")
    axes[1, 0].set_xticks(x, labels, rotation=15)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Error metrics vs sampled dense reference")
    axes[1, 0].legend(fontsize=8)

    improvement = []
    for a, b in zip(orig, corrected):
        improvement.append(100.0 * (a - b) / max(a, 1e-30))
    axes[1, 1].bar(labels, improvement)
    axes[1, 1].axhline(0.0, linewidth=0.8)
    axes[1, 1].set_title("Error reduction after NN (%)")
    axes[1, 1].tick_params(axis="x", rotation=15)
    axes[1, 1].set_ylabel("Improvement (%)")

    fig.tight_layout()
    path = f"./{case_name}_profile_error.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"[Plot] saved {path}")
    plt.show()
    return err_u_orig, err_u_nn, err_v_orig, err_v_nn


def run_universal_pipeline(config: dict, plot_results: bool = True):
    case_name = config.get("case_name", "flow_case")
    dense_path = config["meshes"]["dense_msh"]
    coarse_path = config["meshes"]["coarse_msh"]
    print("\n[Step 0] Reading Gmsh connectivity meshes")
    mesh_dense = UnstructuredGmshMesh(dense_path)
    mesh_coarse = UnstructuredGmshMesh(coarse_path)
    print(f"Dense: cells={mesh_dense.n_cells}, boundaries={mesh_dense.boundary_names}")
    print(f"Coarse: cells={mesh_coarse.n_cells}, boundaries={mesh_coarse.boundary_names}")

    print("\n[Step 1] Solve the dense reference case")
    solver_dense = UnstructuredSIMPLE(mesh_dense, config)
    solver_dense.solve(verbose=True)

    print("\n[Step 2-3] Map dense fields to coarse cell centres and evaluate coarse discrete residuals")
    u_sampled, v_sampled, p_sampled = sample_dense_to_coarse(mesh_dense, solver_dense, mesh_coarse)
    solver_template = UnstructuredSIMPLE(mesh_coarse, config)
    res_u, res_v, res_div, u_sampled, v_sampled, p_sampled = compute_discrete_residual(
        solver_template, u_sampled, v_sampled, p_sampled
    )
    for name, arr in (("Res_u", res_u), ("Res_v", res_v), ("Res_div", res_div)):
        print(f"{name}: [{arr.min():.4e}, {arr.max():.4e}], mean|.|={np.mean(np.abs(arr)):.4e}")

    print("\n[Step 4] Train Fourier/RBF models for the coarse-grid residual fields")
    corrector = UniversalNeuralCorrector()
    X_train, y_tensors = corrector.prepare_training_data(solver_template, res_u, res_v, res_div)
    corrector.train(X_train, y_tensors, config.get("nn_config", {}))
    pred_u, pred_v, pred_div = corrector.predict(mesh_coarse)
    if plot_results:
        plot_nn_source_terms(res_u, res_v, res_div, pred_u, pred_v, pred_div, mesh_coarse, case_name)

    print("\n[Step 5] Solve coarse baseline without NN correction")
    solver_coarse_orig = UnstructuredSIMPLE(mesh_coarse, config)
    solver_coarse_orig.solve(verbose=True)

    print("\n[Step 6] Solve coarse case with NN residual-source correction")
    solver_coarse_nn = UnstructuredSIMPLE(mesh_coarse, config)
    solver_coarse_nn.nn_u = pred_u.copy()
    solver_coarse_nn.nn_v = pred_v.copy()
    solver_coarse_nn.nn_div = pred_div.copy()
    solver_coarse_nn.solve(verbose=True)

    if plot_results:
        print("\n[Step 7] Plot flow fields, profiles and error statistics")
        plot_profile_and_error(mesh_dense, solver_dense, mesh_coarse, solver_coarse_orig,
                               solver_coarse_nn, u_sampled, v_sampled, case_name)
        plot_flow_field(mesh_dense, solver_dense, mesh_coarse,
                        solver_coarse_orig, solver_coarse_nn, case_name)

    return solver_dense, solver_coarse_orig, solver_coarse_nn


if __name__ == "__main__":
    dense_path = "../meshes/back_fine.msh" if os.path.exists("../meshes/back_fine.msh") else "./meshes/back_fine.msh"
#    coarse_path = "../meshes/back_coarse.msh" if os.path.exists("../meshes/back_coarse.msh") else "./meshes/back_coarse.msh"
    coarse_path = "../meshes/back_coarse_coarse.msh" if os.path.exists("../meshes/back_coarse_coarse.msh") else "./meshes/back_coarse_coarse.msh"

    config_backward = {
        "case_name": "backward",
        "meshes": {"dense_msh": dense_path, "coarse_msh": coarse_path},
        "fluid": {"rho": 1.0, "mu": 0.01},
        "initial_conditions": {"u": 0.0, "v": 0.0, "p": 0.0},
        "boundary_conditions": {
            "wall": {"type": "wall", "u": 0.0, "v": 0.0},
            "outlet": {"type": "pressure_outlet", "p": 0.0},
            "inlet": {"type": "inlet", "u": 1.0, "v": 0.0},
        },
        "solver": {
            "max_iter": 2000, "tol": 1e-7,
            "alpha_u": 0.7, "alpha_v": 0.7, "alpha_p": 0.3,
            "log_every": 100,
        },
        "nn_config": {"epochs": 4000, "lr": 3e-3, "hidden_dim": 96, "batch_size": 8192},
    }
    run_universal_pipeline(config_backward, plot_results=True)
