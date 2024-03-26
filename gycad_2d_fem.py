import enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.sparse import linalg as sla
from scipy.sparse import coo_matrix
from FEM2D.assembly import StiffElasAssembling2DP1base


def check_intersection(lines_list, line):
    for line_2 in lines_list:
        if check_intersection_two_lines(line, line_2):
            return True

    return False


def check_intersection_two_lines(line_1, line_2):
    # find static coords

    if line_1[0][1] == line_1[1][1]:
        horizontal_1 = True
        static_coord_1 = line_1[0][1]
    else:
        horizontal_1 = False
        static_coord_1 = line_1[0][0]

    if line_2[0][1] == line_2[1][1]:
        horizontal_2 = True
        static_coord_2 = line_2[0][1]
    else:
        horizontal_2 = False
        static_coord_2 = line_2[0][0]

    if horizontal_1 == horizontal_2:
        if static_coord_1 == static_coord_2:
            if horizontal_1:
                dyn_idx = 0
            else:
                dyn_idx = 1

            r_11 = line_1[0][dyn_idx]
            r_12 = line_1[1][dyn_idx]
            r_21 = line_2[0][dyn_idx]
            r_22 = line_2[1][dyn_idx]

            if r_11 <= r_21 and r_21 <= r_12:
                return True
            elif r_11 <= r_22 and r_22 <= r_12:
                return True

            elif r_21 <= r_11 and r_21 <= r_22:
                return True

            elif r_21 <= r_12 and r_12 <= r_22:
                return True

            else:
                return False

        else:
            return False
    else:
        return False

    # check if on one line


def in_rects(point, rects):
    for rect in rects:
        if in_rect(point, rect):
            return True

    return False


def in_rect(point, rect):
    if rect[0][0] <= point[0] and point[0] <= rect[1][0]:
        if rect[0][1] <= point[1] and point[1] <= rect[1][1]:
            return True

        else:
            return False
    else:
        return False


def point_on_line(point, line):
    if line[0][0] == line[1][0] and line[0][0] == point[0]:  # one one vertical line
        if line[0][1] <= point[1] and point[1] <= line[1][1]:
            return True
        else:
            return False

    elif line[0][1] == line[1][1] and line[1][1] == point[1]:  # on one horizontal line
        if line[0][0] <= point[0] and point[0] <= line[1][0]:
            return True
        else:
            return False

    else:
        return False


def point_on_lines(point, lines):
    for i, line in enumerate(lines):
        if point_on_line(point, line):
            return True, i

    return False, 0


def build_mesh(x_points, y_points, excluding_areas, constraint_lines, force_lines):
    q = []
    me = []
    areas = []

    constraint_idx = [[] for i in constraint_lines]
    force_idx = [[] for i in force_lines]

    for eli in range(x_points.shape[0] - 1):
        for elj in range(y_points.shape[0] - 1):
            point_1 = [x_points[eli], y_points[elj]]
            point_2 = [x_points[eli], y_points[elj + 1]]
            point_3 = [x_points[eli + 1], y_points[elj + 1]]
            point_4 = [x_points[eli + 1], y_points[elj]]

            # check if all points inside excluding box
            if in_rects(point_1, excluding_areas):
                if in_rects(point_2, excluding_areas):
                    if in_rects(point_3, excluding_areas):
                        if in_rects(point_4, excluding_areas):
                            continue

            if point_1 not in q:
                q.append(point_1)
                point_1_idx = len(q) - 1
                c_check_pol, c_line_index = point_on_lines(point_1, constraint_lines)
                if c_check_pol:
                    constraint_idx[c_line_index].append(point_1_idx)

                else:
                    f_check_pol, f_line_index = point_on_lines(point_1, force_lines)
                    if f_check_pol:
                        force_idx[f_line_index].append(point_1_idx)

            else:
                point_1_idx = q.index(point_1)

            if point_2 not in q:
                q.append(point_2)
                point_2_idx = len(q) - 1
                c_check_pol, c_line_index = point_on_lines(point_2, constraint_lines)
                if c_check_pol:
                    constraint_idx[c_line_index].append(point_2_idx)

                else:
                    f_check_pol, f_line_index = point_on_lines(point_2, force_lines)
                    if f_check_pol:
                        force_idx[f_line_index].append(point_2_idx)

            else:
                point_2_idx = q.index(point_2)

            if point_3 not in q:
                q.append(point_3)
                point_3_idx = len(q) - 1
                c_check_pol, c_line_index = point_on_lines(point_3, constraint_lines)
                if c_check_pol:
                    constraint_idx[c_line_index].append(point_3_idx)

                else:
                    f_check_pol, f_line_index = point_on_lines(point_3, force_lines)
                    if f_check_pol:
                        force_idx[f_line_index].append(point_3_idx)

            else:
                point_3_idx = q.index(point_3)

            if point_4 not in q:
                q.append(point_4)
                point_4_idx = len(q) - 1
                c_check_pol, c_line_index = point_on_lines(point_4, constraint_lines)
                if c_check_pol:
                    constraint_idx[c_line_index].append(point_4_idx)

                else:
                    f_check_pol, f_line_index = point_on_lines(point_4, force_lines)
                    if f_check_pol:
                        force_idx[f_line_index].append(point_4_idx)

            else:
                point_4_idx = q.index(point_4)

            area = (
                (point_3[0] - point_1[0]) * (point_3[1] - point_1[1]) / 2
            )  # check if it is positive

            # two the elements with the same area
            areas.append(area)
            areas.append(area)

            me.append([point_1_idx, point_2_idx, point_3_idx])
            me.append([point_3_idx, point_4_idx, point_1_idx])

    return q, me, areas, constraint_idx, force_idx


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N

    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        "trunc({name},{a:.2f},{b:.2f})".format(name=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )

    return new_cmap


def lk(E=1, nu=0.3):
    """
    The function computes the stiffness matrix for Q4 element
    with given material properties:
        E is Youngs Modulus;
        nu is Poisson's ratio.

    return:
        stiffniss matirix - np.array with shape (8, 8)
    """
    # E=1 # ! need to synchronize material properties
    k = np.array(
        [
            1 / 2 - nu / 6,
            1 / 8 + nu / 8,
            -1 / 4 - nu / 12,
            -1 / 8 + 3 * nu / 8,
            -1 / 4 + nu / 12,
            -1 / 8 - nu / 8,
            nu / 6,
            1 / 8 - 3 * nu / 8,
        ]
    )
    KE = (
        E
        / (1 - nu**2)
        * np.array(
            [
                [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
            ]
        )
    )

    return KE


def oc(nelx, nely, x, volfrac, dc, dv, g):
    """
    The function finds next X for optimization in SIMP method
    and upadates optimality criterion.
    """
    l1 = 0
    l2 = 1e9
    move = 0.2
    # reshape to perform vector operations
    xnew = np.zeros(nelx * nely)
    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        xnew[:] = np.maximum(
            0.0,
            np.maximum(
                x - move,
                np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid))),
            ),
        )

        gt = g + np.sum((dv * (xnew - x)))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
    return (xnew, gt)


def el_mask_2_nod_mask(el_mask, nelx, nely, ndof):
    """
    The function converts mask of void elements to
    mask of void nodes. It is used for postprocessing
    Inputs:
        el_mask - np.array, (nelx, nely), 0 if the element is void, 1 if element is material
        nelx    - int, element number per row
        nely    - int, element number per column
        ndof    - int, number of dofs (degrees of freedom)

    Return:
        node_mask - np.array, (number_of_nodes, 1), 0 if the node is void, 1 if the node is material
    """
    node_mask = np.zeros((int(ndof / 2), 1))
    for elx in range(nelx):
        for ely in range(nely):
            if el_mask[elx, ely] == 0:
                continue
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely

            node_mask[n1] = 1
            node_mask[n2] = 1
            node_mask[n1 + 1] = 1
            node_mask[n2 + 1] = 1

    return node_mask


class GymCAD2dFEM:
    def __init__(self):
        self.max_index = 100
        self.state_size = 40
        self.rect_init_size = 4
        self.moving_step = 1
        self.force_module = 0.1
        self.la = 0.5769
        self.mu = 0.3846

        self.topology_treshold = 0.5

        # FIXME: add flag for diffrent types of mesh: static and dynamic
        self.reset()

    def reset(self):
        self.action = 0
        self.current_index = 0
        self.objective = 0
        self.obj_opt = 0

        self.x_points = [0, self.state_size]
        self.y_points = [0, self.state_size]

        self.current_rect = np.array([[0, 0], [0, 0]])
        self.previous_rects = []
        self.state_workspace = np.zeros((self.state_size, self.state_size))
        self.state_top_optimazed = np.zeros((self.state_size, self.state_size))
        self.state_top_optimazed_bin = np.zeros((self.state_size, self.state_size))
        self.state_current_rect = np.zeros((self.state_size, self.state_size)) 
        self.state_previous_rects = np.zeros((self.state_size, self.state_size))
        self.state_intersections = np.zeros((self.state_size, self.state_size))

        self.constraints = []
        self.forces = []
        self.force_places = []

        self.generate_task()
        self.generate_working_rect()

        ############################## init FEM problem (for static mesh case) Q4 elements #################################

        self.init_FEM()
        self.init_FEM_problem()
        self.optimization(60)

        #####################################################################################################################

        ############################# upload saved FEM problem ##############################################################

        # TO do

        #####################################################################################################################

        self.update_workspace()

        # FIXME: add normalization on onjective function
        # self.calculate_objective()

        return (self.state_workspace, {})

    def solve_via_SIMP(self):
        sK = (
            (self.KE.flatten()[np.newaxis]).T
            * (self.Emin + (self.xPhys) ** self.penal * (self.Emax - self.Emin))
        ).flatten(order="F")
        K = coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()

        # Remove constrained dofs from matrix
        K = K[self.free, :][:, self.free]

        # Solve system with super LU factorization
        K = K.tocsc()  #  Need CSR for SuperLU factorisation
        lu = sla.splu(K)
        self.u[self.free, 0] = lu.solve(self.f[self.free, 0])

        print(self.u.max())
        print(self.u.min())

        self.ce[:] = (
            np.dot(self.u[self.edofMat].reshape(self.nelx * self.nely, 8), self.KE)
            * self.u[self.edofMat].reshape(self.nelx * self.nely, 8)
        ).sum(1)
        local_obj = (
            (self.Emin + self.xPhys**self.penal * (self.Emax - self.Emin)) * self.ce
        ).sum()

        # convert Q4 element to triangular
        tri_mesh = []
        for i, elem in enumerate(self.conn):
            if i not in self.excluding_elements:
                tri_mesh.append([elem[0], elem[1], elem[2]])
                tri_mesh.append([elem[0], elem[3], elem[2]])

        disp_x = self.u[::2]
        disp_y = self.u[1::2]
        disp_module = np.sqrt(disp_x**2 + disp_y**2)

        q_w_disp_f = self.coords + self.u
        q_w_disp_x = q_w_disp_f[::2]
        q_w_disp_y = q_w_disp_f[1::2]

        q_w_disp = np.zeros((int(self.ndof / 2), 2))

        q_w_disp[:, 0] = q_w_disp_x.flatten().astype(q_w_disp.dtype)
        q_w_disp[:, 1] = q_w_disp_y.flatten().astype(q_w_disp.dtype)

        constraint_indeces = []
        for c_line in self.constraint_idx:
            by_1d_index = []
            for c_node in c_line:
                by_1d_index.append(int(c_node[0] / 2))

            constraint_indeces.append(by_1d_index)

        force_indeces = []
        for f_line in self.force_idx:
            by_1d_index = []
            for f_node in f_line:
                by_1d_index.append(int(f_node[0] / 2))

            force_indeces.append(by_1d_index)

        fig_1 = plt.figure(1)
        self.plot_mesh(
            tri_mesh,
            q_w_disp,
            constraint_indeces,
            force_indeces,
            disp_module,
            "Obj: " + str(round(local_obj, 3)),
        )

    def init_FEM_problem(self):
        self.dofs = np.arange(2 * (self.nelx + 1) * (self.nely + 1))

        fixed = []
        for c_line in self.constraint_idx:
            for c_node in c_line:
                fixed.append(c_node[0])
                fixed.append(c_node[1])

        self.fixed = np.array(fixed)
        self.free = np.setdiff1d(self.dofs, self.fixed)

        self.f = np.zeros((self.ndof, 1))
        self.u = np.zeros((self.ndof, 1))

        for i, f_line in enumerate(self.force_idx):
            av_force = self.force_module / len(f_line)
            for node in f_line:
                dir = self.forces[i][1]
                x_dof = node[0]
                y_dof = node[1]

                if dir == 0:
                    self.f[x_dof] = -av_force
                elif dir == 1:
                    self.f[y_dof] = av_force
                elif dir == 2:
                    self.f[x_dof] = av_force
                elif dir == 3:
                    self.f[y_dof] = -av_force

        # print(self.force_places)
        # print(self.coords.max())
        # print(self.coords.min())

    def init_FEM(self):
        self.penal = 3
        self.rmin = 5.4
        self.ft = 1
        nelx = self.state_size
        nely = self.state_size
        self.nelx = nelx
        self.nely = nely
        self.volfrac = 1
        self.cell_size = 1  # edge size for one element

        # Young's modulus and Poissnon's ratio
        E = 1
        v = 0.3

        # Max and min stiffness
        self.Emin = 1e-9
        self.Emax = 1.0

        # number of dofs:
        self.ndof = 2 * (nelx + 1) * (nely + 1)

        # Allocate design variables (as array), initialize and allocate sens.
        self.x = self.volfrac * np.ones(nely * nelx, dtype=float)
        self.xold = self.x.copy()
        self.xPhys = self.x.copy()
        self.xPhys_previous = self.x.copy()
        self.g = 0  # must be initialized to use the NGuyen/Paulino OC approach
        self.dc = np.zeros((nely, nelx), dtype=float)

        # FE: Build the index vectors for the for coo matrix format.
        self.KE = lk(E, v)
        self.edofMat = np.zeros((nelx * nely, 8), dtype=int)
        self.conn = (
            []
        )  # matrix of element (row corresponds to the element, column corresponds to the indeces of nodes)
        self.coords = np.zeros(
            (self.ndof, 1)
        )  # coords for each dof (even - X, odd - Y)

        self.constraint_idx = [[] for i in self.constraints]
        self.force_idx = [[] for i in self.force_places]

        for elx in range(nelx):
            for ely in range(nely):  # iterate each Q4 element
                el = ely + elx * nely
                n1 = (nely + 1) * elx + ely
                n2 = (nely + 1) * (elx + 1) + ely

                # save indeces of nodes for each element
                self.edofMat[el, :] = np.array(
                    [
                        2 * n1 + 2,
                        2 * n1 + 1,
                        2 * n2 + 2,
                        2 * n2 + 1,
                        2 * n2,
                        2 * n2 + 3,
                        2 * n1,
                        2 * n1 + 3,
                    ]
                )

                # save indeces of nodes for element
                self.conn.append([n2, n1, n1 + 1, n2 + 1])
                # if elx == 0 and ely == 0:
                # print([n2, n1, n1 + 1, n2 + 1])
                # print([n1, n2+1, n2, n1 + 1])
                #   print([2*n1+2, 2*n1+1, 2*n2+2, 2*n2+1, 2*n2, 2*n2+3, 2*n1, 2*n1+3])

                # center of the node
                x_center = self.cell_size / 2 + self.cell_size * elx
                y_center = -self.cell_size / 2 - self.cell_size * ely

                points_x_coords = [
                    x_center - self.cell_size / 2,
                    x_center + self.cell_size / 2,
                    x_center + self.cell_size / 2,
                    x_center - self.cell_size / 2,
                ]

                points_y_coords = [
                    y_center - self.cell_size / 2,
                    y_center - self.cell_size / 2,
                    y_center + self.cell_size / 2,
                    y_center + self.cell_size / 2,
                ]

                points_x_indeces = [2 * n1 + 2, 2 * n2 + 2, 2 * n2, 2 * n1]

                points_y_indeces = [2 * n1 + 3, 2 * n2 + 3, 2 * n2 + 1, 2 * n1 + 1]
                # points_y_indeces = [2*n1+3, 2*n2+3, 2*n2+1, 2*n1 + 1]

                # if elx == 0 and ely == 0:
                # print([n2, n1, n1 + 1, n2 + 1])
                # print([2*n1+2, 2*n1+1, 2*n2+2, 2*n2+1, 2*n2, 2*n2+3, 2*n1, 2*n1+3])
                # print("y_idx", points_y_indeces)
                #   print("y_crd", points_y_coords)
                #   print("x_idx", points_x_indeces)
                #   print("x_crd", points_x_coords)

                # save x coords of node
                # self.coords[2*n1+2] = x_center - self.cell_size/2
                # self.coords[2*n2+2] = x_center + self.cell_size/2
                # self.coords[2*n2] = x_center + self.cell_size/2
                # self.coords[2*n1] = x_center - self.cell_size/2

                # save y coords of node
                # self.coords[2*n1+3] = y_center - self.cell_size/2
                # self.coords[2*n2+3] = y_center - self.cell_size/2
                # self.coords[2*n2+1] = y_center + self.cell_size/2
                # self.coords[2*n1+1] = y_center + self.cell_size/2

                for i in range(4):
                    node = [points_x_coords[i], points_y_coords[i] * (-1)]
                    indeces = [points_x_indeces[i], points_y_indeces[i]]

                    c_check_pol, c_line_index = point_on_lines(node, self.constraints)
                    if c_check_pol:
                        if indeces not in self.constraint_idx[c_line_index]:
                            self.constraint_idx[c_line_index].append(indeces)

                    else:
                        f_check_pol, f_line_index = point_on_lines(
                            node, self.force_places
                        )
                        if f_check_pol:
                            if indeces not in self.force_idx[f_line_index]:
                                # print(node)
                                self.force_idx[f_line_index].append(indeces)

                    self.coords[indeces[0]] = node[0]  # save x index coord
                    self.coords[indeces[1]] = node[1]  # save y index coord

        # Construct the index pointers for the coo format
        self.iK = np.kron(self.edofMat, np.ones((8, 1))).flatten()
        self.jK = np.kron(self.edofMat, np.ones((1, 8))).flatten()

        # Filter: Build (and assemble) the index+data vectors for the coo matrix format
        self.nfilter = int(
            self.nelx * self.nely * ((2 * (np.ceil(self.rmin) - 1) + 1) ** 2)
        )
        self.iH = np.zeros(self.nfilter)
        self.jH = np.zeros(self.nfilter)
        self.sH = np.zeros(self.nfilter)
        cc = 0
        for i in range(self.nelx):
            for j in range(self.nely):
                row = i * self.nely + j
                kk1 = int(np.maximum(i - (np.ceil(self.rmin) - 1), 0))
                kk2 = int(np.minimum(i + np.ceil(self.rmin), nelx))
                ll1 = int(np.maximum(j - (np.ceil(self.rmin) - 1), 0))
                ll2 = int(np.minimum(j + np.ceil(self.rmin), self.nely))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * self.nely + l
                        fac = self.rmin - np.sqrt(
                            ((i - k) * (i - k) + (j - l) * (j - l))
                        )
                        self.iH[cc] = row
                        self.jH[cc] = col
                        self.sH[cc] = np.maximum(0.0, fac)
                        cc = cc + 1

        # Finalize assembly and convert to csc format
        self.H = coo_matrix(
            (self.sH, (self.iH, self.jH)),
            shape=(self.nelx * self.nely, self.nelx * self.nely),
        ).tocsc()
        self.Hs = self.H.sum(1)

        self.dv = np.ones(self.nely * self.nelx)
        self.dc = np.ones(self.nely * self.nelx)
        self.ce = np.ones(self.nely * self.nelx)

    def generate_task(self):
        self.generate_constraints()
        self.generate_force()

    def generate_constraints(self):
        # the fixation for x and y used (next will be slide)
        n_constraints = np.random.choice(np.array([1, 2]), 1, False)[0]
        constraints_counter = 0

        constraints = []

        # generate number of generate n of generate_constraints
        while constraints_counter < n_constraints:
            # genrate side
            side_idx = np.random.choice(np.arange(0, 4), 1, False)[0]

            # generate_length
            length = np.random.choice(
                np.arange(int(self.state_size / 5), self.state_size), 1, False
            )[0]

            # generate first point
            placement = np.random.choice(np.arange(self.state_size - length), 1, False)[
                0
            ]

            if side_idx == 0:
                first_point = [0, self.state_size - length - placement]
                second_point = [0, self.state_size - placement]

            elif side_idx == 1:
                first_point = [self.state_size - length - placement, self.state_size]
                second_point = [self.state_size - placement, self.state_size]
            elif side_idx == 2:
                first_point = [self.state_size, self.state_size - length - placement]
                second_point = [self.state_size, self.state_size - placement]

            elif side_idx == 3:
                first_point = [self.state_size - length - placement, 0]
                second_point = [self.state_size - placement, 0]

            line = [first_point, second_point]
            # add if there is no intersections
            if check_intersection(constraints, line):
                continue

            else:
                constraints.append(line)

                constraints_counter += 1

        # create constraints without intersections

        # collect constraints

        self.constraints = constraints

        # colect to x and y coords
        for line in constraints:
            self.x_points.append(line[0][0])
            self.x_points.append(line[1][0])
            self.y_points.append(line[0][1])
            self.y_points.append(line[1][1])

    def generate_force(self):
        # the fixation for x and y used (next will be slide)
        n_forces = np.random.choice(np.array([1, 2]), 1, False)[0]
        force_counter = 0

        force_places = []
        force_directions = []
        force_side_indeces = []

        # generate number of generate n of generate_constraints
        while force_counter < n_forces:
            # genrate side
            side_idx = np.random.choice(np.arange(0, 4), 1, False)[0]

            # generate_length
            length = np.random.choice(
                np.arange(int(self.state_size / 8), self.state_size), 1, False
            )[0]

            # generate first point
            placement = np.random.choice(np.arange(self.state_size - length), 1, False)[
                0
            ]

            if side_idx == 0:
                first_point = [0, self.state_size - length - placement]
                second_point = [0, self.state_size - placement]

            elif side_idx == 1:
                first_point = [self.state_size - length - placement, self.state_size]
                second_point = [self.state_size - placement, self.state_size]
            elif side_idx == 2:
                first_point = [self.state_size, self.state_size - length - placement]
                second_point = [self.state_size, self.state_size - placement]

            elif side_idx == 3:
                first_point = [self.state_size - length - placement, 0]
                second_point = [self.state_size - placement, 0]

            line = [first_point, second_point]
            # add if there is no intersections
            if check_intersection(force_places, line):
                continue

            else:
                if check_intersection(self.constraints, line):
                    continue
                else:
                    force_places.append(line)

                    direction = np.random.choice(np.arange(0, 4), 1, False)[0]
                    force_directions.append(direction)
                    force_side_indeces.append(side_idx)

                    force_counter += 1

        # create constraints without intersections

        # collect constraints

        forces = []
        for i, line in enumerate(force_places):
            forces.append([line, force_directions[i], force_side_indeces[i]])
            self.x_points.append(line[0][0])
            self.x_points.append(line[1][0])
            self.y_points.append(line[0][1])
            self.y_points.append(line[1][1])

        self.forces = forces
        self.force_places = force_places

    def generate_working_rect(self):
        self.current_rect[0, :] = int(
            self.state_size / 2 - self.rect_init_size / 2
        )  # first point
        self.current_rect[1, :] = int(
            self.state_size / 2 + self.rect_init_size / 2
        )  # second point

        self.update_current_state()

    def transform_current_rect(self, action, moving_step = 1):
        action_dicoding = [
            [0, 0, -1],  # 1
            [0, 0, 1],  # 2
            [1, 0, 1],  # 3
            [1, 0, -1],  # 4
            [1, 1, 1],  # 5
            [1, 1, -1],  # 6
            [0, 1, -1],  # 7
            [0, 1, 1],
        ]  # 8

        parsed_action = action_dicoding[action]

        current_rect_copy = self.current_rect.copy()
        current_rect_copy[parsed_action[0], parsed_action[1]] += (
            parsed_action[2] * moving_step
        )

        d_x = current_rect_copy[1, 0] - current_rect_copy[0, 0]
        d_y = current_rect_copy[1, 1] - current_rect_copy[0, 1]

        if d_x > 0 and d_y > 0:
            self.current_rect = current_rect_copy

            self.update_current_state()
            self.update_workspace()
            self.calculate_objective()

    def update_previous_rects(self):
        self.previous_rects.append(self.current_rect.copy())
        #self.x_points.append(self.current_rect[0][0])
        #self.x_points.append(self.current_rect[1][0])
        #self.y_points.append(self.current_rect[0][1])
        #self.y_points.append(self.current_rect[1][1])

        self.state_previous_rects += self.state_current_rect
        self.state_previous_rects = np.clip(self.state_previous_rects, 0, 1)
   
    def update_current_state(self):
        
        self.state_current_rect *= 0

        self.state_current_rect[
            self.current_rect[0, 1] : self.current_rect[1, 1],
            self.current_rect[0, 0] : self.current_rect[1, 0],
        ] = 1

        self.state_current_rect = np.flipud(self.state_current_rect)

    def update_workspace_fem(self):
        # preprocess mesh_points

        # self.x = self.volfrac * np.ones(nely * nelx, dtype=float)
        self.xPhys = np.ones(self.nely * self.nelx, dtype=float)

        excluding_elements = []

        for elx in range(self.nelx):
            for ely in range(self.nely):
                el_idx = ely + elx * self.nely

                x_center = self.cell_size / 2 + self.cell_size * elx
                y_center = self.cell_size / 2 + self.cell_size * ely

                # prepare excluding areas
                excluding_areas = self.previous_rects.copy()
                excluding_areas.append(self.current_rect)
                excluding_areas = np.array(excluding_areas)

                if in_rects([x_center, y_center], excluding_areas):
                    excluding_elements.append(el_idx)

        self.excluding_elements = np.array(excluding_elements)
        self.xPhys[self.excluding_elements] = 0

        self.solve_via_SIMP()
        plt.show()

    def update_workspace(self):
        
        # coeffs
        intersections_coeff = -1
        topology_coeff = 0.5
        previous_coeff = 0.5 
        current_coeff = 1.0

        # find all intersections
        # FIXME
        state_intersections_w_c = self.state_top_optimazed_bin + self.state_current_rect
        state_intersections_w_c = (state_intersections_w_c > 1) * 1 
        state_intersections_w_p = self.state_top_optimazed_bin + self.state_previous_rects
        state_intersections_w_p = (state_intersections_w_p > 1) * 1 

        state_intersections = state_intersections_w_c + state_intersections_w_p
        state_intersections = np.clip(state_intersections, 0, 1)

        self.state_intersections = state_intersections 

        # update state workspace
        self.state_workspace *= 0
        self.state_workspace[self.state_top_optimazed_bin == 1] = topology_coeff
        self.state_workspace[self.state_previous_rects == 1] = previous_coeff 
        self.state_workspace[self.state_current_rect == 1] = current_coeff
        self.state_workspace[state_intersections == 1] = intersections_coeff

    def calculate_objective(self):
        intersections_penalty = 10
        self.obj = self.state_previous_rects.sum() + self.state_current_rect.sum() - intersections_penalty * self.state_intersections.sum()

    def receive_action(self, action):
        self.action = action

    def step_side(self, side_idx, moving_step):

        """

        actions_options = [0, 1, 2, 3, 4] -> [-2, -1, 0, 1, 2]
        """
        if self.current_index > self.max_index:
            return self.state_workspace, 0, True, False, {}

        else:

            side_decode_list = [[1, 2], [3, 4], [5, 6], [7, 8]]
            action = action - 2

            if action < 0:
                transform_action = side_decode_list[side_idx][0]

            elif action > 0:
                transform_action

    def step(self, action):
        if self.current_index > self.max_index:
            return self.state_workspace, 0, True, False, {}

        else:
            self.receive_action(action)

            if self.action == 0:

                self.update_previous_rects()
                self.generate_working_rect()
                self.update_workspace()
                self.calculate_objective()

            else:
                self.transform_current_rect(self.action - 1)

                # self.update_workspace()
                # self.calculate_objective()

            self.current_index += 1

            # reward politics
            reward = None  # FIXME

            return self.state_workspace, reward, False, False, {}

    def optimization_step(self):
        """
        The function excute the optimization step.
        """

        # Setup and solve FE problem
        sK = (
            (self.KE.flatten()[np.newaxis]).T
            * (self.Emin + (self.xPhys) ** self.penal * (self.Emax - self.Emin))
        ).flatten(order="F")
        K = coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()

        # Remove constrained dofs from matrix
        K = K[self.free, :][:, self.free]

        # Solve system with super LU factorization
        K = K.tocsr()  #  Need CSR for SuperLU factorisation
        lu = sla.splu(K)
        self.u[self.free, 0] = lu.solve(self.f[self.free, 0])

        # Objective and sensitivity
        self.ce[:] = (
            np.dot(self.u[self.edofMat].reshape(self.nelx * self.nely, 8), self.KE)
            * self.u[self.edofMat].reshape(self.nelx * self.nely, 8)
        ).sum(1)
        self.obj = (
            (self.Emin + self.xPhys**self.penal * (self.Emax - self.Emin)) * self.ce
        ).sum()
        self.dc[:] = (
            -self.penal * self.xPhys ** (self.penal - 1) * (self.Emax - self.Emin)
        ) * self.ce
        self.dv[:] = np.ones(self.nely * self.nelx)

        # Sensitivity filtering:
        if self.ft == 0:  # Sensitivity based
            self.dc[:] = np.asarray(
                (self.H * (self.x * self.dc))[np.newaxis].T / self.Hs
            )[:, 0] / np.maximum(0.001, self.x)
        elif self.ft == 1:  # Density based
            self.dc[:] = np.asarray(self.H * (self.dc[np.newaxis].T / self.Hs))[:, 0]
            self.dv[:] = np.asarray(self.H * (self.dv[np.newaxis].T / self.Hs))[:, 0]

        # Optimality criteria
        self.xold[:] = self.x
        (self.x[:], self.g) = oc(
            self.nelx, self.nely, self.x, self.volfrac, self.dc, self.dv, self.g
        )

        # save the previous result
        self.xPhys_previous = self.xPhys.copy()

        # Filter design variables
        if self.ft == 0:
            self.xPhys[:] = self.x  # Sensitivity based
        elif self.ft == 1:
            self.xPhys[:] = np.asarray(self.H * self.x[np.newaxis].T / self.Hs)[
                :, 0
            ]  # Density based

        # Compute the change by the inf. norm
        self.change = np.linalg.norm(
            self.x.reshape(self.nelx * self.nely, 1)
            - self.xold.reshape(self.nelx * self.nely, 1),
            np.inf,
        )

    def optimization(self, n_iter, ploting=False, plotting_disp = True, fname = None):
        """
        The function excutes optimization iterations and saves history.
        Input:
            n_iter          - int, iteration number;
            trsh_chan       - float, when changes are small stop optimization;
            history_step    - int, increment for history savings (used to save memory)
        """

        self.volfrac = 0.4
        treshold = 0.5
        image_fname = "Top_optimization"
        if fname != None:
            image_fname = fname

        # Allocate design variables (as array), initialize and allocate sens.
        self.x = self.volfrac * np.ones(self.nely * self.nelx, dtype=float)
        self.xold = self.x.copy()
        self.xPhys = self.x.copy()
        self.xPhys_previous = self.x.copy()

        local_loop = 0
        while local_loop < n_iter:  # self.change>trsh_chang and
            self.optimization_step()
            # print("Iteration N: ", local_loop, "\t optimizer: TopoSIMP \t Objective function value: ", round(self.obj_opt, 2))
            local_loop += 1


        self.state_top_optimazed = np.rot90(
            self.xPhys.copy().reshape((self.nelx, self.nely))
        )

        self.state_top_optimazed_bin = (self.state_top_optimazed > self.topology_treshold) * 1

        if ploting:
            self.opt_plotting(treshold, image_fname, plotting_disp)

    def upload_solution(self, xPhys, plotting=False, plotting_disp=False):
        # xPhsy shoul be flatten !!!!
        self.xPhys = xPhys

        sK = (
            (self.KE.flatten()[np.newaxis]).T
            * (self.Emin + (self.xPhys) ** self.penal * (self.Emax - self.Emin))
        ).flatten(order="F")
        K = coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()

        # Remove constrained dofs from matrix
        K = K[self.free, :][:, self.free]

        # Solve system with super LU factorization
        K = K.tocsr()  #  Need CSR for SuperLU factorisation
        lu = sla.splu(K)
        self.u[self.free, 0] = lu.solve(self.f[self.free, 0])

        if plotting:
            self.opt_plotting(0.5, "Result_int.png", plotting_disp)
        

    def opt_plotting(self, treshold, image_fname, plotting_disp = True):

        self.ce[:] = (
            np.dot(self.u[self.edofMat].reshape(self.nelx * self.nely, 8), self.KE)
            * self.u[self.edofMat].reshape(self.nelx * self.nely, 8)
        ).sum(1)

        x_local = (self.xPhys.copy() > treshold) * 1
        local_obj = (
            (self.Emin + x_local**self.penal * (self.Emax - self.Emin)) * self.ce
        ).sum()

        el_mask = (self.xPhys.copy().reshape((self.nelx, self.nely)) > treshold) * 1
        # get void mask for nodes
        node_mask = el_mask_2_nod_mask(el_mask, self.nelx, self.nely, self.ndof)
        bool_mask = (node_mask > 0)[:, 0]

        indeces_all_nodes = np.arange(self.coords[::2].shape[0])
        masked_indeces = indeces_all_nodes[bool_mask]

        bool_mask = (node_mask > 0)[:, 0]

        indeces_all_nodes = np.arange(self.coords[::2].shape[0])
        masked_indeces = indeces_all_nodes[bool_mask]

        # convert Q4 element to triangular
        tri_mesh = []
        for i, elem in enumerate(self.conn):
            if np.any(
                np.isin(np.array(elem), masked_indeces)
            ):  # exclude void elements
                tri_mesh.append([elem[0], elem[1], elem[2]])
                tri_mesh.append([elem[0], elem[3], elem[2]])

        disp_x = self.u[::2]
        disp_y = self.u[1::2]
        disp_module = np.sqrt(disp_x**2 + disp_y**2)
        
        if plotting_disp:
            q_w_disp_f = self.coords + self.u
        else:
            q_w_disp_f = self.coords

        q_w_disp_x = q_w_disp_f[::2]
        q_w_disp_y = q_w_disp_f[1::2]

        q_w_disp = np.zeros((int(self.ndof / 2), 2))

        q_w_disp[:, 0] = q_w_disp_x.flatten().astype(q_w_disp.dtype)
        q_w_disp[:, 1] = q_w_disp_y.flatten().astype(q_w_disp.dtype)

        constraint_indeces = []
        for c_line in self.constraint_idx:
            by_1d_index = []
            for c_node in c_line:
                by_1d_index.append(int(c_node[0] / 2))

            constraint_indeces.append(by_1d_index)

        force_indeces = []
        for f_line in self.force_idx:
            by_1d_index = []
            for f_node in f_line:
                by_1d_index.append(int(f_node[0] / 2))

            force_indeces.append(by_1d_index)

        fig_1 = plt.figure(1)

        self.plot_mesh(
            tri_mesh,
            q_w_disp,
            constraint_indeces,
            force_indeces,
            disp_module,
            "Obj: " + str(round(local_obj, 3)),
        )

        plt.savefig(image_fname)
        plt.show()

    def plot_mesh(
        self,
        me,
        q,
        constraint_indeces,
        force_indeces,
        disp_module=None,
        title="No title",
    ):
        stop_counter = 0
        offset = 10

        plt.axis("equal")
        plt.title(title)
        plt.ylim([-offset, self.state_size + offset])
        plt.xlim([-offset, self.state_size + offset])

        for element in me:
            if disp_module is None:
                node_1_q = q[element[0]]
                node_2_q = q[element[1]]
                node_3_q = q[element[2]]

                m_color = "black"
                width = 0.2

                plt.plot(
                    [node_1_q[0], node_2_q[0]],
                    [node_1_q[1], node_2_q[1]],
                    linewidth=width,
                    c=m_color,
                    marker="o",
                )
                plt.plot(
                    [node_2_q[0], node_3_q[0]],
                    [node_2_q[1], node_3_q[1]],
                    linewidth=width,
                    c=m_color,
                    marker="o",
                )
                plt.plot(
                    [node_3_q[0], node_1_q[0]],
                    [node_3_q[1], node_1_q[1]],
                    linewidth=width,
                    c=m_color,
                    marker="o",
                )

            else:
                nodes_q = q[element]
                nodes_x = nodes_q[:, 0]
                nodes_y = nodes_q[:, 1]
                c = disp_module[element].flatten()
                cmap = truncate_colormap(plt.get_cmap("jet"), c.min(), c.max())
                try:
                    plt.tripcolor(nodes_x, nodes_y, c, cmap=cmap, shading="gouraud")
                except:
                    print("____________")
                    print(nodes_x)
                    print(nodes_y)
                    print(c)
                plt.plot(nodes_x, nodes_y, "k-", linewidth=0.5)

            stop_counter += 1

        for constraint in constraint_indeces:
            x = np.array(q)[constraint][:, 0]
            y = np.array(q)[constraint][:, 1]

            plt.plot(x, y, color="red", marker="o")

        for i, force in enumerate(force_indeces):
            force_decode = [[0, 0, -3, 0], [0, 0, 0, 3], [0, 0, 3, 0], [0, 0, 0, -3]]
            plt.plot(
                np.array(q)[force][:, 0],
                np.array(q)[force][:, 1],
                color="blue",
                marker="o",
            )

            force_dir = self.forces[i][1]
            force_arrow = np.array(force_decode)[force_dir]

            if self.forces[i][2] == 0:
                x_center = -1
                y_center = (self.forces[i][0][1][1] - self.forces[i][0][0][1]) / 2
                y_center += self.forces[i][0][0][1]

            elif self.forces[i][2] == 1:
                y_center = self.state_size + 1
                x_center = (self.forces[i][0][1][0] - self.forces[i][0][0][0]) / 2
                x_center += self.forces[i][0][0][0]

            elif self.forces[i][2] == 2:
                x_center = self.state_size + 1
                y_center = (self.forces[i][0][1][1] - self.forces[i][0][0][1]) / 2
                y_center += self.forces[i][0][0][1]

            elif self.forces[i][2] == 3:
                y_center = -1
                x_center = (self.forces[i][0][1][0] - self.forces[i][0][0][0]) / 2
                x_center += self.forces[i][0][0][0]

            force_arrow[0] += x_center
            # force_arrow[2] += x_center
            force_arrow[1] += y_center
            # force_arrow[3] += y_center

            plt.arrow(
                force_arrow[0],
                force_arrow[1],
                force_arrow[2],
                force_arrow[3],
                width=0.1,
                head_width=1,
                head_length=2,
                color="blue",
            )

        # plt.show()


if __name__ == "__main__":
    env = GymCAD2dFEM()
