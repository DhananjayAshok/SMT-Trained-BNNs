from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
import numpy as np


def int_array(x):
    return x


def shape(x):
    """
    Can check shapes of arrays and nested lists
    """
    if hasattr(x, 'shape'):
        return x.shape
    if type(x) != list:
        return tuple()
    else:
        if len(x) == 0:
            return tuple()
        return (len(x),) + shape(x[0])


def is_false(item):
    return item == 0 or item == -1 or item is False


class CNFBuilder:
    @staticmethod
    def get_IDPool():
        return IDPool()

    @staticmethod
    def implies(x, y):
        return [-x, y]

    @staticmethod
    def iff(x, y):
        return CNFBuilder.implies(x, y), CNFBuilder.implies(y, x)

    @staticmethod
    def aiffboc(a, b, c):
        """
        Returns clauses for A <=> B | C
        """
        aiboc = [-a, b, c]
        bocia0 = [-b, a]
        bocia1 = [-c, a]
        return aiboc, bocia0, bocia1

    @staticmethod
    def aiffbacod(a, b, c, d):
        """
        Returns clauses for A <=> (B and C) or D
        """
        aibacod0 = [-a, b, d]
        aibacod1 = [-a, c, d]
        bacodia0 = [-b, -c, a]
        bacodia1 = [-d, a]
        return aibacod0, aibacod1, bacodia0, bacodia1

    @staticmethod
    def sequential_counter(var_list, vpool=None, prefix="", C=None):
        """
        var_list is a list of integers as per pysat CNF
        vpool is an IDPool with all variables stored till now, or a fresh one is created
        prefix is a string identifier used to distinguish between multiple scs in the same CNF
        If C is none we assume C = len(var_list) = m
        r[m, C] is True iff at least C elements of the input sequence is true.
        """
        clauses = []
        if vpool is None:
            vpool = IDPool()
        r = {}
        m = len(var_list)
        if C is None:
            C = m
        for i in range(1, m + 1):
            for j in range(1, C + 1):
                r[(i, j)] = vpool.id(f"{prefix}_r{i, j}")
        if m == 0 or C <= 0:
            raise ValueError(f"Yo What")
        clauses.extend(CNFBuilder.iff(var_list[0], r[1, 1]))
        for j in range(2, C + 1):
            clauses.append([-r[1, j]])
        for i in range(2, m + 1):
            clauses.extend(CNFBuilder.aiffboc(r[i, 1], var_list[i - 1], r[i - 1, 1]))
        for i in range(2, m + 1):
            for j in range(2, C + 1):
                clauses.extend(CNFBuilder.aiffbacod(r[i, j], var_list[i - 1], r[i - 1, j - 1], r[i - 1, j]))
        return clauses, vpool, r[m, C]

    @staticmethod
    def xnor_link(a, b, c):
        """
        Returns clauses to force (a XNOR b) <=> c
        """
        return [a, -b, -c], [-a, b, -c], [a, b, c], [-a, -b, c]

    @staticmethod
    def linear_product_link(xs, ws, xws):
        """
        n input dimensionality, m output dimensionality
        xs: nested list shape [n_datapoints, n]
        ws: nested list shape [m, n]
        xws: nested list shape [n_datapoints, m, n] (technically XW^T presum as sum is done during activation h)
        the activation of h[t, i] will be 1 iff sum of all (over i) xws[t, i] > constant cutoff
        """
        clauses = []
        n_datapoints, m, n = shape(xws)
        for t in range(n_datapoints):
            for j in range(m):
                ws_j = ws[j]
                xws_j = xws[t][j]
                for i in range(n):
                    c = CNFBuilder.xnor_link(xs[t][i], ws_j[i], xws_j[i])
                    clauses.extend(c)
        return clauses

    @staticmethod
    def conv_product_link(xs, ws, xws):
        clauses = []
        n_datapoints, out_channels, new_h, new_w, in_channels, kernel_size, _ = shape(xws)
        for d in range(n_datapoints):
            for j in range(out_channels):
                for h in range(new_h):
                    for w in range(new_w):
                        s = []
                        for c in range(in_channels):
                            sl = xs[d][c][w:w + kernel_size][h:h + kernel_size]
                            for k_i in range(kernel_size):
                                for k_j in range(kernel_size):
                                    cl = CNFBuilder.xnor_link(sl[k_i][k_j], ws[j][c][k_i][k_j],
                                                              xws[j][new_h][new_w][c][k_i][k_j])
                                    clauses.extend(cl)
        return clauses

    @staticmethod
    def create_linear_layer(in_features, out_features, n_datapoints=1, ws=[], xs=[], id_pool=None, layer_id="0",
                            do_linking=True, do_activation=True):
        """
        Returns lists xs, ws, xws which contain the integer IDs of the respective variables in the SAT problem.
        ws is a nested list len(ws) = out_features, len(ws[j]) = in_features for all j
        xws is shape where n_datapoints, out_features, in_features

        returns a list of clauses which encodes xws = Wx if do_linking is True.

        Typically the next layer h is such that h[j] = 1 <=> sum of all xws[j] > 0
            (Hence C is set to be more than half of the number of input bits)

        """
        xws = []
        clauses = []
        if id_pool is None:
            id_pool = CNFBuilder.get_IDPool()

        if xs == []:
            for d in range(n_datapoints):
                x_list = []
                for i in range(in_features):
                    x = id_pool.id(f"X{d, i}")
                    x_list.append(x)
                xs.append(x_list)
        xs = int_array(xs)

        if ws == []:
            for j in range(out_features):
                w_list = []
                for iii in range(in_features):
                    w = id_pool.id(f"Layer[{layer_id}]|w{j, iii}")
                    w_list.append(w)
                ws.append(w_list)
        ws = int_array(ws)

        for d in range(n_datapoints):
            xw_d = []
            for j in range(out_features):
                xw_j = []
                for i in range(in_features):
                    xw = id_pool.id(f"Layer[{layer_id}]|xw{d, j, i}")
                    xw_j.append(xw)
                xw_d.append(xw_j)
            xws.append(xw_d)
        xws = int_array(xws)

        if do_linking:
            clauses.extend(CNFBuilder.linear_product_link(xs, ws, xws))

        if do_activation:
            hs = []
            for d in range(n_datapoints):
                h_list = []
                for j in range(out_features):
                    h_clauses, _, h = CNFBuilder.sequential_counter(xws[d][j], vpool=id_pool,
                                                                    prefix=f"h[{layer_id}]{d, j}",
                                                                    C=(in_features // 2) + 1)
                    h_list.append(h)
                    clauses.extend(h_clauses)
                hs.append(h_list)
            hs = int_array(hs)
            return xs, ws, xws, hs, clauses
        else:
            return xs, ws, xws, clauses

    @staticmethod
    def create_conv_layer(in_channels, out_channels, kernel_size=2, n_datapoints=1, ws=[], xs=[],
                          in_height=None, in_width=None, id_pool=None, layer_id="0",
                          do_linking=True, do_activation=True):
        """
        Returns lists xs, ws, xws which contain the integer IDs of the respective variables in the SAT problem.
        xs is a nested list of shape: (n_datapoints, in_channels, in_height, in_width).
            If xs is empty the in_height and in_width must be specified
        ws is a nested list of shape: (out_channels, in_channels, kernel_size, kernel_size)
        xws is shape where (n_datapoints, out_channels, new_h, new_w, in_channels, kernel_size, kernel_size)
            where new_h and new_w are given by the height width kernel size relationship for Conv Nets (H'=H-K+1)

        returns a list of clauses which encodes xws = conv(W, x) iff do_linking is True.

        Typically, the next layer h is such that h[i, j, k] = 1 <=> sum of all xws[i, j, k] > 0
            (Hence C is set to be more than half of in_channels * kernel_size^2 )

        """
        if xs == []:
            assert in_height is not None and in_width is not None

        clauses = []
        if id_pool is None:
            id_pool = CNFBuilder.get_IDPool()

        if xs == []:
            for d in range(n_datapoints):
                x_list = []
                for i in range(in_channels):
                    x_list1 = []
                    for he in range(in_height):
                        x_list2 = []
                        for wi in range(in_width):
                            x = id_pool.id(f"X{d, i, he, wi}")
                            x_list2.append(x)
                        x_list1.append(x_list2)
                    x_list.append(x_list1)
                xs.append(x_list)
        xs = int_array(xs)

        if ws == []:
            for j in range(out_channels):
                w_list = []
                for iii in range(in_channels):
                    w_list1 = []
                    for k in range(kernel_size):
                        w_list2 = []
                        for kk in range(kernel_size):
                            w = id_pool.id(f"Layer[{layer_id}]|w{j, iii, k, kk}")
                            w_list2.append(w)
                        w_list1.append(w_list2)
                    w_list.append(w_list1)
                ws.append(w_list)
        ws = int_array(ws)

        xws = []
        new_h = in_height - kernel_size + 1
        new_w = in_width - kernel_size + 1
        for d in range(n_datapoints):
            xw_d = []
            for j in range(out_channels):
                xw_j = []
                for he in range(new_h):
                    xw_he = []
                    for wi in range(new_w):
                        xw_wi = []
                        for i in range(in_channels):
                            xw_i = []
                            for k in range(kernel_size):
                                xw_k = []
                                for kk in range(kernel_size):
                                    xw = id_pool.id(f"Layer[{layer_id}]|xw{d, j, new_h, new_w, i, k, kk}")
                                    xw_k.append(xw)
                                xw_i.append(xw_k)
                            xw_wi.append(xw_i)
                        xw_he.append(xw_wi)
                    xw_j.append(xw_he)
                xw_d.append(xw_j)
            xws.append(xw_d)
        xws = int_array(xws)

        if do_linking:
            clauses.extend(CNFBuilder.conv_product_link(xs, ws, xws))

        if do_activation:
            hs = []
            for d in range(n_datapoints):
                h_list = []
                for j in range(out_channels):
                    h_list1 = []
                    for he in range(new_h):
                        h_list2 = []
                        for wi in range(new_w):
                            in_features = in_channels * (kernel_size**2)
                            h_clauses, _, h = CNFBuilder.sequential_counter(xws[d][j][he][wi], vpool=id_pool,
                                                                    prefix=f"h[{layer_id}]{d, j, he, wi}",
                                                                    C=(in_features // 2) + 1)
                            clauses.extend(h_clauses)
                            h_list2.append(h)
                        h_list1.append(h_list2)
                    h_list.append(h_list1)
                hs.append(h_list)
            hs = int_array(hs)
            return xs, ws, xws, hs, clauses
        else:
            return xs, ws, xws, clauses

    @staticmethod
    def assign_values(inp_array, inp_list, val_type=""):
        """
        Returns the constraints for assigning every element of inp_array to the corresponding inp_list values.
        """
        clauses = []
        if val_type == "x":  # shape (n_datapoints, dim)
            for i in range(len(inp_array)):
                for j in range(len(inp_array[0])):
                    if is_false(inp_array[i, j]):
                        clauses.append([-inp_list[i][j]])
                    else:
                        clauses.append([inp_list[i][j]])

        if val_type == "img_x":  # shape (n_datapoints, channels, H, W)
            n_datapoints, channels, H, W = shape(inp_array)
            for i in range(n_datapoints):
                for c in range(channels):
                    for h in range(H):
                        for w in range(W):
                            if is_false(inp_array[i, c, h, w]):
                                clauses.append([-inp_list[i][c][h][w]])
                            else:
                                clauses.append([inp_list[i][c][h][w]])

        if val_type == "w":  # Shape (output_dim, input_dim)
            for m in range(len(inp_array)):
                for n in range(len(inp_array[0])):
                    if is_false(inp_array[m][n]):
                        clauses.append([-inp_list[m][n]])
                    else:
                        clauses.append([inp_list[m][n]])

        if val_type == "o":  # Shape (n_datapoints,) or (n_datapoints, 1)
            for i in range(len(inp_array)):
                one_dim = len(inp_array[0]) == 1
                if one_dim:
                    if is_false(inp_array[i]):
                        clauses.append([-inp_list[i][0]])
                    else:
                        clauses.append([inp_list[i][0]])
                else:
                    for j in range(len(inp_array[0])):
                        if is_false(inp_array[i][j]):
                            clauses.append([-inp_list[i][j]])
                        else:
                            clauses.append([inp_list[i][j]])
        return clauses


class CNFDebugger:
    @staticmethod
    def cnf_int_to_var_name(var, vpool):
        item = vpool.obj(abs(var))
        return item

    @staticmethod
    def print_cnf_vars(cnf=None, id_pool=None, skip=None):
        assert (cnf is not None and hasattr(cnf, 'vpool')) or id_pool is not None
        vpool = None
        if cnf is None:
            vpool = id_pool
        else:
            vpool = cnf.vpool
        vars = []
        for inp in range(1, vpool.top):
            if skip is not None and inp in skip:
                continue
            name = CNFDebugger.cnf_int_to_var_name(inp, vpool)
            if skip is not None and name in skip:
                continue
            if name is not None:
                vars.append(f"{inp, name}")
        print(vars)
        return vars

    @staticmethod
    def clause_translate(clauses, vpool):
        translated = []
        for clause in clauses:
            terms = []
            for term in clause:
                name = CNFDebugger.cnf_int_to_var_name(term, vpool)
                if name is None:
                    name = "?"
                if term < 0:
                    name = f"~{name}"
                terms.append(name)
            translated.append(terms)
        return translated

    @staticmethod
    def get_solver_model(solver, cnf):
        status = solver.get_status()
        if not status:
            print(f"Model has no solution")
            return
        model = solver.get_model()
        m = {}
        for var in model:
            name = CNFDebugger.cnf_int_to_var_name(var, cnf.vpool)
            if name is not None:
                m[name] = var > 0
        return m
