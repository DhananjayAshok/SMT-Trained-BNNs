from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
import numpy as np


def int_array(x):
    return x


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
        n_datapoints, m, n = len(xws), len(xws[0]), len(xws[0][0])
        for t in range(n_datapoints):
            for j in range(m):
                ws_j = ws[j]
                xws_j = xws[t][j]
                for i in range(n):
                    c = CNFBuilder.xnor_link(xs[t][i], ws_j[i], xws_j[i])
                    clauses.extend(c)
        return clauses

    @staticmethod
    def create_linear_layer(in_features, out_features, n_datapoints=1, ws=[], xs=[], id_pool=None, layer_id="0",
                            do_linking=True, do_activation=True):
        """
        Returns lists xs, ws, xws which contain the integer IDs of the respective variables in the SAT problem.
        ws is a nested list len(ws) = out_features, len(ws[j]) = in_features for all j
        xws is shape where n_datapoints, out_features, in_features

        returns a list of clauses which encodes xws = Wx if do_linking is True.

        Typically the next layer h is such that h[j] = 1 <=> sum of all xws[j] > constant

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
                                                                    C=(in_features+1)//2)
                    h_list.append(h)
                    clauses.extend(h_clauses)
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
        if val_type == "x":
            for i in range(len(inp_array)):
                for j in range(len(inp_array[0])):
                    if inp_array[i, j] == 0 or not inp_array[i][j]:
                        clauses.append([-inp_list[i][j]])
                    else:
                        clauses.append([inp_list[i][j]])

        if val_type == "w":
            pass
        if val_type == "o":
            for i in range(len(inp_array)):
                if (hasattr(inp_array, 'shape') and len(inp_array.shape) == 1) or not hasattr(inp_array, 'shape'):
                    if inp_array[i] == 0 or not inp_array[i]:
                        clauses.append([-inp_list[i]])
                    else:
                        clauses.append([inp_list[i]])
                else:
                    for j in range(len(inp_array[0])):
                        if inp_array[i] == 0 or not inp_array[i]:
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


