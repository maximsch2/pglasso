#cython: cdivision=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: infer_types=True
#cython: profile=False
#cython: embed_signature=True

import numpy as np
cimport numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import itertools
import gc

# an object that will represent a pathway, it would hold its own part of the theta matrix as well as connectivity information

include "dpshift.pyx"


cdef class Message:
    cdef double[:, ::1] data
    cdef long[::1] idx
    def __init__(self, double[:, ::1] data, long[::1] idx):
        self.data = data
        self.idx = idx

cdef class DoubleMsg:
    cdef int msg_from
    cdef Message msg1, msg2
    def __init__(self, int fr, Message m1, Message m2):
        self.msg_from = fr
        self.msg1 = m1
        self.msg2 = m2


@cython.profile(False)
cdef inline bint is_sorted(long[:] arr):
    # enable this function wheen debugging
    return True
    cdef int i
    for i in xrange(arr.shape[0] - 1):
        if arr[i] > arr[i+1]:
            return False
    return True


@cython.profile(False)
cdef inline bint construct_mapping(long[::1] map_from, long[::1] map_to, long[::1] res):
    cdef int i, j, n, n2

    assert map_from.shape[0] == res.shape[0]
    assert is_sorted(map_from)
    assert is_sorted(map_to)
    cdef bint result = False

    n = map_from.shape[0]
    n2 = map_to.shape[0]

    # quickly exit if there is nothing to do here
    if (n == 0) or (n2 == 0) or (map_from[0] > map_to[n2-1]) or (map_to[0] > map_from[n-1]):
        res[:] = -1
        return False

    i = 0
    j = 0
    while (i < n) and (j < n2):
        if (i < n) and (j < n2) and (map_to[j] == map_from[i]):
            res[i] = j
            result = True
            i += 1
            j += 1

        if j < n2:
            while (i < n) and (map_to[j] > map_from[i]):
                res[i] = -1
                i += 1

        if i < n:
            while (j < n2) and (map_to[j] < map_from[i]):
                j += 1


    while i < n:
        res[i] = -1
        i += 1

    return result



cpdef do_dot_np(a, b):
    return np.dot(a, b)

cpdef do_dot(double[:, :] a, double[:, :] b):
    assert a.shape[1] == b.shape[0]
    assert a.shape[0] == b.shape[1]
    cdef int i, j, k, n1, n2
    cdef double val
    n1 = a.shape[0]
    n2 = a.shape[1]
    cdef double[:, ::1] result = np.zeros((a.shape[0], a.shape[0]))
    for i in xrange(n1):
        for j in xrange(i, n1):
            val = 0
            for k in xrange(n2):
                val += a[i, k]*b[k, j]
            result[i, j] = val
            result[j, i] = val

    return result

cdef class CSimplePathwayGeneral(object):
    cdef public double[:, ::1] Theta, shift, S, W, Told, U, my_slice
    cdef public long[::1] pathway, slice_nonzero, vars_responsible, vars_other, local_resp, vars_resp_idx, vars_other_idx, vars_other_nnz, vars_resp_nnz, neigb_processed
    cdef public int p, nvars, pathway_id
    cdef dict neighbours, precomputed_params
    cdef bint debug
    cdef public list nieb_intersections
    cdef DoubleMsg precomputed_message

    def __init__(self, S, int pathway_id, pathway, neighbours, int n_pathways, Theta = None, debug=False):
        self.pathway_id = pathway_id
        self.pathway = np.array(pathway).copy()
        assert is_sorted(self.pathway)

        self.p = S.shape[0]
        self.nvars = len(self.pathway)
        self.neighbours = neighbours # a dictionary: {neibid : list of variables}
        if Theta is None:
            self.Theta = np.eye(self.nvars)
        else:
            self.Theta = Theta[np.ix_(self.pathway, self.pathway)].copy()
        self.Told = self.Theta.copy()
        self.shift = np.zeros((self.nvars, self.nvars))
        self.S = S[np.ix_(self.pathway, self.pathway)].copy()
        self.W = self.S.copy()
        self.U = np.zeros_like(self.S)
        self.debug = debug

        self.neigb_processed = np.zeros(n_pathways, dtype=np.int64)
        self.precomputed_message = None


    cpdef clear_messages(self):
        self.shift[:, :] = 0
        self.neigb_processed[:] = 0
        self.precomputed_message = None


    cpdef set_resposible_for(self, long[::1] vars_responsible, long[::1] vars_other):
        cdef int i, j, no, nr, var, k
        assert is_sorted(vars_responsible)
        assert is_sorted(vars_other)
        nr = vars_responsible.shape[0]
        no = vars_other.shape[0]
        if no > 0:
            self.my_slice = np.zeros((nr, no), dtype=np.float64, order='C')
        else:
            self.my_slice = None
        self.slice_nonzero = np.zeros(no+1, dtype=np.int64)
        self.vars_responsible = vars_responsible
        self.vars_other = vars_other
        self.vars_resp_idx = np.zeros(nr, dtype=np.int64)
        self.vars_resp_nnz = np.zeros(nr, dtype=np.int64)
        self.vars_other_idx = np.zeros(no, dtype=np.int64)
        self.vars_other_nnz = np.zeros(no, dtype=np.int64)


        cdef long[::1] vars_other_idx = np.zeros(vars_other.shape[0], dtype=np.int64)
        construct_mapping(vars_other, self.pathway, vars_other_idx)

        self.local_resp = np.zeros(vars_responsible.shape[0], dtype=np.int64)
        construct_mapping(vars_responsible, self.pathway, self.local_resp)

        for i in xrange(no):
            j = vars_other_idx[i]
            if j >= 0:
                self.slice_nonzero[i] = 1
                for k in xrange(nr):
                    self.my_slice[k, i] = self.Theta[self.local_resp[k], j]

    cdef fix_msg2_new2(self, np.ndarray[np.float64_t, ndim=2] outside, long[::1] vars_outside):
        cdef int i, j, n, mi, mj, neib_id
        cdef long[::1] intersection, mapping
        cdef np.ndarray[np.int64_t, ndim=1] neighbour

        for neib_id, neibval in self.neighbours.viewitems():
            #if neib_id in self.neigb_processed:
            if self.neigb_processed[neib_id] == 1:
                continue
            neighbour = neibval
            n = neighbour.shape[0]
            mapping = np.zeros(n, dtype=np.int64)
            if not construct_mapping(neighbour, vars_outside, mapping):
                continue
            for i in xrange(n):
                mi = mapping[i]
                if mi < 0:
                    continue
                outside[mi, mi] = 0
                for j in xrange(i+1, n):
                    mj = mapping[j]
                    if mj < 0:
                        continue
                    outside[mi, mj] = 0
                    outside[mj, mi] = 0

        cdef bint keep = False
        cdef long remove = 0
        for i in xrange(vars_outside.shape[0]):
            keep = False
            for j in xrange(vars_outside.shape[0]):
                if outside[i, j] != 0:
                    keep = True
                    break
            if keep:
               break
        return keep



    cdef construct_second_message(self):
        cdef np.ndarray[np.float64_t, ndim=2] outside
        cdef int i, j, n, ii, jj, n2
        cdef long[::1] local_outside, vars_outside
        cdef int n_outside = self.pathway.shape[0] - self.vars_responsible.shape[0]
        cdef bint keep = False
        n = self.pathway.shape[0]
        n2 = self.vars_responsible.shape[0]
        vars_outside = np.zeros(n_outside, dtype=np.int64)

        ii = 0
        for i in xrange(n):
            jj = self.pathway[i]
            keep = True
            for j in xrange(n2):
                if self.vars_responsible[j] == jj:
                    keep = False
                    break
            if keep:
                vars_outside[ii] = self.pathway[i]
                ii += 1
                if ii == n_outside:
                    break


        local_outside = np.zeros(n_outside, dtype=np.int64)
        #construct_mapping(vars_outside, self.vars_other, local_outside)
        construct_mapping(vars_outside, self.pathway, local_outside)



        outside = np.zeros((n_outside, n_outside), dtype=np.float64)
        for i in xrange(n_outside):
            ii = local_outside[i]
            for j in xrange(n_outside):
                jj = local_outside[j]
                outside[i, j] = self.Theta[ii, jj] - self.shift[ii, jj]



        assert is_sorted(vars_outside), "vars_outside is not sorted"

        if self.fix_msg2_new2(outside, vars_outside):
            return Message(outside, vars_outside)
        else:
            return None

    cdef marginalize1_get_data(self):
        cdef int i, j, k, n
        cdef np.ndarray[np.float64_t, ndim=2] matrix
        cdef long[::1] local_resp = self.local_resp
        n = local_resp.shape[0]
        matrix = np.zeros((n, n), dtype=np.float64)
        for i in xrange(n):
            for j in xrange(n):
                matrix[i, j] = self.Theta[local_resp[i], local_resp[j]] - self.shift[local_resp[i], local_resp[j]]


        return matrix

    cdef construct_first_message_new(self):
        cdef int i, j, k, ii, jj, nnz
        cdef double val
        cdef double[:, ::1] local_slice
        cdef np.ndarray[np.float64_t, ndim=2] arr1_reduced
        cdef long[::1] out_idx, local_idx


        local_slice = self.my_slice
        cdef int no = local_slice.shape[1]
        cdef int nr = local_slice.shape[0]

        if self.local_resp.shape[0] == 0:
            return None

        nnz = 0
        for i in xrange(no):
            if self.slice_nonzero[i] == 1:
                for k in xrange(nr):
                    if self.my_slice[k, i] != 0:
                        nnz += 1
                        break
                else:
                    self.slice_nonzero[i] = 0

        if self.debug:
            print "total nnz:", nnz, "out of", no

        if nnz == 0:
            return None

        out_idx = np.zeros(nnz, dtype=np.int64)
        local_idx = np.zeros(nnz, dtype=np.int64)

        j = 0
        for i in xrange(no):
            if self.slice_nonzero[i] == 1:
                out_idx[j] = self.vars_other[i]
                local_idx[j] = i
                j += 1

        if (nnz == 0) or (nr == 0):
            return None

        arr1_reduced = np.zeros((nr, nnz), dtype=np.float64)


        for i in xrange(nnz):
            ii = local_idx[i]
            for j in xrange(nr):
                arr1_reduced[j, i] = local_slice[j, ii]

        matrix = self.marginalize1_get_data()

        try:
            arr2 = la.solve(matrix, arr1_reduced)
            #arr2 = sla.solve(matrix, arr1_reduced, sym_pos=True)
        except Exception as e:
            print "solve failed", e
            print "pathway_id", self.pathway_id
            raise e


        if arr2.shape[0] + arr2.shape[1] < 100:
            result = do_dot(arr1_reduced.T, arr2) # faster implementation for small arrays
        else:
            result = do_dot_np(arr1_reduced.T, arr2)

        assert is_sorted(out_idx), "out_idx is not sorted"
        msg1 = Message(result, out_idx)
        return msg1


    cpdef marginalize(self):
        if self.precomputed_message is None:
            msg1 = self.construct_first_message_new()
            msg2 = self.construct_second_message()
            self.precomputed_message = DoubleMsg(self.pathway_id, msg1, msg2)

        return self.precomputed_message


    cdef process_msg1(self, Message msg, bint do_rollback):
        if msg is None:
            return

        cdef double[:, ::1] data
        cdef int nr, no, i, j, ii, jj, li, lj, i0
        cdef long[::1] vars_resp_idx, vars_other_idx, local_resp, idx
        cdef double val

        idx = msg.idx
        vars_resp_idx = self.vars_resp_idx

        # check this quickly
        if not construct_mapping(self.vars_responsible, idx, vars_resp_idx):
            return # no intersection

        data = msg.data


        nr = self.vars_responsible.shape[0]
        no = self.vars_other.shape[0]


        vars_other_idx = self.vars_other_idx
        local_resp = self.local_resp

        cdef long[::1] nonzero_resp = self.vars_resp_nnz
        cdef long nonzero_resp_count = 0
        for j in xrange(nr):
            if vars_resp_idx[j] >= 0:
                nonzero_resp[nonzero_resp_count] = j
                nonzero_resp_count  += 1

        # put stuff that intersects vars_resposible into shift
        cdef double[:, ::1] shift = self.shift
        if do_rollback:
            for i0 in xrange(nonzero_resp_count):
                i = nonzero_resp[i0]
                ii = vars_resp_idx[i]
                li = local_resp[i]
                shift[li, li] -= data[ii, ii]
                for j in xrange(i0+1, nonzero_resp_count):
                    j = nonzero_resp[j]
                    jj = vars_resp_idx[j]

                    lj = local_resp[j]
                    val = data[ii, jj]

                    shift[li, lj] -= val
                    shift[lj, li] -= val
        else:
            for i0 in xrange(nonzero_resp_count):
                i = nonzero_resp[i0]
                ii = vars_resp_idx[i]
                li = local_resp[i]
                shift[li, li] += data[ii, ii]
                for j in xrange(i0+1, nonzero_resp_count):
                    j = nonzero_resp[j]
                    jj = vars_resp_idx[j]

                    lj = local_resp[j]
                    val = data[ii, jj]

                    shift[li, lj] += val
                    shift[lj, li] += val


        # put the rest of that stuff into my_slice
        if vars_other_idx.shape[0] == 0:
            return


        if not construct_mapping(self.vars_other, idx, vars_other_idx):
            return

        cdef double[:, ::1] my_slice = self.my_slice
        cdef long[::1] nonzero_other
        cdef long nonzero_other_count = 0
        nonzero_other = self.vars_other_nnz
        if do_rollback:
            for i in xrange(nonzero_resp_count):
                i = nonzero_resp[i]
                ii = vars_resp_idx[i]
                for j in xrange(nonzero_other_count):
                    j =  nonzero_other[j]
                    jj = vars_other_idx[j]

                    val = data[ii, jj]
                    my_slice[i, j] += val
        else:
            for j in xrange(no):
                if vars_other_idx[j] >= 0:
                    self.slice_nonzero[j] = 1
                    nonzero_other[nonzero_other_count] = j
                    nonzero_other_count  += 1

            for i in xrange(nonzero_resp_count):
                i = nonzero_resp[i]
                ii = vars_resp_idx[i]
                for j in xrange(nonzero_other_count):
                    j =  nonzero_other[j]
                    jj = vars_other_idx[j]

                    val = data[ii, jj]
                    my_slice[i, j] -= val

    cdef process_msg2(self, Message msg, bint do_rollback):
        if msg is None:
            return

        cdef double[:, ::1] data
        cdef long[::1] idx, vars_resp_idx, vars_other_idx
        cdef int nr, no, i, j, ii, jj
        cdef double val

        idx = msg.idx

        # quick hacks to discard useless messages
        if not construct_mapping(self.vars_responsible, idx, self.vars_resp_idx):
            return

        vars_resp_idx = np.zeros(idx.shape[0], dtype=np.int64)
        if not construct_mapping(idx, self.vars_responsible, vars_resp_idx):
            return

        vars_other_idx = np.zeros(idx.shape[0], dtype=np.int64)
        if not construct_mapping(idx, self.vars_other, vars_other_idx):
            return

        # put data from msg2 in others
        # want data that is (resp, other) not (resp, resp)
        data = msg.data
        if do_rollback:
            for j in xrange(vars_other_idx.shape[0]):
                jj = vars_other_idx[j]
                if jj < 0:
                    continue

                for i in xrange(vars_resp_idx.shape[0]):
                    ii = vars_resp_idx[i]
                    if ii < 0:
                        continue

                    self.my_slice[ii, jj] -= data[j, i]
        else:
            for j in xrange(vars_other_idx.shape[0]):
                jj = vars_other_idx[j]
                if jj < 0:
                    continue
                self.slice_nonzero[jj] = 1

                for i in xrange(vars_resp_idx.shape[0]):
                    ii = vars_resp_idx[i]
                    if ii < 0:
                        continue

                    self.my_slice[ii, jj] += data[j, i]



    cpdef process_one_message(self, DoubleMsg m, bint only1 = False, bint do_rollback = False):
        #assert tpl[0] not in self.neigb_processed
        self.precomputed_message = None # we are being updated, will need to recompute message next time
        if not do_rollback:
            self.neigb_processed[m.msg_from] = 1
            #self.neigb_processed.append(m.msg_from)
        else:
            self.neigb_processed[m.msg_from] = 0
            #self.neigb_processed.remove(m.msg_from)

        self.process_msg1(m.msg1, do_rollback = do_rollback)
        if m.msg2 is not None and not only1:
            self.process_msg2(m.msg2, do_rollback = do_rollback)


    cpdef double local_update_params(self, double lmbda, double tol, component):
        cdef int i, j
        if self.debug:
            for kk in component:
                if kk != self.pathway_id:
                    pass
                    #assert kk in self.neigb_processed

        with nogil:
            for i in xrange(self.nvars):
                for j in xrange(self.nvars):
                    self.Told[i, j] = self.Theta[i, j]
                    self.Theta[i, j] = self.Theta[i, j] - self.shift[i, j]


        assert np.max(np.abs(self.shift)) < 1e3, np.max(np.abs(self.shift)) # some sanity check

        new_py_dpglasso_shift(self.S, lmbda, self.shift, X=self.Theta, invX=self.W, U=self.U, outer_tol=tol/2.5, outerMaxiter=75, check_nan=True)
        if self.debug:
            print np.array(self.shift)
            print np.array(self.Theta)
        cdef double err, tmp
        err = 0
        with nogil:
            for i in xrange(self.nvars):
                for j in xrange(self.nvars):
                    self.Theta[i, j] = self.Theta[i, j] + self.shift[i, j]
                    tmp = fabs(self.Told[i, j] - self.Theta[i, j])
                    if tmp > err:
                        err = tmp
        return err

    cpdef sync_from(self, CSimplePathwayGeneral other):
        # sync local data with neighbour
        cdef int n_local, i, j, ii, jj
        cdef double[:, ::1] local_Theta, other_Theta


        if other.pathway_id not in self.neighbours:
            return

        n_local = self.pathway.shape[0]
        cdef long[::1] mapping = np.zeros(n_local, dtype=np.int64)
        if not construct_mapping(self.pathway, other.pathway, mapping):
            return

        local_Theta = self.Theta
        other_Theta = other.Theta

        for i in xrange(n_local):
            ii = mapping[i]
            if ii < 0:
                continue
            for j in xrange(n_local):
                jj = mapping[j]
                if jj < 0:
                    continue
                local_Theta[i, j] = other_Theta[ii, jj]


cpdef np.ndarray to_array(set obj):
    cdef int i, n, val
    cdef np.ndarray[np.int64_t, ndim=1] result
    n = len(obj)
    result = np.zeros(n, dtype=np.int64)
    i = 0
    for val in obj:
        result[i] = val
        i += 1
    return result



import sys
import networkx as nx
import copy
import random

def construct_result(objs, p):
    res = np.eye(p)
    for obj in objs:
        if obj is not None:
            res[np.ix_(obj.pathway, obj.pathway)] = np.array(obj.Theta)
    return res

def setup_pathways(graph, pathways, S, Theta=None, debug=False, full_setup=False):
    n = len(pathways)
    objs = []
    pathways = map(lambda x: sorted(x), pathways)
    pathways = map(np.array, pathways)
    for i in xrange(n):
        cur_p = set(pathways[i])
        neibs = {}
        skip = False
        if len(cur_p) == 0:
            print "Empty pathway!"
            skip = True
        elif i in graph:
            for j in graph[i].keys():
                if j < len(pathways) and len(set(pathways[j]) & cur_p) > 0:
                    neibs[j] = pathways[j]
                    if set(pathways[j]) & cur_p == cur_p:
                        skip = True
                        assert False, ("Pathway %d is a complete subset of pathway %d" % (i, j))
                        break

        if full_setup:
            for j in xrange(n):
                if len(pathways[j]) > 0:
                    neibs[j] = pathways[j]
            print "setting up %d neigbs" % len(neibs)
        if skip:
            objs.append(None)
        else:
            new = CSimplePathwayGeneral(S, i, pathways[i], neibs, len(pathways), Theta=Theta, debug=debug)
            objs.append(new)
    return objs


cpdef get_pathway_graph(pathways):
    pathwaynames = range(len(pathways))
    pathway_graph = nx.Graph()
    for name in pathwaynames:
        pathway_graph.add_node(name)

    for i, name in enumerate(pathwaynames):
        for j, name2 in enumerate(pathwaynames):
            t = len(set(pathways[i]).intersection(pathways[j]))
            if (t > 0) and (i != j):
                pathway_graph.add_edge(name, name2, {"weight": -t})
    return pathway_graph


@cython.wraparound(True)
cpdef tuple compute_order_new(list objs, list order):
    all_vals = []
    responsible = [set(objs[order[-1]].pathway)]
    other = [set()]
    cur_set = set()
    for i in reversed(order):
        cur_set.update(objs[i].pathway)
        all_vals.append(copy.copy(cur_set))
        if len(all_vals) > 1:
            responsible.append(<set>all_vals[-1] - <set>all_vals[-2])
            other.append(all_vals[-1] - responsible[-1])
    return list(reversed(responsible)), list(reversed(other))

cpdef apply_order_new(list responsible, list other, list order, list objs):
    for i, k in enumerate(order):
        resp0 = to_array(responsible[i])
        resp0.sort()
        other0 = to_array(other[i])
        other0.sort()
        objs[k].set_resposible_for(resp0, other0)


cdef void compute_order_new2_and_apply(list objs, list order, int t):
    cdef long[::1] resp_vals, resp_counts, other_vals, other_counts, pathway, cur_resp, cur_other
    cdef int i, j, k, n, maxp, all_resp, ii, jj, kk, runsum
    cdef CSimplePathwayGeneral obj
    n = len(order)
    maxp = objs[0].p


    resp_vals = np.empty(maxp, dtype=np.int64)
    resp_counts = np.zeros(n, dtype=np.int64)
    resp_vals[:] = -1

    for i in xrange(n):
        obj = objs[<int>order[i]]
        pathway = obj.pathway
        for j in xrange(pathway.shape[0]):
            k = resp_vals[pathway[j]]
            if k >= 0:
                resp_counts[k] -= 1
            resp_vals[pathway[j]] = i
            resp_counts[i] += 1

    all_resp = 0

    for i in xrange(n):
        all_resp += resp_counts[i]



    runsum = 0
    for i in xrange(t):
        runsum += resp_counts[i]
        cur_resp = np.zeros(resp_counts[i], dtype=np.int64)
        cur_other = np.zeros(all_resp - runsum, dtype=np.int64)
        ii = 0
        jj = 0
        for j in xrange(maxp):
            k = resp_vals[j]
            if k == i:
                cur_resp[ii] = j
                ii += 1
            elif k > i:
                cur_other[jj] = j
                jj += 1

        #assert ii == cur_resp.shape[0], "%d %d" % (ii, cur_resp.shape[0])
        #assert jj == cur_other.shape[0], "%d %d" % (jj, cur_other.shape[0])
        obj = objs[<int>order[i]]
        obj.set_resposible_for(cur_resp, cur_other)





cpdef double do_one_pass_DC_inner_recursive2(list objs, list order, double lmbda, double tol, list component, int t, int msg):
        # setup:
        cdef int i, j
        compute_order_new2_and_apply(objs, order, t)
        cdef DoubleMsg obj_msg
        if len(order) > 16:
            order_set = set(order)
        else:
            order_set = order

        # clenup + get all the information from outside:
        for i in xrange(t):
            (<CSimplePathwayGeneral>objs[<int>order[i]]).clear_messages()

        for j in component:
            if j not in order_set:
                obj_msg = (<CSimplePathwayGeneral>objs[j]).marginalize()
                # TODO: ? is this message even relevant??

                for i in xrange(t):
                    (<CSimplePathwayGeneral>objs[<int>order[i]]).process_one_message(obj_msg)

        # marginalize:
        for i in xrange(t):
            obj_msg = (<CSimplePathwayGeneral>objs[<int>order[i]]).marginalize()
            for j in xrange(i+1, t):
                (<CSimplePathwayGeneral>objs[<int>order[j]]).process_one_message(obj_msg)

        return do_one_pass_DC_inner2(objs, order[t:], lmbda, tol, component, msg)

cpdef double do_one_pass_DC_inner2(list objs, list order, double lmbda, double tol, list component, int msg):
    cdef CSimplePathwayGeneral obj, obj2
    cdef int i, k
    cdef double tmp1, tmp2
    if len(order) == 1:
        # inner case, will gather all messages and update parameters
        k = order[0]
        obj = objs[k]
        obj.clear_messages()
        obj.set_resposible_for(obj.pathway, to_array(set()))
        for i in component:
            if i != k:
                obj2 = objs[i]
                obj.process_one_message(obj2.marginalize(), only1=True)
        if msg > 0:
            print ".",
            sys.stdout.flush()
        res = obj.local_update_params(lmbda, tol, component)
        for i in component:
            if i != k:
                obj2 = objs[i]
                obj2.sync_from(obj)
        return res

    else:
        k = len(order)
        t = k/2


        tmp1 = do_one_pass_DC_inner_recursive2(objs, order, lmbda, tol, component, t, msg)


        order = list(reversed(order))
        t = len(order) - t

        tmp2 = do_one_pass_DC_inner_recursive2(objs, order, lmbda, tol, component, t, msg)


        return tmp1 + tmp2

cpdef double do_one_pass_DC_inner_recursive(list objs, list order, double lmbda, double tol, list component, int t, int msg):
        # setup:
        cdef int i, j, N
        cdef list responsible, other
        responsible, other = compute_order_new(objs, order)
        apply_order_new(responsible, other, order[:t], objs)
        cdef CSimplePathwayGeneral obj, obj2
        cdef DoubleMsg obj_msg
        N = len(order)

        # clenup + get all the information from outside:
        for i in xrange(t):
            obj = objs[<int>order[i]]
            obj.clear_messages()

        for j in component:
            if j not in order:
                obj_msg = (<CSimplePathwayGeneral>objs[j]).marginalize()
                for i in order[:t]:
                    obj = objs[i]
                    obj.process_one_message(obj_msg)

        # marginalize:
        for i in xrange(t):
            obj_msg = objs[<int>order[i]].marginalize()
            for j in xrange(i+1, t):
                obj2 = objs[<int>order[j]]
                obj2.process_one_message(obj_msg)

        cdef double result = do_one_pass_DC_inner(objs, order[t:], lmbda, tol, component, msg)

        for i in xrange(t):
            i = t - i - 1
            obj_msg = objs[<int>order[i]].marginalize()
            for j in xrange(i+1, t):
                obj2 = objs[<int>order[j]]
                obj2.process_one_message(obj_msg, do_rollback=True)

        return result


cpdef double do_one_pass_DC_inner(list objs, list order, double lmbda, double tol, list component, int msg):
    cdef CSimplePathwayGeneral obj, obj2
    cdef int i, k
    cdef double tmp1, tmp2
    if len(order) == 1:
        # inner case, will gather all messages and update parameters
        k = order[0]
        obj = objs[k]
        obj.clear_messages()
        obj.set_resposible_for(obj.pathway, to_array(set()))
        for i in component:
            if i != k:
                obj2 = objs[i]
                obj.process_one_message(obj2.marginalize(), only1=True)
        if msg > 0:
            print ".",
            sys.stdout.flush()
        res = obj.local_update_params(lmbda, tol, component)
        for i in component:
            if i != k:
                obj2 = objs[i]
                obj2.sync_from(obj)
        return res

    else:
        k = len(order)
        t = k/2


        tmp1 = do_one_pass_DC_inner_recursive(objs, order, lmbda, tol, component, t, msg)


        order = list(reversed(order))
        t = len(order) - t

        tmp2 = do_one_pass_DC_inner_recursive(objs, order, lmbda, tol, component, t, msg)


        return tmp1 + tmp2

def cpglasso_DC(S, pathways, lmbda, tol=1e-4, maxIter=50, debug=False, Theta=None, int msg=2):
    pathways = map(lambda x: list(sorted(x)), pathways)
    graph = get_pathway_graph(pathways)
    tree = nx.minimum_spanning_tree(graph)
    objs = setup_pathways(graph, pathways, S, Theta=Theta, debug=debug)
    j = 1
    components = nx.connected_components(tree)
    for component in components:
        component = list(component) # convert set to list for NetworkX > 1.10
        j += 1
        source = random.choice(component)
        for i in xrange(maxIter):
            if msg > 0:
                print "iter", i,
                sys.stdout.flush()
            order = list(nx.depth_first_search.dfs_postorder_nodes(tree, source=source))
            #order = list(nx.breadth_first_search.bfs_successors(tree, source=source).viewkeys())
            err = do_one_pass_DC_inner2(objs, order, lmbda, tol, order, msg=msg)
            if i % 2 == 1:
                gc.collect()
            if msg > 0:
                print err
                sys.stdout.flush()
            if err < tol:
                break
        else:
            print "reached maximum number of iterations"


    return construct_result(objs, S.shape[0])



def get_random_spanning_tree(graph):
    graph_copy = nx.Graph(graph)
    for edge in graph.edges_iter():
        graph_copy[edge[0]][edge[1]]["weight"] = random.random()
    return nx.minimum_spanning_tree(graph_copy)


@cython.wraparound(True)
def compute_order(pathway_sets, order, pindex):
    all_vals = []
    responsible = [set(pathway_sets[pindex])]
    other = [set()]
    cur_set = set()
    for i in reversed(order):
        cur_set.update(pathway_sets[i])
        all_vals.append(copy.copy(cur_set))
        if len(all_vals) > 1:
            responsible.append(all_vals[-1] - all_vals[-2])
            other.append(all_vals[-1] - responsible[-1])
    return responsible, other

@cython.wraparound(True)
def apply_order(responsible, other, order, objs):
    for i, k in enumerate(reversed(order)):
        resp0 = to_array(responsible[i])
        resp0.sort()
        other0 = to_array(other[i])
        other0.sort()
        objs[k].set_resposible_for(resp0, other0)

@cython.wraparound(True)
def update_pathway(objs, tree, pathway_sets, pindex, lmbda=0.1, update=True, tol=1e-6, debug=False):
    for obj in objs:
        obj.clear_messages()
    order = list(nx.depth_first_search.dfs_postorder_nodes(tree, source=pindex))

    responsible, other = compute_order(pathway_sets, order, pindex)
    apply_order(responsible, other, order, objs)

    assert order[-1] == pindex
    for i, k in enumerate(order):
        if k != pindex:
            msg = objs[k].marginalize()
            for j in order[i+1:]:
                objs[j].process_one_message(msg)

    res = -1
    if update:
        res = objs[pindex].local_update_params(lmbda, tol, range(len(objs)))
        for i in xrange(len(objs)):
            if i != pindex:
                objs[i].sync_from(objs[pindex])
    return res


def cpglasso(S, pathways, lmbda, tol=1e-4, random_trees=False, debug=False, maxIter=25):
    """
    > cpglasso(S, pathways, lmbda, tol=1e-4, random_trees=False, debug=False, maxIter=25)
    S - covariance matrix
    pathways - list of pathways (pathwya = list of indices of nodes that are present in it)
    lmbda - regularization parameter
    """
    k = len(pathways)
    graph = get_pathway_graph(pathways)
    tree = nx.minimum_spanning_tree(graph)
    pathways = map(lambda x: list(sorted(x)), pathways)
    objs = setup_pathways(graph, pathways, S, debug=debug)

    for i in xrange(maxIter):
        if random_trees:
            tree = get_random_spanning_tree(graph)
        err = 0

        for j in xrange(k):
            err += update_pathway(objs, tree, pathways,  j, lmbda=lmbda, tol=tol)
        print err
        if err < tol:
            break

    return construct_result(objs, S.shape[0])
