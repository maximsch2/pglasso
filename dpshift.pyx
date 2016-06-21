from libc.math cimport fabs, fmax, fmin
import numpy as np
cimport numpy as np
cimport cython


# this is an implementation of shifted graphical lasso problem based on DP-GLASSO algorithm

@cython.profile(False)
cdef inline double new_cy_fsign(double x, double y) nogil:
    return fabs(x) if y >= 0 else -fabs(x)


@cython.profile(False)
cdef inline double new_compute_obj3(double[:] grad, double[:, :] u, double[:, :] b, double[:, :] shift, double w22, int cur_index) nogil:
    cdef double res = 0
    # grad.dot(u+b)
    for i in xrange(grad.shape[0]):
        res += grad[i]*(u[cur_index, i] + b[cur_index, i])*0.5 - shift[cur_index, i]*u[cur_index, i]*w22
    return res


@cython.profile(False)
cdef inline void initial_compute_grad3(double[:, :] Q, double[:, :] u, double[:, :] b, double [:] grad, int cur_index) nogil:
    cdef int qq, i, j
    cdef double tmp
    qq = Q.shape[0]

    # grad =  * Q.dot(u+b)
    for i in xrange(qq):
        tmp = 0.0
        if i == cur_index:
            continue
        for j in xrange(qq):
            if j == cur_index:
                continue
            tmp += Q[i, j] * (b[cur_index, j] + u[cur_index, j])
        grad[i] = tmp

@cython.profile(False)
cdef inline void new_cy_box_qp_f_shift2(double[:, :] Q, double[:, :] u, double[:, :] b, double rho, double[:] grad, int MAXIT, double tol, double[:, :] shift, double w22, int cur_index) nogil:
    cdef int qq, outer, i, j, k
    cdef double objcur, objold, dlx, bb, tt, diff, uold, tmp, unew
    qq = Q.shape[0]

    initial_compute_grad3(Q, u, b, grad, cur_index)

    objcur = new_compute_obj3(grad, u, b, shift, w22, cur_index)
    objold = objcur
    outer = 0
    dlx = 100500
    while (dlx >= tol) and (outer <= MAXIT):
        outer += 1
        for j in xrange(qq):
            if j == cur_index:
                continue

            uold = u[cur_index, j]

            bb = uold + (shift[cur_index, j]*w22 - grad[j])/Q[j, j]

            if fabs(bb) < rho:
                unew = bb
            elif bb > 0:
                unew = rho
            else:
                unew = -rho

            if unew != uold:
                diff = unew - uold
                u[cur_index, j] = unew
                for i in xrange(qq):
                    if i == cur_index:
                        continue
                    grad[i] += diff * Q[j, i]

        objcur = new_compute_obj3(grad, u, b, shift, w22, cur_index)
        dlx = fabs(objcur-objold)/(fabs(objold)+1e-6)
        objold = objcur



cpdef new_py_dpglasso_shift(double[:, :] Sigma,double rho, double [:, :] shift,double[:, :] X = None, double[:, :] invX = None, double[:, :] U = None, int outerMaxiter = 100, double outer_tol=1e-5, int check_nan = False):
    cdef int p, i, j, cd_iter, ii, kk
    cdef double w22, tmp, tol, tol2
    cdef double THRESHTOL = fmin(1e-7, outer_tol)
    cdef double diff_max, base_max

    p = Sigma.shape[0]
    # TODO: check that Sigma is symmetric and square
    objs = []
    diffs = []
    # if(is.null(X))  X= diag( 1/(rep(rho,p) + diag(Sigma)) )
    if X is None:
        X = np.diag(1./(np.diag(Sigma) + rho))
        assert invX is not None

    if U is None:
        U = np.zeros((p, p))
    # if(is.null(invX)) { U.mat = diag(rep(rho,p)); invX<- Sigma + U.mat } else {U.mat <- invX - Sigma; }

    if invX is None:
        U = np.eye(p)*rho
        invX = np.array(Sigma) + U
    else:
        for i in xrange(p):
            for j in xrange(p):
                U[i, j] = invX[i, j] - Sigma[i, j]

        #U = np.array(invX) - Sigma

    # TODO, check that all dimensions match

    cdef list rel_err = []
    cdef list obj_vals = []
    ii = 0
    kk = 0
    tol = 10
    cdef double[:, :] vec_diagsold = X.copy()
    cdef double[:] grad_vec = np.zeros(p)

    with nogil:
        while kk < outerMaxiter:
            ii += 1
            cd_iter = ii % p




            w22 = Sigma[cd_iter, cd_iter] + rho

            new_cy_box_qp_f_shift2(X, U , Sigma, rho, grad_vec, 1000, fmin(1e-6, outer_tol), shift, w22, cd_iter)


            for i in xrange(p):
                if i == cd_iter:
                    continue
                tmp = - grad_vec[i] / w22
                if fabs(U[cd_iter, i]) < rho*(1-THRESHTOL):
                    tmp = - shift[cd_iter, i]
                if fabs(tmp) < THRESHTOL:
                    tmp = 0
                X[cd_iter, i] = tmp
                X[i, cd_iter] = tmp

                # slows things down:
                #  U[i, cd_iter] = U[cd_iter, i]

            # formula (5.5)
            tmp = 0
            for i in xrange(p):
                if i == cd_iter:
                    continue
                tmp += (U[cd_iter, i] + Sigma[cd_iter, i])*X[cd_iter, i]
            X[cd_iter, cd_iter] = (1 - tmp)/w22


            U[cd_iter, cd_iter] = rho

            if cd_iter == p-1: # time to compute objective function
                if check_nan:
                    with gil:
                        if np.isnan(X).any():
                            print "NaNs in the solution"
                            return None
                kk += 1

                diff_max = 0
                base_max = 0
                for i in xrange(p):
                    for j in xrange(p):
                        tmp = fabs(X[i, j] - vec_diagsold[i, j])
                        if tmp > diff_max:
                            diff_max = tmp
                        tmp = fabs(vec_diagsold[i, j])
                        if tmp > base_max:
                            base_max = tmp
                tol = diff_max/base_max

                vec_diagsold[:, :] = X
                if tol < outer_tol:
                    break

        for i in xrange(p):
            for j in xrange(p):
                invX[i, j] = Sigma[i, j] + U[i, j]
    #print "Exited after", kk, "iterations"
    return X, invX, objs, diffs
