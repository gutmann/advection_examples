import numpy as np

def upwind(q,u):
    qnew = q.copy()

    fluxout = q[1:] * u
    fluxin = q[:-1] * u

    qnew[1:] = q[1:] + fluxin - fluxout

    # wrap around boundary conditions
    qnew[0] = qnew[-1]

    return qnew


def ftcs(q,u):
    """forward in time, centered in space solution"""
    dqdt = np.zeros(q.shape)

    # center the q scalar in space
    qedge = np.zeros(q.shape)
    qedge[:-1] = (q[1:] + q[:-1])/2

    # wrap around boundary conditions
    qedge[-1] = qedge[0]

    dqdt[1:] = (qedge[:-1] - qedge[1:]) * u
    dqdt[0] = dqdt[-1]
    return dqdt


def adamsbashforth_3(q, u, qold1, qold2):
    """Third order Adams-Bashforth"""
    ab3_dqdt = (23 * ftcs(q,u)
             -  16 * ftcs(qold1,u)
             +   5 * ftcs(qold2,u) ) / 12

    qnew = q + ab3_dqdt

    return qnew

adamsbashforth = adamsbashforth_3

def rungakutta_4(q, u):
    """Fourth order Runga-Kutta"""
    k0 = ftcs(q,     u)
    k1 = ftcs(q+k0/2,u)
    k2 = ftcs(q+k1/2,u)
    k3 = ftcs(q+k2,  u)

    qnew = q + (k0 + 2*k1 + 2*k2 + k3) / 6

    return qnew

rungakutta = rungakutta_4
