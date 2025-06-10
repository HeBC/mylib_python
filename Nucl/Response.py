#!/usr/bin/env python3
import sys, os, subprocess, itertools, math
import numpy as np
import functools
import copy
import gzip
from scipy.constants import physical_constants
from scipy import integrate
from scipy.special import gamma
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j, clebsch_gordan
from scipy.special import spherical_jn
import pandas as pd
if(__package__==None or __package__==""):
    from Orbits import Orbits, OrbitsIsospin
    import ModelSpace
    import nushell2snt
    import BasicFunctions
    import Operator
else:
    from . import Orbits, OrbitsIsospin
    from . import ModelSpace
    from . import nushell2snt
    from . import BasicFunctions
    from . import Operator

@functools.lru_cache(maxsize=None)
def _threej(j1, j2, j3, m1, m2, m3):
    return float(wigner_3j(j1, j2, j3, m1, m2, m3))
@functools.lru_cache(maxsize=None)
def _sixj(j1, j2, j3, j4, j5, j6):
    return float(wigner_6j(j1, j2, j3, j4, j5, j6))
@functools.lru_cache(maxsize=None)
def _ninej(j1, j2, j3, j4, j5, j6, j7, j8, j9):
    return float(wigner_9j(j1, j2, j3, j4, j5, j6, j7, j8, j9))
@functools.lru_cache(maxsize=None)
def _clebsch_gordan(j1, j2, j3, m1, m2, m3):
    return float(clebsch_gordan(j1, j2, j3, m1, m2, m3))
def _ls_coupling(la, ja, lb, jb, Lab, Sab, J):
    return np.sqrt( (2*ja+1)*(2*jb+1)*(2*Lab+1)*(2*Sab+1) ) * \
            np.float( wigner_9j( la, 0.5, ja, lb, 0.5, jb, Lab, Sab, J) )
def _Kronecker_delta(i: int, j: int):
    if(i==j): return 1
    if(i!=j): return 0
hc = physical_constants['reduced Planck constant times c in MeV fm'][0]
m = (physical_constants['proton mass energy equivalent in MeV'][0] + physical_constants['neutron mass energy equivalent in MeV'][0])/2
mu_p = physical_constants["proton magn. moment to nuclear magneton ratio"][0] 
mu_n = physical_constants["neutron magn. moment to nuclear magneton ratio"][0] 
k_is = mu_p + mu_n
k_iv = mu_p - mu_n
gA = 1.27
mpi = 138
class Response(Operator):

    def set_response(self, Q, hw, response_name, isospin=0):
        gA = -1.27
        FA = gA
        Fp = 2*m * FA / (mpi**2+Q**2)
        self.allocate_operator(self.ms)
        if(response_name=="M"): 
            M0 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            M1 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            M0.set_response_elemental(self.rankJ, self.rankJ, Q, hw, 0, "M")
            M1.set_response_elemental(self.rankJ, self.rankJ, Q, hw, 1, "M")
            self.one = (M0.one + M1.one) * 0.5
        if(response_name=="Mls"): 
            tmp0 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms, skew=True)
            tmp1 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms, skew=True)
            tmp0.set_response(Q, hw, "Phipp", isospin=0)
            tmp1.set_response(Q, hw, "Phipp", isospin=1)
            self.one = (0.5 * (k_is - 1) * tmp0.one + 0.5 * (k_iv - 1) * tmp1.one) * 0.25 * Q**2 / m**2
        if(response_name=="L5"): # iL5
            tmp = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            tmp.set_response(Q, hw, "Sigmapp", isospin=1)
            self.one = - 0.5 * (FA - Q**2 / (2*m) * Fp) * tmp.one 
        if(response_name=="Tel"): 
            s0 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms, skew=True)
            s1 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms, skew=True)
            l0 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms, skew=True)
            l1 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms, skew=True)
            l0.set_response(Q, hw, "Deltap", isospin=0)
            l1.set_response(Q, hw, "Deltap", isospin=1)
            s0.set_response(Q, hw, "Sigma", isospin=0)
            s1.set_response(Q, hw, "Sigma", isospin=1)
            self.one = ((l0.one + l1.one) * 0.5 + (s0.one * k_is + s1.one * k_iv) * 0.25) * Q / m 
        if(response_name=="ConvTel"):
            l0 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms, skew=True)
            l1 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms, skew=True)
            l0.set_response(Q, hw, "Deltap", isospin=0)
            l1.set_response(Q, hw, "Deltap", isospin=1)
            self.one = (l0.one + l1.one) * 0.5 * Q / m 
        if(response_name=="SpinTel"):
            s0 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms, skew=True)
            s1 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms, skew=True)
            s0.set_response(Q, hw, "Sigma", isospin=0)
            s1.set_response(Q, hw, "Sigma", isospin=1)
            self.one = (s0.one * k_is + s1.one * k_iv) * 0.25 * Q / m 
        if(response_name=="Tel5"): # iTel5
            tmp = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            tmp.set_response(Q, hw, "Sigmap", isospin=1)
            self.one = - (FA * 0.5) * tmp.one
        if(response_name=="Tmag"): # iTmag
            s0 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            s1 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            l0 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            l1 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            l0.set_response(Q, hw, "Delta", isospin=0)
            l1.set_response(Q, hw, "Delta", isospin=1)
            s0.set_response(Q, hw, "Sigmap", isospin=0)
            s1.set_response(Q, hw, "Sigmap", isospin=1)
            self.one = ((l0.one + l1.one) * 0.5 - (s0.one * k_is + s1.one * k_iv) * 0.25) * Q / m
        if(response_name=="SpinTmag"): # iTmag
            s0 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            s1 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            s0.set_response(Q, hw, "Sigmap", isospin=0)
            s1.set_response(Q, hw, "Sigmap", isospin=1)
            self.one = - (s0.one * k_is + s1.one * k_iv) * 0.25 * Q / m
        if(response_name=="ConvTmag"): # iTmag
            l0 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            l1 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            l0.set_response(Q, hw, "Delta", isospin=0)
            l1.set_response(Q, hw, "Delta", isospin=1)
            self.one = (l0.one + l1.one) * 0.5 * Q / m
        if(response_name=="Tmag5"):
            tmp = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            tmp.set_response(Q, hw, "Sigma", isospin=1)
            self.one = FA * 0.5 * tmp.one 
        if(response_name=="Delta"):
            tmp = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            tmp.set_response_elemental(self.rankJ, self.rankJ, Q, hw, isospin, "Md")
            self.one = tmp.one
        if(response_name=="Deltap"):
            tmp1 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms, skew=True)
            tmp2 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms, skew=True)
            tmp1.set_response_elemental(self.rankJ+1, self.rankJ, Q, hw, isospin, "Md")
            if(self.rankJ>0): tmp2.set_response_elemental(self.rankJ-1, self.rankJ, Q, hw, isospin, "Md")
            self.one = 1/np.sqrt(2*self.rankJ+1) * (-np.sqrt(self.rankJ) * tmp1.one + np.sqrt(self.rankJ+1) * tmp2.one)
        if(response_name=="Sigma"):
            tmp = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms, skew=True)
            tmp.set_response_elemental(self.rankJ, self.rankJ, Q, hw, isospin, "Ms")
            self.one = tmp.one
        if(response_name=="Sigmap"):
            tmp1 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            tmp2 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            tmp1.set_response_elemental(self.rankJ+1, self.rankJ, Q, hw, isospin, "Ms")
            if(self.rankJ>0): tmp2.set_response_elemental(self.rankJ-1, self.rankJ, Q, hw, isospin, "Ms")
            self.one = 1/np.sqrt(2*self.rankJ+1) * (-np.sqrt(self.rankJ) * tmp1.one + np.sqrt(self.rankJ+1) * tmp2.one)
        if(response_name=="Sigmapp"):
            tmp1 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            tmp2 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            tmp1.set_response_elemental(self.rankJ+1, self.rankJ, Q, hw, isospin, "Ms")
            if(self.rankJ>0): tmp2.set_response_elemental(self.rankJ-1, self.rankJ, Q, hw, isospin, "Ms")
            self.one = 1/np.sqrt(2*self.rankJ+1) * (np.sqrt(self.rankJ+1) * tmp1.one + np.sqrt(self.rankJ) * tmp2.one)
        if(response_name=="Omega"):
            tmp = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            tmp.set_response_elemental(self.rankJ, self.rankJ, Q, hw, isospin, "Msd")
            self.one = tmp.one
        if(response_name=="Omegap"):
            tmp1 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms, skew=True)
            tmp2 = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms, skew=True)
            tmp1.set_response(Q, hw, isospin, "Omega")
            tmp2.set_response(Q, hw, isospin, "Sigmapp")
            self.one = tmp1.one + 0.5 * tmp2.one
        if(response_name=="Phipp"):
            tmp = Response(rankJ=self.rankJ, rankP=self.rankP, rankZ=self.rankZ, ms=self.ms)
            tmp.set_response_elemental(self.rankJ, self.rankJ, Q, hw, isospin, "Mls")
            self.one = tmp.one 

    def set_response_elemental(self, L, J, Q, hw, isospin, function_name):
        self.allocate_operator(self.ms)
        lam = self.rankJ
        orbits = self.ms.orbits
        if(function_name == "M"): func = self.func_M
        if(function_name == "Ms"): func = self.func_Ms
        if(function_name == "Md"): func = self.func_Md
        if(function_name == "Msd"): func = self.func_Msd
        if(function_name == "Mls"): func = self.func_Mls
        for p in range(1,orbits.get_num_orbits()+1):
            for q in range(p,orbits.get_num_orbits()+1):
                op = orbits.get_orbit(p)
                oq = orbits.get_orbit(q)
                f_iso = 0
                if(isospin==0): f_iso = 1
                if(isospin==1): 
                    if(op.z==oq.z):
                        if(op.z==-1): f_iso = 1
                        if(op.z== 1): f_iso =-1
                    else:
                        f_iso = 1
                if(abs(op.z-oq.z) != 2*self.rankZ): continue
                if((-1)**(op.l+oq.l) != self.rankP): continue
                if(self._triag(op.j, 2*lam, oq.j)): continue
                me = func(op, oq, L, J, Q, hw) * f_iso
                if(self.rankJ==0 and self.rankP==1 and self.rankZ==0): me /= np.sqrt(op.j+1)
                self.set_1bme(p, q, me)

    def func_M(self, op, oq, L, J, Q, hw):
        if(L!=J): raise ValueError
        n_p, l_p, j_p = op.n, op.l, op.j*0.5
        n_q, l_q, j_q = oq.n, oq.l, oq.j*0.5
        integral = integrate.quad(lambda r: r**2 * BasicFunctions.HO_radial(r, n_p, l_p, hw) * BasicFunctions.HO_radial(r, n_q, l_q, hw) * spherical_jn(J, r*Q/hc), 0, np.inf)[0]
        return (1/np.sqrt(4*np.pi)) * (-1)**(J + j_q + 0.5) * \
                np.sqrt((2*l_p+1)*(2*l_q+1)*(2*j_p+1)*(2*j_q+1)*(2*J+1)) * \
                _sixj(l_p, j_p, 0.5, j_q, l_q, J) * _threej(l_p, J, l_q, 0, 0, 0) * \
                integral

    def func_Ms(self, op, oq, L, J, Q, hw):
        n_p, l_p, j_p = op.n, op.l, op.j*0.5
        n_q, l_q, j_q = oq.n, oq.l, oq.j*0.5
        integral = integrate.quad(lambda r: r**2 * BasicFunctions.HO_radial(r, n_p, l_p, hw) * BasicFunctions.HO_radial(r, n_q, l_q, hw) * spherical_jn(L, r*Q/hc), 0, np.inf)[0]
        return (np.sqrt(3)/np.sqrt(2*np.pi)) * (-1)**l_p * \
                np.sqrt((2*l_p+1)*(2*l_q+1)*(2*j_p+1)*(2*j_q+1)*(2*L+1)*(2*J+1)) * \
                _ninej(l_p, l_q, L, 0.5, 0.5, 1, j_p, j_q, J) * _threej(l_p, L, l_q, 0, 0, 0) * \
                integral

    def func_Md(self, op, oq, L, J, Q, hw):
        n_p, l_p, j_p = op.n, op.l, op.j*0.5
        n_q, l_q, j_q = oq.n, oq.l, oq.j*0.5
        integral1 = integrate.quad(lambda r: BasicFunctions.HO_radial(r, n_p, l_p, hw) * spherical_jn(L, r*Q/hc) * \
                (r**2 * BasicFunctions.HO_radial_deriv(r, n_q, l_q, hw) - l_q*r*BasicFunctions.HO_radial(r, n_q, l_q, hw)), 0, np.inf)[0] / (Q/hc)
        integral2 = integrate.quad(lambda r: BasicFunctions.HO_radial(r, n_p, l_p, hw) * spherical_jn(L, r*Q/hc) * \
                (r**2 * BasicFunctions.HO_radial_deriv(r, n_q, l_q, hw) + (l_q+1)*r*BasicFunctions.HO_radial(r, n_q, l_q, hw)), 0, np.inf)[0] / (Q/hc)
        res = (1/np.sqrt(4*np.pi)) * (-1)**(L+j_q+1/2) * \
                np.sqrt((2*l_p+1)*(2*j_p+1)*(2*j_q+1)*(2*L+1)*(2*J+1)) * \
                _sixj(l_p, j_p, 0.5, j_q, l_q, J) * \
                -np.sqrt(l_q+1) * np.sqrt(2*l_q+3) * _sixj(L, 1, J, l_q, l_p, l_q+1) * _threej(l_p, L, l_q+1, 0, 0, 0) * integral1
        if(l_q>0): res +=(1/np.sqrt(4*np.pi)) * (-1)**(L+j_q+1/2) * \
                np.sqrt((2*l_p+1)*(2*j_p+1)*(2*j_q+1)*(2*L+1)*(2*J+1)) * \
                _sixj(l_p, j_p, 0.5, j_q, l_q, J) * \
                np.sqrt(l_q) * np.sqrt(2*l_q-1) * _sixj(L, 1, J, l_q, l_p, l_q-1) * _threej(l_p, L, l_q-1, 0, 0, 0) * integral2
        return res

    def func_Msd(self, op, oq, L, J, Q, hw):
        if(L!=J): raise ValueError
        n_p, l_p, j_p = op.n, op.l, op.j*0.5
        n_q, l_q, j_q = oq.n, oq.l, oq.j*0.5
        integral1 = integrate.quad(lambda r: BasicFunctions.HO_radial(r, n_p, l_p, hw) * spherical_jn(J, r*Q/hc) * \
                (r**2 * BasicFunctions.HO_radial_deriv(r, n_q, l_q, hw) - l_q*r*BasicFunctions.HO_radial(r, n_q, l_q, hw)), 0, np.inf)[0] / (Q*hc)
        integral2 = integrate.quad(lambda r: BasicFunctions.HO_radial(r, n_p, l_p, hw) * spherical_jn(J, r*Q/hc) * \
                (r**2 * BasicFunctions.HO_radial_deriv(r, n_q, l_q, hw) + (l_q+1)*r*BasicFunctions.HO_radial(r, n_q, l_q, hw)), 0, np.inf)[0] / (Q*hc)
        return (1/np.sqrt(4*np.pi)) * (-1)**l_p * \
                np.sqrt((2*l_p+1)*(2*j_p+1)*(2*j_q+1)*(4*j_q-2*l_q+1)*(2*J+1)) * \
                _sixj(l_p, j_p, 0.5, j_q, 2*j_q-l_q, J) * _threej(l_p, J, 2*j_p-l_q, 0, 0, 0) * \
                (-_kronecker_delta(int(j_q-0.5), l_q) * integral1 + _kronecker_delta(int(j_q-0.5), l_q-1) * integral2)
                                
    def func_Mls(self, op, oq, L, J, Q, hw):
        if(L!=J): raise ValueError
        def phi1(na, la, ja, nb, lb, jb, J, Q):
            res = 0
            for lam in [J, J+1]:
                res += (-1)**(J+lam) * (2*lam+1) * \
                        _sixj(J+1, 1, lam, 1, J, 1) * _sixj(J+1, 1, lam, lb, la, lb+1) * \
                        _ninej(la, lb, lam, 0.5, 0.5, 1, ja, jb, J)
            res *= np.sqrt((2*lb+3) * (lb+1)) * _threej(la, J+1, lb+1, 0, 0, 0)
            integral = integrate.quad(lambda r: BasicFunctions.HO_radial(r, na, la, hw) * spherical_jn(J+1, r*Q/hc) * \
                (r**2 * BasicFunctions.HO_radial_deriv(r, nb, lb, hw) - lb*r*BasicFunctions.HO_radial(r, nb, lb, hw)), 0, np.inf)[0] / Q * hc
            res *= integral
            return res

        def phi2(na, la, ja, nb, lb, jb, J, Q):
            res = 0
            if(lb==0): return res
            for lam in [J, J+1]:
                res += (-1)**(J+lam) * (2*lam+1) * \
                        _sixj(J+1, 1, lam, 1, J, 1) * _sixj(J+1, 1, lam, lb, la, lb-1) * \
                        _ninej(la, lb, lam, 0.5, 0.5, 1, ja, jb, J)
            res *= np.sqrt((2*lb-1) * lb) * _threej(la, J+1, lb-1, 0, 0, 0)
            integral = integrate.quad(lambda r: BasicFunctions.HO_radial(r, na, la, hw) * spherical_jn(J+1, r*Q/hc) * \
                (r**2 * BasicFunctions.HO_radial_deriv(r, nb, lb, hw) + (lb+1)*r*BasicFunctions.HO_radial(r, nb, lb, hw)), 0, np.inf)[0] / Q * hc
            res *= integral
            return -res

        def phi3(na, la, ja, nb, lb, jb, J, Q):
            res = 0
            if(J==0): return res
            for lam in [J-1, J]:
                res += (-1)**(J+lam) * (2*lam+1) * \
                        _sixj(J-1, 1, lam, 1, J, 1) * _sixj(J-1, 1, lam, lb, la, lb+1) * \
                        _ninej(la, lb, lam, 0.5, 0.5, 1, ja, jb, J)
            res *= np.sqrt((2*lb+3) * (lb+1)) * _threej(la, J-1, lb+1, 0, 0, 0)
            integral = integrate.quad(lambda r: BasicFunctions.HO_radial(r, na, la, hw) * spherical_jn(J-1, r*Q/hc) * \
                (r**2 * BasicFunctions.HO_radial_deriv(r, nb, lb, hw) - lb*r*BasicFunctions.HO_radial(r, nb, lb, hw)), 0, np.inf)[0] / Q * hc
            res *= integral
            return res

        def phi4(na, la, ja, nb, lb, jb, J, Q):
            res = 0
            if(J==0 or lb==0): return res
            for lam in [J-1, J]:
                res += (-1)**(J+lam) * (2*lam+1) * \
                        _sixj(J-1, 1, lam, 1, J, 1) * _sixj(J-1, 1, lam, lb, la, lb-1) * \
                        _ninej(la, lb, lam, 0.5, 0.5, 1, ja, jb, J)
            res *= np.sqrt((2*lb-1) * lb) * _threej(la, J-1, lb-1, 0, 0, 0)
            integral = integrate.quad(lambda r: BasicFunctions.HO_radial(r, na, la, hw) * spherical_jn(J-1, r*Q/hc) * \
                (r**2 * BasicFunctions.HO_radial_deriv(r, nb, lb, hw) + (lb+1)*r*BasicFunctions.HO_radial(r, nb, lb, hw)), 0, np.inf)[0] / Q * hc
            res *= integral
            return -res

        n_p, l_p, j_p = op.n, op.l, op.j*0.5
        n_q, l_q, j_q = oq.n, oq.l, oq.j*0.5
        fact = (-1)**l_p * 6 * np.sqrt((2*l_p+1) * (2*j_p+1) * (2*j_q+1)) / np.sqrt(4*np.pi)
        res = fact * np.sqrt((2*J+3) * (J+1)) * (phi1(n_p, l_p, j_p, n_q, l_q, j_q, J, Q) + phi2(n_p, l_p, j_p, n_q, l_q, j_q, J, Q))
        if(J>0): res += fact * np.sqrt((2*J-1) * J) * (phi3(n_p, l_p, j_p, n_q, l_q, j_q, J, Q) + phi4(n_p, l_p, j_p, n_q, l_q, j_q, J, Q))
        return res
