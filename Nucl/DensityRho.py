import numpy as np
from scipy.special import genlaguerre, gamma
from math import factorial
from scipy.constants import physical_constants
from dataclasses import dataclass, field
from typing import List
import re, itertools
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class OrbitData:
    n: List[int] = field(default_factory=list)
    l: List[int] = field(default_factory=list)
    j: List[int] = field(default_factory=list)
    eps: List[float] = field(default_factory=list)
    occ: List[float] = field(default_factory=list)
    U: List[List[float]] = field(default_factory=list)

    def num_levels(self):
        return len(self.n)  # or any other field

@dataclass
class NuclearShellStructureData:
    proton: OrbitData = field(default_factory=OrbitData)
    neutron: OrbitData = field(default_factory=OrbitData)

class Density_CoordinateSpace:
    def __init__(self):
        ##################################
        ###      global variables      ###
        ##################################
        # Class-specific configuration
        self.l_letter = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']

        # Mesh/grid parameters
        self.r_min = 0
        self.r_max = 10
        self.r_meshsize = 500
        self.rmesh = np.linspace(self.r_min, self.r_max, self.r_meshsize)

        # Final density storage
        self.Rho_final = np.zeros(self.r_meshsize)

        # Bra/ket data objects
        self.bra_data = NuclearShellStructureData() # final state  bra (<bra|)
        self.ket_data = NuclearShellStructureData() # intial state ket (|ket>) 

        # core
        # number of levels
        self.TheNumSPLsCore_p = 0
        self.TheNumSPLsCore_n = 0
        # particle number in core
        self.p_core = 0
        self.n_core = 0
        # total number of orbits
        self.total_levels = 0
        self.total_level_p = 0
        self.total_level_n = 0
        # 
        self.hw = 0

    # Set hw
    def SetFrequency(self, hw):
        self.hw = hw

    # Set the core
    def SetCore(self, Core_p, Core_n):
        self.p_core = Core_p
        self.n_core = Core_n
        # determine the number of levels in the core
        temp_p = 0
        for j2 in self.ket_data.proton.j:
            if temp_p == self.p_core:
                break
            temp_p += j2+1
            self.TheNumSPLsCore_p += 1
        temp_n = 0
        for j2 in self.ket_data.neutron.j:
            if temp_n == self.n_core:
                break
            temp_n += j2+1
            self.TheNumSPLsCore_n += 1
        # determine number of levels
        self.total_level_p = len(self.ket_data.proton.n)
        self.total_level_n = len(self.ket_data.neutron.n)
        self.total_levels = self.total_level_p + self.total_level_n

    # Define a function to parse each line
    def parse_line(self, line, data):
        # Regular expression to match the line format
        # pattern = r"\s*(\d+):\s+(\d+)\s+(\d+)\s+(\d+)\s+([-\d]+)\s+([-\d.]+)\s+([-\d.]+)\s+"
        pattern = r"\s*(\d+):\s+(\d+)\s+(\d+)\s+(\d+)\s+([-\d]+)\s+([-\d.]+)\s+([-\d.]+)\s+\|\s+(.*)"

        match = re.match(pattern, line)
        if match:
            index = int(match.group(1))
            n = int(match.group(2))
            l = int(match.group(3))
            two_j = int(match.group(4))
            two_tz = int(match.group(5))
            spe = float(match.group(6))
            occ = float(match.group(7))
            overlaps = [float(val) for val in match.group(8).split()]
            # print(overlaps)
            if two_tz == -1: #proton
                data.proton.n.append(n)
                data.proton.l.append(l)
                data.proton.j.append(two_j)
                data.proton.eps.append(spe)
                data.proton.occ.append(occ)
                data.proton.U.append(overlaps)
            else:
                data.neutron.n.append(n)
                data.neutron.l.append(l)
                data.neutron.j.append(two_j)
                data.neutron.eps.append(spe)
                data.neutron.occ.append(occ)
                data.neutron.U.append(overlaps)
            return [index, n, l, two_j, two_tz, spe, occ]
        else:
            return None

    # Read and parse the file
    def ReadSPLs(self, filename, data):
        if filename == '':
            print('We need HF single-particle states!! exit!')
            exit(0)
        with open(filename, 'r') as file:
            #next(file)  # Skip the header line
            for line in file:
                parsed_line = self.parse_line(line, data)

    # HO radial wavefunction
    def HO_radial_wavefunction(self, n, l, r, hw=1.0):
        """
        Harmonic oscillator radial wavefunction R_{n,l}(r)
        Parameters:
            n : radial quantum number (number of nodes)
            l : orbital angular momentum
            r : radial grid (numpy array)
            b : oscillator length parameter (in fm)
            hw: frequency of the HO basis (in MeV)
        Returns:
            R_nl(r) as a numpy array
        """
        hc = physical_constants['reduced Planck constant times c in MeV fm'][0]
        m_n = physical_constants['neutron mass energy equivalent in MeV'][0]
        m_p = physical_constants['proton mass energy equivalent in MeV'][0]
        m_nucl = (m_p + m_n)*0.5
        b = hc / np.sqrt( m_nucl * hw)  # Suhonen Eq. 3.43
        rho = r / b
        laguerre_poly = genlaguerre(n, l + 0.5)
        norm = np.sqrt(2 * factorial(n) / (b**3 * gamma(n + l + 1.5)))
        R_nl = norm * rho**l * np.exp(-0.5 * rho**2) * laguerre_poly(rho**2)
        return R_nl

    # flip bra and ket
    def invert_bra_ket(self):
        self.bra_data, self.ket_data = self.ket_data, self.bra_data

    def CleanRho(self):
        self.Rho_final.fill(0.0)

    def GetProtonOrbitsNum(self):
        return len(self.ket_data.proton.n)

    def GetNeutronOrbitsNum(self):
        return len(self.ket_data.neutron.n)

    def Get_orbit_index_HF(self, n, l, j, tz):
        temp_index = 0 
        if tz == -1:      # Proton
            for temp_index in range(self.GetProtonOrbitsNum()):
                if self.ket_data.proton.n[temp_index] == n and self.ket_data.proton.l[temp_index] == l and self.ket_data.proton.j[temp_index] == j:
                    break
            temp_index += temp_index

        elif tz == 1:   # Neutron
            for temp_index in range(self.GetNeutronOrbitsNum()):
                if self.ket_data.neutron.n[temp_index] == n and self.ket_data.neutron.l[temp_index] == l and self.ket_data.neutron.j[temp_index] == j:
                    break
            temp_index += temp_index + 1
        else:
            print('The isospin tz should be 1 or -1 ! instead of  ', tz)
            exit(0)
        return temp_index              

    def Get_orbit_from_HF_index(self, HF_index, data):
        if HF_index % 2 == 0:   # proton
            return data.proton
        if HF_index % 2 == 1:   # proton
            return data.neutron

    def Get_local_index_from_HFindx(self, HF_index):
        if HF_index % 2 == 0:   # proton
            return int(HF_index / 2)
        if HF_index % 2 == 1:   # proton
            return int((HF_index - 1)/ 2)

    def Is_proton_or_neutron_index(self, HF_index):
        if HF_index % 2 == 0:   # proton
            return -1
        if HF_index % 2 == 1:   # proton
            return 1

    def AddToRho_1b(self, index_HF_i, index_HF_j, reducedME):
        # <i bra | ket  j>
        Rho_tmp_bra = np.zeros(self.r_meshsize)
        Rho_tmp_ket = np.zeros(self.r_meshsize)
        index_i = self.Get_local_index_from_HFindx(index_HF_i)
        index_j = self.Get_local_index_from_HFindx(index_HF_j)
        temp_bra = self.Get_orbit_from_HF_index(index_HF_i, self.bra_data)
        temp_ket = self.Get_orbit_from_HF_index(index_HF_j, self.ket_data)
        for ho_n in range(len(temp_bra.U[index_i])):
            Rho_tmp_bra += temp_bra.U[index_i][ho_n] * self.HO_radial_wavefunction(ho_n, temp_bra.l[index_i], self.rmesh, self.hw)
        
        for ho_n in range(len(temp_ket.U[index_j])):
            Rho_tmp_ket += temp_ket.U[index_j][ho_n] * self.HO_radial_wavefunction(ho_n, temp_ket.l[index_j], self.rmesh, self.hw)    
        
        self.Rho_final += reducedME * Rho_tmp_bra * Rho_tmp_ket

    def AddToRho_2b(self, index_HF_i, index_HF_j, index_HF_k, index_HF_l, reducedME):
        # <ij bra | ket  kl>
        Rho_tmp_bra_i = np.zeros(self.r_meshsize)
        Rho_tmp_bra_j = np.zeros(self.r_meshsize)
        Rho_tmp_ket_k = np.zeros(self.r_meshsize)
        Rho_tmp_ket_l = np.zeros(self.r_meshsize)

        index_i = self.Get_local_index_from_HFindx(index_HF_i)
        index_j = self.Get_local_index_from_HFindx(index_HF_j)
        index_k = self.Get_local_index_from_HFindx(index_HF_k)
        index_l = self.Get_local_index_from_HFindx(index_HF_l)

        temp_bra_i = self.Get_orbit_from_HF_index(index_HF_i, self.bra_data)
        temp_bra_j = self.Get_orbit_from_HF_index(index_HF_j, self.bra_data)
        temp_ket_k = self.Get_orbit_from_HF_index(index_HF_k, self.ket_data)
        temp_ket_l = self.Get_orbit_from_HF_index(index_HF_l, self.ket_data)
        
        for ho_n in range(len(temp_bra_i.U[index_i])):
            Rho_tmp_bra_i += temp_bra_i.U[index_i][ho_n] * self.HO_radial_wavefunction(ho_n, temp_bra_i.l[index_i], self.rmesh, self.hw)
        for ho_n in range(len(temp_bra_j.U[index_j])):
            Rho_tmp_bra_j += temp_bra_j.U[index_j][ho_n] * self.HO_radial_wavefunction(ho_n, temp_bra_j.l[index_j], self.rmesh, self.hw)        
        
        for ho_n in range(len(temp_ket_k.U[index_k])):
            Rho_tmp_ket_k += temp_ket_k.U[index_k][ho_n] * self.HO_radial_wavefunction(ho_n, temp_ket_k.l[index_k], self.rmesh, self.hw)    
        for ho_n in range(len(temp_ket_l.U[index_l])):
            Rho_tmp_ket_l += temp_ket_l.U[index_l][ho_n] * self.HO_radial_wavefunction(ho_n, temp_ket_l.l[index_l], self.rmesh, self.hw)    
        
        self.Rho_final += reducedME * Rho_tmp_bra_i * Rho_tmp_bra_j * Rho_tmp_ket_k * Rho_tmp_ket_l

    def OutputRho(self, filename):
        """
        Save the radial mesh points and Rho_final values to a text file.

        The output file will have two columns:
        r (fm)   Rho(r)
        """
        with open(filename, 'w') as f:
            for r, rho in zip(self.rmesh, self.Rho_final):
                f.write(f"{r:.6f} {rho:.6e}\n")
        return

    def PlotDensity(self, outputfilename = 'radial_wavefunction.pdf', titilename = 'Density'):
        plt.plot(self.rmesh, self.Rho_final, label=f'Density')
        plt.xlabel('r (fm)')
        plt.ylabel(r'$\rho(r)$ (fm$^{-3}$)')
        plt.title(titilename)
        plt.legend()
        plt.grid()
        plt.savefig(outputfilename)
        #plt.show()
        plt.close()
        return

    # core
    def Core_ParticleDensity(self, data = None):
        if data is None:
            data = self.ket_data

        Rho_core = np.zeros(self.r_meshsize)
        for spl_core_p in range(self.TheNumSPLsCore_p):
            Rcore_p = np.zeros(self.r_meshsize)
            for ho_n in range(len(data.proton.U[spl_core_p])):
                Rcore_p += data.proton.U[spl_core_p][ho_n] * self.HO_radial_wavefunction(ho_n, data.proton.l[spl_core_p], self.rmesh, self.hw)
            Rho_core += ( data.proton.j[spl_core_p] + 1) * Rcore_p * Rcore_p

        for spl_core_n in range(self.TheNumSPLsCore_n):
            Rcore_n = np.zeros(self.r_meshsize)
            for ho_n in range(len(data.neutron.U[spl_core_n])):
                Rcore_n += data.neutron.U[spl_core_n][ho_n] * self.HO_radial_wavefunction(ho_n, data.neutron.l[spl_core_n], self.rmesh, self.hw)
            Rho_core += ( data.neutron.j[spl_core_n] + 1) * Rcore_n * Rcore_n
        self.Rho_final += Rho_core
        return Rho_core

    # charge density for the core, assume that in unit of e
    def Core_ChargeDensity(self, data = None):
        if data is None:
            data = self.ket_data
            
        Rho_core = np.zeros(self.r_meshsize)
        for spl_core_p in range(self.TheNumSPLsCore_p):
            Rcore_p = np.zeros(self.r_meshsize)
            for ho_p in range(len(data.proton.U[spl_core_p])):
                Rcore_p += data.proton.U[spl_core_p][ho_p] * self.HO_radial_wavefunction(ho_p, data.proton.l[spl_core_p], self.rmesh, self.hw)
            Rho_core += ( data.proton.j[spl_core_p] + 1) * Rcore_p * Rcore_p
        self.Rho_final += Rho_core
        return Rho_core

    # Neutron density for the core
    def Core_NeutronDensity(self, data = None):
        if data is None:
            data = self.ket_data
            
        Rho_core = np.zeros(self.r_meshsize)
        for spl_core_n in range(self.TheNumSPLsCore_n):
            Rcore_n = np.zeros(self.r_meshsize)
            for ho_n in range(len(data.neutron.U[spl_core_n])):
                Rcore_n += data.neutron.U[spl_core_n][ho_n] * self.HO_radial_wavefunction(ho_n, data.neutron.l[spl_core_n], self.rmesh, self.hw)
            Rho_core += ( data.neutron.j[spl_core_n] + 1) * Rcore_n * Rcore_n
        self.Rho_final += Rho_core
        return Rho_core


    def Core_Density(self, op):
        # we assume that the bra and ket are the same
        # we only include the onebody operator only here!!        
        orbits_op = op.ms.orbits
        norbs = orbits_op.get_num_orbits()

        Rho_core = np.zeros(self.r_meshsize)
        # proton 1b
        for local_i in range(self.TheNumSPLsCore_p):
            # only include the diagonal term for the core
            local_j = local_i 
            data = self.ket_data.proton
            op_i = orbits_op.get_orbit_index( data.n[local_i], data.l[local_i], data.j[local_i], -1)
            oi = orbits_op.get_orbit(op_i)
            HF_i = self.Get_orbit_index_HF(oi.n, oi.l, oi.j, oi.z)

            # must be scalar operator
            reducedME = op.get_1bme(op_i, op_i) 
            # print(oi.n, oi.l, oi.j, oi.z, reducedME)
            Rcore_p = np.zeros(self.r_meshsize)
            for ho_n in range(len(data.U[local_i])):
                Rcore_p += data.U[local_i][ho_n] * self.HO_radial_wavefunction(ho_n, data.l[local_i], self.rmesh, self.hw)
            Rho_core += reducedME * ( data.j[local_i] + 1) * Rcore_p * Rcore_p

        # neutron 1b
        for local_i in range(self.TheNumSPLsCore_n):
            # only include the diagonal term for the core
            local_j = local_i 
            data = self.ket_data.neutron
            op_i = orbits_op.get_orbit_index( data.n[local_i], data.l[local_i], data.j[local_i], 1)
            oi = orbits_op.get_orbit(op_i)
            HF_i = self.Get_orbit_index_HF(oi.n, oi.l, oi.j, oi.z)

            # must be scalar operator
            reducedME = op.get_1bme(op_i, op_i) 

            Rcore_n = np.zeros(self.r_meshsize)
            for ho_n in range(len(data.U[local_i])):
                Rcore_n += data.U[local_i][ho_n] * self.HO_radial_wavefunction(ho_n, data.l[local_i], self.rmesh, self.hw)
            Rho_core += reducedME * ( data.j[local_i] + 1) * Rcore_n * Rcore_n
        self.Rho_final += Rho_core
        return













