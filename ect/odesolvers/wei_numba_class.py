"""jit compilable implementaiton of Wei et. al 2014."""
import math

import numpy as np

from numba_baseclass import RK4Solver


class Wei(RK4Solver):
    """RK4 solver of the cell model described by in Wei et. al 2014."""

    def _rhs(self, t: float, y) -> np.ndarray:
        G_K = 25.0         # Voltage gated conductance       [mS/mm^2]
        G_Na = 30.0        # Voltage gated conductance       [mS/mm^2]
        g_l_cl = 0.1       # Calcium leak conductance        [mS/mm^2]
        g_l_k = 0.05       # Potassium leak conductence      [mS/mm^2]
        g_l_na = 0.0247    # Natrium leak conductance        [mS/mm^2]
        C = 1.0            # Capacitance representing the lipid bilayer

        # Ion Concentration related Parameters
        dslp = 0.25        # potassium diffusion coefficient
        gmag = 5.0         # maximal glial strength
        sigma = 0.17       # Oxygen diffusion coefficient
        Ukcc2 = 3e-1       # maximal KCC2 cotransporteer trength
        Unkcc1 = 1e-1      # maximal KCC1 cotransporter strength
        rho = 8e-1         # maximal pump rate

        # Volume
        vol = 1.4368e-15    # unit:m^3, when r=7 um,v=1.4368e-15 m^3
        beta0 = 7

        # Time Constant
        tau = 1e-3

        # Bath concentrations
        Kbath = 8.5     # Potassium bath
        Obath = 32      # Oxygen bath

        # Variables
        v, m, h, n, NKo, NKi, NNao, NNai, NClo, NCli, voli, O = y
        O = max(O, 0)

        volo = (1 + 1/beta0)*vol - voli     # Extracellular volume
        beta = voli/volo                    # Ratio of intracelluar to extracelluar volume

        fo = 1/(1 + math.exp((2.5 - Obath)/0.2))
        fv = 1/(1 + math.exp((beta - 20)/2))
        dslp *= fo*fv
        gmag *= fo

        Ko = NKo/volo       # mM
        Ki = NKi/voli
        Nao = NNao/volo
        Nai = NNai/voli
        Clo = NClo/volo
        Cli = NCli/voli

        # Parameters for step current
        Iext = 0

        alpha = 4*math.pi*(3*voli/(4*math.pi))**(2/3)
        F = 96485.33
        gamma = alpha/(F*voli)*1e-2

        # Gating variables
        alpha_m = 0.32*(54 + v)/(1 - math.exp(-(v + 54)/4))
        beta_m = 0.28*(v + 27)/(math.exp((v + 27)/5) - 1)

        alpha_h = 0.128*math.exp(-(v + 50)/18)
        beta_h = 4/(1 + math.exp(-(v + 27)/5))

        alpha_n = 0.032*(v + 52)/(1 - math.exp(-(v + 52)/5))
        beta_n = 0.5*math.exp(-(v + 57)/40)

        # Pump, glia and diffusion
        p = rho/(1 + math.exp((20 - O)/3))/gamma
        I_pump = p/(1 + math.exp((25 - Nai)/3))/(1 + math.exp(3.5 - Ko))
        I_glia = gmag/(1 + math.exp((18 - Ko)/2.5))
        Igliapump = (p/3/(1 + math.exp((25 - 18)/3)))*(1/(1 + math.exp(3.5 - Ko)))
        I_diff = dslp*(Ko - Kbath) + I_glia + 2*Igliapump*gamma

        # Cloride transporter (mM/s)
        fKo = (1/(1 + math.exp((16 - Ko)/1)))
        FKCC2 = Ukcc2*math.log((Ki*Cli)/(Ko*Clo))
        FNKCC1 = Unkcc1*fKo*(math.log((Ki*Cli)/(Ko*Clo)) + math.log((Nai*Cli)/(Nao*Clo)))

        # Reversal potential
        E_K = 26.64*math.log(Ko/Ki)
        E_Na = 26.64*math.log(Nao/Nai)
        E_Cl = 26.64*math.log(Cli/Clo)

        # Currents    (uA/cm^2)
        INa = G_Na*m**3*h*(v - E_Na) + g_l_na*(v - E_Na)
        IK = G_K*n**4*(v - E_K) + g_l_k*(v - E_K)
        IL = g_l_cl*(v - E_Cl)

        # Output
        dotv = (-INa - IK - IL - I_pump + Iext)/C
        dotm = alpha_m*(1 - m)-beta_m*m
        doth = alpha_h*(1 - h)-beta_h*h
        dotn = alpha_n*(1 - n)-beta_n*n

        dotNKo = tau*(gamma*beta*(IK - 2.0*I_pump) - I_diff + FKCC2*beta + FNKCC1*beta)*volo
        dotNKi = tau*(-gamma*(IK - 2.0*I_pump) - FKCC2 - FNKCC1)*voli

        dotNNao = tau*(gamma*beta*(INa + 3.0*I_pump) + FNKCC1*beta)*volo
        dotNNai = tau*(-gamma*(INa + 3.0 * I_pump) - FNKCC1)*voli

        dotNClo = tau*(-gamma*beta*IL + FKCC2*beta + 2*FNKCC1*beta)*volo
        dotNCli = tau*(gamma*IL - FKCC2 - 2*FNKCC1)*voli

        # intracellular volume dynamics
        r1 = vol/voli
        r2 = 1/beta0*vol/((1 + 1/beta0)*vol - voli)
        Ai = 132
        Ao = 18
        pii = Nai + Cli + Ki + Ai*r1
        pio = Nao + Ko + Clo + Ao*r2

        vol_hat = vol*(1.1029 - 0.1029*math.exp((pio - pii)/20))
        dotVoli = -(voli - vol_hat)/0.25*tau

        # oxygen dynamics
        dotO = tau*(-5.3*(I_pump + Igliapump)*gamma + sigma*(Obath - O))

        return np.array([
            dotv, dotm, doth, dotn,
            dotNKo, dotNKi, dotNNao, dotNNai,
            dotNClo, dotNCli, dotVoli, dotO
            ])


if __name__ == "__main__":
    from numba import jitclass
    from numba_baseclass import JITSPEC

    jit_Solver = jitclass(JITSPEC)(Wei)
    solver = jit_Solver((0.1,)*12, 0.1, 0.01)   # TODO: proper IC
    solver.solve()
    print(solver.solution)
    print(solver.time)