import math

import numpy as np

from numba import (
    jit,
    float64,
)


NOPYTHON = True


@jit(float64(float64), cache=True, nopython=NOPYTHON, nogil=True)
def gamma_func(voli):
    alpha = 4*math.pi*(3*voli/(4*math.pi))**(2/3)
    F = 96485.33
    return alpha/(F*voli)*1e-2


@jit(float64(float64, float64, float64, float64), cache=True, nopython=NOPYTHON, nogil=True)
def I_K_func(V, n, Ko, Ki):
    G_K = 25
    G_KL = 0.05
    E_K = 26.64*math.log(Ko/Ki)
    return G_K*n**4*(V - E_K) + G_KL*(V - E_K)


@jit(float64(float64, float64, float64, float64, float64), cache=True, nopython=NOPYTHON, nogil=True)
def I_Na_func(V, m, h, Nao, Nai):
    G_Na = 30
    G_NaL = 0.0247
    E_Na = 26.64*math.log(Nao/Nai)
    return G_Na*m**3*h*(V - E_Na) + G_NaL*(V - E_Na)


@jit(float64(float64, float64, float64), cache=True, nopython=NOPYTHON, nogil=True)
def I_Cl_func(V, Clo, Cli):
    G_ClL = 0.1
    E_Cl = 26.64*math.log(Cli/Clo)
    return G_ClL*(V - E_Cl)


@jit(float64(float64, float64, float64, float64), cache=True, nopython=NOPYTHON, nogil=True)
def I_pump_func(Nai, Ko, O, gamma):
    rho = 8e-1       # maximal pump rate
    p = rho/(1 + math.exp((20 - O)/3))/gamma
    return p/(1 + math.exp((25 - Nai)/3))*1/(1 + math.exp(3.5 - Ko))


@jit(float64(float64, float64[:], float64), nopython=True, cache=True, nogil=NOPYTHON)
def I(V, s, t):
    m, h, n, NKo, NKi, NNao, NNai, NClo, NCli, voli, O = s
    O = max(O, 0)
    vol = 1.4368e-15    # unit:m^3, when r=7 um,v=1.4368e-15 m^3
    beta0 = 7
    volo = (1 + 1/beta0)*vol - voli     # Extracellular volume
    C = 1

    Ko = NKo/volo       # mM
    Ki = NKi/voli
    Nao = NNao/volo
    Nai = NNai/voli
    Clo = NClo/volo
    Cli = NCli/voli

    I_K = I_K_func(V, n, Ko, Ki)
    I_Na = I_Na_func(V, m, h, Nao, Nai)
    I_Cl = I_Cl_func(V, Clo, Cli)
    gamma = gamma_func(voli)
    I_pump = I_pump_func(Nai, Ko, O, gamma)
    return -1*(I_Na + I_K + I_Cl + I_pump)/C 


@jit(float64[:](float64, float64[:], float64), nopython=NOPYTHON, cache=True, nogil=True)
def F(V, s, t):
    m, h, n, NKo, NKi, NNao, NNai, NClo, NCli, voli, O = s
    O = max(O, 0)

    G_K = 25       # Voltage gated conductance       [mS/mm^2]
    G_Na = 30       # Voltage gated conductance       [mS/mm^2]
    G_ClL = 0.1    # leak conductances               [mS/mm^2]
    G_Kl = 0.05
    G_NaL = 0.0247 
    C = 1         # Capacitance representing the lipid bilayer
    
    # Ion Concentration related Parameters
    dslp = 0.25     # potassium diffusion coefficient
    gmag = 5        # maximal glial strength
    sigma = 0.17    # Oxygen diffusion coefficient
    Ukcc2 = 3e-1      # maximal KCC2 cotransporteer trength
    Unkcc1 = 1e-1    # maximal KCC1 cotransporter strength
    rho = 8e-1       # maximal pump rate

    # Volume 
    vol = 1.4368e-15    # unit:m^3, when r=7 um,v=1.4368e-15 m^3
    beta0 = 7

    # Time Constant
    tau = 1e-3
    Kbath = 8.5
    Obath = 32

    gamma = gamma_func(voli)
    volo = (1 + 1/beta0)*vol - voli     # Extracellular volume
    beta = voli/volo                    # Ratio of intracelluar to extracelluar volume

    # Gating variables
    alpha_m = 0.32*(54 + V)/(1 - math.exp(-(V + 54)/4))
    beta_m = 0.28*(V + 27)/(math.exp((V + 27)/5) - 1)

    alpha_h = 0.128*math.exp(-(V + 50)/18)
    beta_h = 4/(1 + math.exp(-(V + 27)/5))

    alpha_n = 0.032*(V + 52)/(1 - math.exp(-(V + 52)/5))
    beta_n = 0.5*math.exp(-(V + 57)/40)

    dotm = alpha_m*(1 - m) - beta_m*m
    doth = alpha_h*(1 - h) - beta_h*h
    dotn = alpha_n*(1 - n) - beta_n*n

    Ko = NKo/volo       # mM
    Ki = NKi/voli
    Nao = NNao/volo
    Nai = NNai/voli
    Clo = NClo/volo
    Cli = NCli/voli

    fo = 1/(1 + math.exp((2.5 - Obath)/0.2))
    fv = 1/(1 + math.exp((beta - 20)/2))
    dslp *= fo*fv
    gmag *= fo

    p = rho/(1 + math.exp((20 - O)/3))/gamma
    I_glia = gmag/(1 + math.exp((18 - Ko)/2.5))
    Igliapump = (p/3/(1 + math.exp((25 - 18)/3)))*(1/(1 + math.exp(3.5 - Ko)))
    I_diff = dslp*(Ko - Kbath) + I_glia + 2*Igliapump*gamma

    I_K = I_K_func(V, n, Ko, Ki)
    I_Na = I_Na_func(V, m, h, Nao, Nai)
    I_Cl = I_Cl_func(V, Clo, Cli)
    I_pump = I_pump_func(Nai, Ko, O, gamma)

    # Cloride transporter (mM/s)
    fKo = 1/(1 + math.exp(16 - Ko))
    FKCC2 = Ukcc2*math.log((Ki*Cli)/(Ko*Clo))
    FNKCC1 = Unkcc1*fKo*(math.log((Ki*Cli)/(Ko*Clo)) + math.log((Nai*Cli)/(Nao*Clo)))

    dotNKo = tau*volo*(gamma*beta*(I_K -  2.0*I_pump) - I_diff + FKCC2*beta + FNKCC1*beta)
    dotNKi = -tau*voli*(gamma*(I_K - 2.0*I_pump) + FKCC2 + FNKCC1)

    dotNNao = tau*volo*beta*(gamma*(I_Na + 3.0*I_pump) + FNKCC1)
    dotNNai = -tau*voli*(+gamma*(I_Na + 3.0*I_pump) + FNKCC1)

    dotNClo = beta*volo*tau*(FKCC2 - gamma*I_Cl + 2*FNKCC1)
    dotNCli = voli*tau*(gamma*I_Cl - FKCC2 - 2*FNKCC1)

    r1 = vol/voli
    r2 = 1/beta0*vol/((1 + 1/beta0)*vol - voli)
    pii = Nai + Cli + Ki + 132*r1
    pio = Nao + Ko + Clo + 18*r2

    vol_hat = vol*(1.1029 - 0.1029*math.exp((pio - pii)/20))
    dotvoli = -(voli - vol_hat)/0.25*tau
    dotO = tau*(-5.3*(I_pump + Igliapump)*gamma + sigma*(Obath - O))

    return np.array((dotm, doth, dotn, dotNKo, dotNKi, dotNNao, dotNNai, dotNClo, dotNCli, dotvoli, dotO))

