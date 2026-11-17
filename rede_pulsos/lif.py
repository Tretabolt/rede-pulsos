"""Neurônio LIF — os mesmos parâmetros dos scripts soltos em ~/workspace/neuronio-hh/.

Uma só equação: tau_m * dV/dt = -(V - E_L) + R_m * I
Repouso E_L = -65 mV, limiar -50 mV, reset -65 mV, tau_m = 10 ms,
R_m = 2, refratário 2 ms. O pico é desenhado como +30 mV só para o gráfico;
no modelo, o disparo é instantâneo.
"""

E_L = -65.0
V_TH = -50.0
V_RESET = -65.0
TAU_M = 10.0
R_M = 2.0
TAU_REF = 2.0
V_SPIKE_PLOT = 30.0


class LIF:
    def __init__(self, nome, e_l=E_L, v_th=V_TH, v_reset=V_RESET,
                 tau_m=TAU_M, r_m=R_M, tau_ref=TAU_REF):
        self.nome = nome
        self.e_l = e_l
        self.v_th = v_th
        self.v_reset = v_reset
        self.tau_m = tau_m
        self.r_m = r_m
        self.tau_ref = tau_ref
        self.v = e_l
        self.ref = 0
        self.disparos = []  # tempos (ms) dos disparos neste ensaio

    def resetar(self):
        self.v = self.e_l
        self.ref = 0
        self.disparos = []

    def passo(self, i_ext, dt, t):
        """Avança um passo dt. Devolve True se disparou neste passo."""
        if self.ref > 0:
            self.v = self.v_reset
            self.ref -= 1
            return False
        v_next = self.v + dt * (-(self.v - self.e_l) + self.r_m * i_ext) / self.tau_m
        if v_next >= self.v_th:
            self.disparos.append(round(t + dt, 1))
            self.v = V_SPIKE_PLOT
            self.ref = int(self.tau_ref / dt)
            return True
        self.v = v_next
        return False
