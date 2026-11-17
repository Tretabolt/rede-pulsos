"""Sinapse com STDP — mesma regra do script stdp_dois_lif.py.

Quando o pré dispara, s salta +1 e depois decai com tau_syn = 8 ms.
A corrente que chega no pós é I_sin = w * s.

STDP clássico, par mais próximo (Bi & Poo), Δt = t_pós - t_pré:
  Δt > 0 (pré antes do pós): Δw = +A+ * exp(-Δt / tau+)
  Δt < 0 (pós antes do pré): Δw = -A- * exp(Δt / tau-)
  A+ = 0.6, A- = 0.4, tau+ = tau- = 20 ms, w limitado a [0, 30].
A+ maior que A- é um atalho para o efeito aparecer em 8 ensaios;
na biologia os dois são parecidos e o aprendizado é mais lento,
precisando de homeostase para não disparar (sem ela, w satura em 30).

Sinapse inibitória (inibitoria=True, w negativo): o sinal do STDP é
invertido, porque "fortalecer" aqui é ficar mais negativo.
  pré antes do pós (a inibição falhou em segurar o pós):
      Δw = -A+ * exp(-Δt / tau+)   # w fica mais negativo
  pós antes do pré:
      Δw = +A- * exp(Δt / tau-)    # w fica menos negativo
Sem essa inversão, o STDP clássico aplicado a w negativo enfraqueceria
a inibição justamente quando ela falha — o erro clássico de aplicar a
regra excitatória a uma sinapse inibitória.

R-STDP (acumular_elegibilidade=True): o Δw de cada par pré/pós não é
aplicado na hora; é somado em self.elegibilidade durante o ensaio e só
no fim, fora daqui, aplicar_elegibilidade(recompensa) faz
w += recompensa * elegibilidade, com recompensa +1 (acerto) ou -1 (erro).
A regra de Δw é a mesma do STDP clássico, incluindo a correção de sinal
da inibitória — só o momento da aplicação muda.
"""
import math

TAU_SYN = 8.0
A_PLUS = 0.6
A_MINUS = 0.4
TAU_PLUS = 20.0
TAU_MINUS = 20.0
W_MIN, W_MAX = 0.0, 30.0


class Sinapse:
    def __init__(self, pre, pos, w, tau_syn=TAU_SYN, stdp=False,
                 a_plus=A_PLUS, a_minus=A_MINUS,
                 tau_plus=TAU_PLUS, tau_minus=TAU_MINUS,
                 w_min=W_MIN, w_max=W_MAX, inibitoria=False,
                 acumular_elegibilidade=False):
        self.pre = pre          # neurônio pré-sináptico (objeto LIF)
        self.pos = pos          # neurônio pós-sináptico (objeto LIF)
        self.w = float(w)
        self.w_inicial = float(w)
        self.tau_syn = tau_syn
        self.stdp = stdp
        self.inibitoria = inibitoria
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_min = w_min
        self.w_max = w_max
        self.s = 0.0
        self.ultimo_pre = None
        self.ultimo_pos = None
        self.acumular_elegibilidade = acumular_elegibilidade
        self.elegibilidade = 0.0

    def corrente(self):
        return self.w * self.s

    def resetar(self, manter_peso=True):
        self.s = 0.0
        self.ultimo_pre = None
        self.ultimo_pos = None
        self.elegibilidade = 0.0
        if not manter_peso:
            self.w = self.w_inicial

    def _aplicar_dw(self, dw):
        """Aplica Δw na hora (STDP clássico) ou acumula na elegibilidade
        (R-STDP: a recompensa do fim do ensaio decide o sinal)."""
        if self.acumular_elegibilidade:
            self.elegibilidade += dw
        else:
            self.w += dw
            self.w = float(min(max(self.w, self.w_min), self.w_max))

    def aplicar_elegibilidade(self, recompensa):
        """R-STDP: w += recompensa * elegibilidade acumulada no ensaio.
        Recompensa +1 (acerto) ou -1 (erro). Zera a elegibilidade."""
        self.w += recompensa * self.elegibilidade
        self.w = float(min(max(self.w, self.w_min), self.w_max))
        self.elegibilidade = 0.0
        return self.w

    def apos_passo(self, pre_disparou, pos_disparou, dt, t):
        """Decai s, soma o salto do pré e aplica STDP. t é o tempo do passo."""
        self.s = self.s * math.exp(-dt / self.tau_syn)
        if pre_disparou:
            self.s += 1.0
        if not self.stdp:
            if pre_disparou:
                self.ultimo_pre = round(t + dt, 1)
            if pos_disparou:
                self.ultimo_pos = round(t + dt, 1)
            return
        # Depressão: o pós disparou pouco antes deste pré.
        if pre_disparou and self.ultimo_pos is not None:
            dt_par = self.ultimo_pos - (t + dt)  # negativo
            if -100.0 < dt_par < 0:
                dw = -self.a_minus * math.exp(dt_par / self.tau_minus)
                if self.inibitoria:
                    dw = -dw
                self._aplicar_dw(dw)
        if pre_disparou:
            self.ultimo_pre = round(t + dt, 1)
        # Potenciação: o pré disparou pouco antes deste pós.
        if pos_disparou and self.ultimo_pre is not None:
            dt_par = (t + dt) - self.ultimo_pre  # positivo (0 se mesmo passo)
            if 0 < dt_par < 100.0:
                dw = self.a_plus * math.exp(-dt_par / self.tau_plus)
                if self.inibitoria:
                    dw = -dw
                self._aplicar_dw(dw)
        if pos_disparou:
            self.ultimo_pos = round(t + dt, 1)
