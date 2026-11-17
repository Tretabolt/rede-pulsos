"""Rede de LIFs: neurônios na ordem em que foram adicionados, sinapses entre eles.

Ordem de um passo (igual à dos scripts soltos, para os números baterem):
  1. cada neurônio, na ordem, recebe a corrente externa + a soma das
     correntes das sinapses que chegam nele (com o s e o w do momento);
  2. depois que todos dispararam (ou não), cada sinapse decai, salta
     se o pré disparou e aplica STDP se estiver ligado.
"""


class Rede:
    def __init__(self, dt=0.1, t_max=150.0, registrar=False):
        self.dt = dt
        self.t_max = t_max
        self.registrar = registrar
        self.neuronios = {}   # nome -> LIF, na ordem de inserção
        self.sinapses = []    # lista de Sinapse
        self.tempos = []      # tempos registrados (só se registrar=True)
        self.traces = {}      # nome -> lista de V por passo

    def adicionar_neuronio(self, neuronio):
        self.neuronios[neuronio.nome] = neuronio
        return neuronio

    def ligar(self, sinapse):
        self.sinapses.append(sinapse)
        return sinapse

    def _corrente_sinaptica(self, nome):
        return sum(s.corrente() for s in self.sinapses if s.pos.nome == nome)

    def resetar_ensaio(self, manter_pesos=True):
        for n in self.neuronios.values():
            n.resetar()
        for s in self.sinapses:
            s.resetar(manter_peso=manter_pesos)

    def rodar(self, correntes, t_max=None):
        """correntes: dict nome -> função t -> I_ext, ou valor fixo.
        Devolve dict nome -> lista de tempos de disparo."""
        t_max = self.t_max if t_max is None else t_max
        n = int(t_max / self.dt)
        if self.registrar:
            self.tempos = []
            self.traces = {nome: [] for nome in self.neuronios}
        for i in range(n):
            t = i * self.dt
            disparou = {}
            for nome, neuronio in self.neuronios.items():
                c = correntes.get(nome, 0.0)
                i_ext = c(t) if callable(c) else c
                i_total = i_ext + self._corrente_sinaptica(nome)
                disparou[nome] = neuronio.passo(i_total, self.dt, t)
            for s in self.sinapses:
                s.apos_passo(disparou[s.pre.nome], disparou[s.pos.nome], self.dt, t)
            if self.registrar:
                self.tempos.append(t)
                for nome, neuronio in self.neuronios.items():
                    self.traces[nome].append(neuronio.v)
        return {nome: list(n.disparos) for nome, n in self.neuronios.items()}
