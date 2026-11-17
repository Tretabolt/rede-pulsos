"""Treino com STDP: dias bons e chuvosos alternados.

Pergunta do Dan (13/11/2026): treinar dias bons e chuvosos e ver se o
peso da chuva fica inibitório enquanto sol e calor sobem. Se isso
acontecer espontaneamente, é memória específica, não só ganho global?

Ponto de partida fraco, para o aprendizado ter o que corrigir:
  sol 10, calor 8, chuva -3
  sem treino: dia bom 5 disparos -> VAI; dia de chuva 3 -> VAI (errado)

Variante A — STDP só nas excitatórias, chuva fixa em -3.
Variante B — STDP nas três, com STDP inibitório de sinal corrigido
  (ver Sinapse.inibitoria): pré antes do pós fortalece a inibição,
  w fica mais negativo. Sem a correção, o STDP clássico aplicado a
  w negativo enfraqueceria a inibição quando ela falha.

Rode: python3 treino_bike.py
"""
import sys
sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

from rede_pulsos.lif import LIF
from rede_pulsos.sinapse import Sinapse
from rede_pulsos.rede import Rede
from rede_pulsos.tarefa_bike import rodar_cenario

W_INICIAL = {"sol": 10.0, "calor": 8.0, "chuva": -3.0}


def montar(stdp_chuva=False):
    rede = Rede()
    sol = rede.adicionar_neuronio(LIF("sol"))
    calor = rede.adicionar_neuronio(LIF("calor"))
    chuva = rede.adicionar_neuronio(LIF("chuva"))
    decisao = rede.adicionar_neuronio(LIF("decisao"))
    rede.ligar(Sinapse(sol, decisao, W_INICIAL["sol"], stdp=True))
    rede.ligar(Sinapse(calor, decisao, W_INICIAL["calor"], stdp=True))
    rede.ligar(Sinapse(chuva, decisao, W_INICIAL["chuva"], stdp=stdp_chuva,
                       inibitoria=True, w_min=-30.0, w_max=0.0))
    return rede


def pesos(rede):
    return {s.pre.nome: round(s.w, 2) for s in rede.sinapses}


def treinar(rede, ensaios=16):
    historico = []
    for i in range(ensaios):
        bom = (i % 2 == 0)
        kw = dict(sol=1.0, calor=0.7, chuva=0.0) if bom else \
            dict(sol=1.0, calor=0.7, chuva=1.0)
        r = rodar_cenario(rede, **kw)
        historico.append((i + 1, "bom" if bom else "chuva",
                          r["n_decisao"], r["vai"], dict(pesos(rede))))
    return historico


def teste_final(rede):
    out = {}
    for nome, kw in [
        ("bom", dict(sol=1.0, calor=0.7, chuva=0.0)),
        ("chuva", dict(sol=1.0, calor=0.7, chuva=1.0)),
        ("só sol", dict(sol=1.0, calor=0.0, chuva=0.0)),
        ("só calor", dict(sol=0.0, calor=0.7, chuva=0.0)),
    ]:
        r = rodar_cenario(rede, **kw)
        out[nome] = (r["n_decisao"], r["vai"])
    return out


def main():
    for rotulo, stdp_chuva in [("A — chuva fixa", False),
                               ("B — chuva com STDP inibitório", True)]:
        rede = montar(stdp_chuva=stdp_chuva)
        print(f"\n== Variante {rotulo} ==")
        print(f"início: {pesos(rede)}")
        for n, tipo, nd, vai, p in treinar(rede):
            print(f"  ensaio {n:2d} {tipo:5s} decisão={nd:2d} "
                  f"{'VAI' if vai else 'FICA'} pesos={p}")
        print(f"final: {pesos(rede)}")
        for nome, (nd, vai) in teste_final(rede).items():
            print(f"  {nome}: {nd} {'VAI' if vai else 'FICA'}")


if __name__ == "__main__":
    main()
