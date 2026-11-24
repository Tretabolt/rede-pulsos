"""Pesos quantizados + homeostase H1 — a combinação que faltava.

Fio aberto do teste dos pesos quantizados (23/11, pesos_quantizados.py):
a unidade 3.75 chega a 4/4 em 16 ensaios, mas cai para 3/4 em 40 — a
quantização adia a regressão, não a cura. A homeostase H1
(homeostase_exp3.py, 17/11) segura 4/4 de 16 a 40 ensaios sozinha,
mas regride para 3/4 em 80. Pergunta: o acoplamento discreto (ideia
do Takahashi) COM o freio dependente do peso segura 4/4 até 80?

Desenho: Exp 3 (R-STDP com crédito) com dois interruptores
independentes — quantização (pesos só em múltiplos de 3.75, levados
à escada no fim de cada ensaio) e H1 (o Δw que aumenta |w| é
multiplicado por (30-|w|)/30, replicado de
homeostase_exp3.treinar_credito_homeostase). Quatro combinações,
mesmo início (10, 8, -3), treino contínuo até 80 ensaios com
avaliação congelada em 16, 40 e 80 (o congelamento é restaurado
depois de avaliar, como em pesos_quantizados.py, para o treino
continuar com STDP ligado).

Rode: python3 pesos_quantizados_homeostase.py
"""
import sys

sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

from rede_pulsos.tarefa_bike import rodar_cenario
from treino_bike import teste_final
from treino_memoria_especifica import montar_rstdp
from arestas_validas import ablacao, congelar, escore_acerto
from pesos_quantizados import quantizar, snap_rede

UNIDADES = [None, 3.75]  # None = contínuo
MODOS = [None, "H1"]


def pesos(rede):
    return {s.pre.nome: round(s.w, 2) for s in rede.sinapses}


def treinar(rede, inicio, fim, unidade=None, modo=None):
    """Treina os ensaios [inicio, fim], com H1 e/ou quantização."""
    for i in range(inicio, fim + 1):
        bom = (i % 2 == 1)
        kw = dict(sol=1.0, calor=0.7, chuva=0.0) if bom else \
            dict(sol=1.0, calor=0.7, chuva=1.0)
        r = rodar_cenario(rede, **kw)
        certo = (r["vai"] if bom else not r["vai"])
        recompensa = 1 if certo else -1
        for s in rede.sinapses:
            rec = -recompensa if s.inibitoria else recompensa
            if modo == "H1":
                dw_raw = rec * s.elegibilidade
                if dw_raw * s.w > 0:  # aumenta |w| — freio dependente do peso
                    dw_raw *= (30.0 - abs(s.w)) / 30.0
                s.w = float(min(max(s.w + dw_raw, s.w_min), s.w_max))
                s.elegibilidade = 0.0
            else:
                s.aplicar_elegibilidade(rec)
        snap_rede(rede, unidade)


def avaliar(rede, rotulo):
    flags = [s.stdp for s in rede.sinapses]
    congelar(rede)
    r = teste_final(rede)
    margem, acertos = escore_acerto(r)
    ab = ablacao(rede)
    nocivas = [e for e, (_, d) in ab.items() if d > 0]
    for s, f in zip(rede.sinapses, flags):
        s.stdp = f
    print(f"  {rotulo}: pesos={pesos(rede)}")
    print("    teste: " + "; ".join(
        f"{c}: {n} {'VAI' if v else 'FICA'}" for c, (n, v) in r.items()))
    print(f"    acertos {acertos}/4, margem {margem} | nocivas {nocivas}")
    return acertos, margem


def main():
    resultados = {}
    for modo in MODOS:
        for unidade in UNIDADES:
            rot = ("H1" if modo else "sem freio") + " + " + \
                  ("3.75" if unidade else "contínuo")
            print(f"\n== {rot} ==")
            rede = montar_rstdp()
            treinar(rede, 1, 16, unidade, modo)
            a16 = avaliar(rede, "16 ensaios")
            treinar(rede, 17, 40, unidade, modo)
            a40 = avaliar(rede, "40 ensaios")
            treinar(rede, 41, 80, unidade, modo)
            a80 = avaliar(rede, "80 ensaios")
            resultados[rot] = (a16, a40, a80)

    print("\n== Comparação (acertos/4 | margem) ==")
    for rot, (a16, a40, a80) in resultados.items():
        print(f"  {rot:22s} 16: {a16[0]}/4 | {a16[1]:3d}   "
              f"40: {a40[0]}/4 | {a40[1]:3d}   80: {a80[0]}/4 | {a80[1]:3d}")


if __name__ == "__main__":
    main()
