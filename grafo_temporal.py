"""Grafo maior, tarefa temporal v0.1 — a recorrência é necessária?

Arquitetura: a do v0.2 (6 nós, 18 sinapses, 15 pares, teto de ISC
1,0 — laterais inibitórias + calor modulador). Tarefa: a temporal
(pista 20-60 ms, atraso 60-170 ms, decisão medida só no atraso).
Treino: Exp 3 com crédito, alternando dia bom / dia de chuva,
16/40/80 ensaios. Nada de escala ou veto — v0.1 testa a TAREFA,
não os mecanismos.

Perguntas: a rede aprende a sustentar a resposta no atraso (4/4)?
Os pares recorrentes (h1-h2, decisão-h1) viram válidos — o laço
importa de verdade — ou a tarefa continua resolvível sem eles?
O ISC sobe em direção a 0,85, ou o colapso se repete?

Rode: python3 grafo_temporal.py
"""
import sys

sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

from rede_pulsos.tarefa_bike import CENARIOS
from rede_pulsos.tarefa_temporal import rodar_cenario_temporal, PISTA_FIM
from grafo_maior import pares, PARES_MAX
from grafo_maior_v2 import montar_grafo_v2, pesos
from arestas_validas import congelar, escore_acerto


def treinar_temporal(rede, inicio, fim):
    """Exp 3 com crédito, sobre a tarefa temporal: alterna dia bom
    e dia de chuva; a recompensa vem do resultado no atraso."""
    for i in range(inicio, fim + 1):
        bom = (i % 2 == 1)
        kw = dict(sol=1.0, calor=0.7, chuva=0.0) if bom else \
            dict(sol=1.0, calor=0.7, chuva=1.0)
        r = rodar_cenario_temporal(rede, **kw)
        certo = (r["vai"] if bom else not r["vai"])
        recompensa = 1 if certo else -1
        for s in rede.sinapses:
            rec = -recompensa if s.inibitoria else recompensa
            s.aplicar_elegibilidade(rec)


def teste_final_temporal(rede):
    out = {}
    for nome, kw in [
        ("bom", dict(sol=1.0, calor=0.7, chuva=0.0)),
        ("chuva", dict(sol=1.0, calor=0.7, chuva=1.0)),
        ("só sol", dict(sol=1.0, calor=0.0, chuva=0.0)),
        ("só calor", dict(sol=0.0, calor=0.7, chuva=0.0)),
    ]:
        r = rodar_cenario_temporal(rede, **kw)
        out[nome] = (r["n_decisao"], r["vai"])
    return out


def ablacao_pares_temporal(rede):
    base_margem, _ = escore_acerto(teste_final_temporal(rede))
    resultado = {}
    for par, sins in sorted(pares(rede).items()):
        guardados = [s.w for s in sins]
        for s in sins:
            s.w = 0.0
        margem, _ = escore_acerto(teste_final_temporal(rede))
        delta = margem - base_margem
        veredito = "válido" if delta < 0 else \
            "nocivo" if delta > 0 else "inerte"
        resultado[par] = (delta, veredito)
        for s, w in zip(sins, guardados):
            s.w = w
    return resultado


def avaliar_temporal(rede, rotulo):
    flags = [s.stdp for s in rede.sinapses]
    congelar(rede)
    r = teste_final_temporal(rede)
    margem, acertos = escore_acerto(r)
    ab = ablacao_pares_temporal(rede)
    validos = [p for p, (_, v) in ab.items() if v == "válido"]
    nocivos = [p for p, (_, v) in ab.items() if v == "nocivo"]
    inertes = [p for p, (_, v) in ab.items() if v == "inerte"]
    isc = len(validos) / PARES_MAX
    for s, f in zip(rede.sinapses, flags):
        s.stdp = f
    print(f"  {rotulo}:")
    print("    teste: " + "; ".join(
        f"{c}: {n} {'VAI' if v else 'FICA'}" for c, (n, v) in r.items()))
    print(f"    acertos {acertos}/4, margem {margem} | "
          f"ISC_Local {len(validos)}/{PARES_MAX} = {isc:.3f} | "
          f"nocivos {len(nocivos)} | inertes {len(inertes)}")
    print("    pares: " + "; ".join(
        f"{a}-{b}: {d:+.0f} ({v})" for (a, b), (d, v) in ab.items()))
    return acertos, margem, isc


def diagnostico_inicial(rede):
    """Antes de treinar: o grafo sem treino já sustenta alguma coisa
    no atraso? Conta disparos na pista e no atraso, por neurônio."""
    for nome, kw in CENARIOS:
        r = rodar_cenario_temporal(rede, **kw)
        pista = {n: len([t for t in v if t < PISTA_FIM])
                 for n, v in r["disparos"].items()}
        atraso = {n: len([t for t in v if t >= PISTA_FIM])
                  for n, v in r["disparos"].items()}
        print(f"  {nome} inicial — pista: " +
              ", ".join(f"{n}:{v}" for n, v in pista.items()))
        print(f"    atraso: " +
              ", ".join(f"{n}:{v}" for n, v in atraso.items()))


def main():
    print("== Grafo temporal v0.1 (v0.2 + tarefa temporal) ==")
    rede = montar_grafo_v2()
    print(f"  início: {pesos(rede)}")
    diagnostico_inicial(rede)
    treinar_temporal(rede, 1, 16)
    a16 = avaliar_temporal(rede, "16 ensaios")
    treinar_temporal(rede, 17, 40)
    a40 = avaliar_temporal(rede, "40 ensaios")
    treinar_temporal(rede, 41, 80)
    a80 = avaliar_temporal(rede, "80 ensaios")
    print("\n== Resumo (acertos/4 | margem | ISC_Local) ==")
    for rot, a in [("16", a16), ("40", a40), ("80", a80)]:
        print(f"  {rot} ensaios: {a[0]}/4 | {a[1]:3d} | {a[2]:.3f}")
    print(f"  pesos aos 80: {pesos(rede)}")


if __name__ == "__main__":
    main()
