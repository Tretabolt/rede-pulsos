"""Grafo maior v0.1 — sair da estrela.

A estrela (3 sensores -> 1 decisão) tem teto estrutural: ISC_Local
máximo 3/6 = 0,50, e nenhum ciclo — a memória vive só nos pesos.
Este é o primeiro grafo com recorrência: 6 nós (sol, calor, chuva,
h1, h2, decisão), 14 sinapses dirigidas, 13 pares não-dirigidos de
15 possíveis — teto de ISC 13/15 ≈ 0,867, acima do limiar 0,85.

Arquitetura:
  sensores -> h1, h2 (12 cada; chuva inibitória -8 nos dois)
  h1, h2 -> decisão (12 cada)
  h1 <-> h2 (8 cada; o par recorrente — o ciclo)
  sensores -> decisão direto (5, 5, -3; o caminho da estrela,
    enfraquecido de propósito: a rota principal agora passa por H)
  decisão -> h1 (-5, inibitória, FIXA sem STDP; o freio que fecha
    o laço: a decisão segura a camada oculta quando dispara)

CORREÇÃO (24/11, após a primeira rodada): são 12 pares, não 13 —
a sinapse decisão->h1 divide o par {h1, decisão} com h1->decisão.
Teto de ISC 12/15 = 0,80, ABAIXO do limiar 0,85: esta arquitetura
não pode chegar lá nem com todos os pares válidos. Para 0,85 é
preciso mais pares — as laterais entre sensores (3 pares, teto 1,0)
são o passo seguinte, não um detalhe.

Treino: Exp 3 com atribuição de crédito (R-STDP, recompensa ±1,
inibitória com sinal trocado), 16/40/80 ensaios, mesmo início para
todas as corridas. Nada de quantização, escala ou veto aqui — v0.1
testa o GRAFO, não os mecanismos; eles entram em v0.2.

Avaliação (rede congelada, como sempre): teste_final nos 4
contextos, e ablação POR PAR (zera as duas direções do par):
válido se o escore piora, nocivo se melhora, inerte se não muda.
ISC_Local = pares válidos / 15 (todos os pares possíveis entre
6 nós). Um par é o que conta, não a direção isolada — coerente
com a estrela, onde cada par tinha uma só direção.

Perguntas: com recorrência + feedback, a rede chega a ISC >= 0,85
com 4/4 na bicicleta? Os pares recorrentes (h1-h2, decisão-h1)
ganham validade — o laço importa — ou ficam inertes?

Rode: python3 grafo_maior.py
"""
import sys

sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

from rede_pulsos.lif import LIF
from rede_pulsos.sinapse import Sinapse
from rede_pulsos.rede import Rede
from rede_pulsos.tarefa_bike import rodar_cenario
from treino_bike import teste_final
from arestas_validas import congelar, escore_acerto

PARES_MAX = 15  # 6 nós: 6*5/2 pares possíveis


def montar_grafo():
    rede = Rede()
    sol = rede.adicionar_neuronio(LIF("sol"))
    calor = rede.adicionar_neuronio(LIF("calor"))
    chuva = rede.adicionar_neuronio(LIF("chuva"))
    h1 = rede.adicionar_neuronio(LIF("h1"))
    h2 = rede.adicionar_neuronio(LIF("h2"))
    decisao = rede.adicionar_neuronio(LIF("decisao"))

    def lig(pre, pos, w, **kw):
        padrao = dict(stdp=True, acumular_elegibilidade=True)
        padrao.update(kw)
        rede.ligar(Sinapse(pre, pos, w, **padrao))

    # sensores -> camada oculta
    for h in (h1, h2):
        lig(sol, h, 12.0)
        lig(calor, h, 12.0)
        lig(chuva, h, -8.0, inibitoria=True, w_min=-30.0, w_max=0.0)
    # camada oculta -> decisão
    lig(h1, decisao, 12.0)
    lig(h2, decisao, 12.0)
    # par recorrente
    lig(h1, h2, 8.0)
    lig(h2, h1, 8.0)
    # caminho direto da estrela, enfraquecido
    lig(sol, decisao, 5.0)
    lig(calor, decisao, 5.0)
    lig(chuva, decisao, -3.0, inibitoria=True, w_min=-30.0, w_max=0.0)
    # feedback fixo: a decisão freia h1
    rede.ligar(Sinapse(decisao, h1, -5.0, stdp=False,
                       inibitoria=True, w_min=-30.0, w_max=0.0))
    return rede


def pesos(rede):
    return {f"{s.pre.nome}->{s.pos.nome}": round(s.w, 2)
            for s in rede.sinapses}


def treinar_credito(rede, inicio, fim):
    """Réplica do treinar_credito do Exp 3, sobre o grafo maior."""
    for i in range(inicio, fim + 1):
        bom = (i % 2 == 1)
        kw = dict(sol=1.0, calor=0.7, chuva=0.0) if bom else \
            dict(sol=1.0, calor=0.7, chuva=1.0)
        r = rodar_cenario(rede, **kw)
        certo = (r["vai"] if bom else not r["vai"])
        recompensa = 1 if certo else -1
        for s in rede.sinapses:
            rec = -recompensa if s.inibitoria else recompensa
            s.aplicar_elegibilidade(rec)


def pares(rede):
    """Pares não-dirigidos presentes: (a, b) com a < b por nome."""
    out = {}
    for s in rede.sinapses:
        a, b = sorted((s.pre.nome, s.pos.nome))
        out.setdefault((a, b), []).append(s)
    return out


def ablacao_pares(rede):
    """Para cada par: zera as duas direções, reavalia, mede o delta
    do escore. Devolve {par: (delta, veredito)}."""
    base_margem, _ = escore_acerto(teste_final(rede))
    resultado = {}
    for par, sins in sorted(pares(rede).items()):
        guardados = [s.w for s in sins]
        for s in sins:
            s.w = 0.0
        margem, _ = escore_acerto(teste_final(rede))
        delta = margem - base_margem
        veredito = "válido" if delta < 0 else \
            "nocivo" if delta > 0 else "inerte"
        resultado[par] = (delta, veredito)
        for s, w in zip(sins, guardados):
            s.w = w
    return resultado


def avaliar(rede, rotulo):
    flags = [s.stdp for s in rede.sinapses]
    congelar(rede)
    r = teste_final(rede)
    margem, acertos = escore_acerto(r)
    ab = ablacao_pares(rede)
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
    """Antes de treinar: cada neurônio dispara no dia bom?"""
    r = rodar_cenario(rede, sol=1.0, calor=0.7, chuva=0.0)
    print("  dia bom inicial, disparos por neurônio: " +
          ", ".join(f"{n}: {len(v)}" for n, v in r["disparos"].items()))
    r = rodar_cenario(rede, sol=1.0, calor=0.7, chuva=1.0)
    print("  dia de chuva inicial, disparos por neurônio: " +
          ", ".join(f"{n}: {len(v)}" for n, v in r["disparos"].items()))


def main():
    print("== Grafo maior v0.1 (6 nós, 14 sinapses, 12 pares) ==")
    rede = montar_grafo()
    print(f"  início: {pesos(rede)}")
    diagnostico_inicial(rede)
    treinar_credito(rede, 1, 16)
    a16 = avaliar(rede, "16 ensaios")
    treinar_credito(rede, 17, 40)
    a40 = avaliar(rede, "40 ensaios")
    treinar_credito(rede, 41, 80)
    a80 = avaliar(rede, "80 ensaios")
    print("\n== Resumo (acertos/4 | margem | ISC_Local) ==")
    for rot, a in [("16", a16), ("40", a40), ("80", a80)]:
        print(f"  {rot} ensaios: {a[0]}/4 | {a[1]:3d} | {a[2]:.3f}")


if __name__ == "__main__":
    main()
