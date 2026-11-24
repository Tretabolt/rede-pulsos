"""Pesos quantizados — o artigo do Takahashi traduzido para a rede-pulsos.

Artigo: M. Takahashi, "Electron-phonon coupling strength in
hydrogen-bonded network crystals in the THz frequency range",
Chemical Physics Impact 11, 100977 (2025). O acoplamento elétron-fônon
A é quantizado: A = n · α · kB, com n = 1..8 — múltiplos inteiros de
uma unidade base (α ≈ 1/137, kB a constante de Boltzmann), em
cristais de ligação de hidrogênio (nicotinamida, tiamina, purina),
na faixa de 1 a 12 THz. Não replicado até 23/11/2026 (verificação
da Alice): trata-se aqui de uma HIPÓTESE DE DESENHO, não de um fato
estabelecido.

Tradução para cá: os pesos sinápticos só existem em múltiplos inteiros
de uma unidade w_unit — o acoplamento que persiste é discreto, como
no artigo. Unidade principal w_unit = 3.75: o teto 30 = 8 unidades,
espelhando n = 1..8 do Takahashi. Sensibilidade: 2.5 (mais fina, 12
níveis) e 5.0 (mais grossa, 6 níveis). O arredondamento é para o
múltiplo mais próximo, com meio para longe de zero.

Desenho: réplica exata do Experimento 3 (R-STDP com atribuição de
crédito, treino_memoria_especifica.treinar_credito), com um passo a
mais — depois de aplicar a elegibilidade no fim de cada ensaio, cada
peso é levado ao múltiplo mais próximo da unidade (dentro de
[w_min, w_max]). Os pesos iniciais NÃO são levados à escada: todas as
corridas partem do mesmo (10, 8, -3), e a quantização passa a valer a
partir da primeira atualização — sem isso, cada unidade começaria de
um ponto diferente e a comparação ficaria confundida. Nada em
treino_bike.py ou treino_memoria_especifica.py foi alterado.

Achado do diagnóstico (antes desta versão): com início levado à
escada, a unidade 2.5 ficava parada — mas o controle mostrou que era
o ponto inicial, não a unidade: partindo de (11.25, 7.5, -3.75), a
2.5 aprende (final 16 ensaios: 20.0, 7.5, -15.0) e a 5.0 também
(30, 20, -25). A quantização cria um limiar de arranque: o sol em
10 gera pouca elegibilidade por ensaio para cruzar meio degrau, e a
rede fica oscilando no mesmo lugar. É um resultado em si — degraus
precisam de um empurrão inicial — e o motivo desta versão usar um
início comum para todas as unidades.

Avaliação: igual a arestas_validas.py — rede congelada (stdp=False)
antes de avaliar (medir não pode treinar), teste_final nos 4
contextos, escore de acerto (margem total + nº de acertos) e vereditos
C3 por aresta (válida / nociva / inerte) via ablação.

Pergunta: a quantização preserva a separação do Exp 3 em 16 ensaios
e impede a regressão em 40 ensaios (saturação que torna o calor
nocivo), ou piora tudo?

Rode: python3 pesos_quantizados.py
"""
import math
import sys

sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

from rede_pulsos.tarefa_bike import rodar_cenario
from treino_bike import teste_final
from treino_memoria_especifica import montar_rstdp
from arestas_validas import ablacao, congelar, escore_acerto

UNIDADES = [None, 3.75, 5.0, 2.5]  # None = contínuo (linha de base)


def quantizar(w, unidade):
    """Múltiplo inteiro de `unidade` mais próximo; meio para longe de zero."""
    x = w / unidade
    n = math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)
    return n * unidade


def snap_rede(rede, unidade):
    """Leva todos os pesos à escada, respeitando [w_min, w_max]."""
    if unidade is None:
        return
    for s in rede.sinapses:
        w = quantizar(s.w, unidade)
        s.w = max(s.w_min, min(s.w_max, w))


def montar_q(unidade):
    # Sem snap inicial de propósito: todas as corridas partem do mesmo
    # (10, 8, -3); a escada vale a partir da primeira atualização.
    return montar_rstdp()


def treinar_credito_q(rede, ensaios=16, inicio=1, unidade=None):
    """Réplica de treinar_credito com o passo de quantização no fim de
    cada ensaio. Devolve (histórico, pesos finais)."""
    historico = []
    for i in range(inicio, inicio + ensaios):
        bom = (i % 2 == 1)
        kw = dict(sol=1.0, calor=0.7, chuva=0.0) if bom else \
            dict(sol=1.0, calor=0.7, chuva=1.0)
        r = rodar_cenario(rede, **kw)
        certo = (r["vai"] if bom else not r["vai"])
        recompensa = 1 if certo else -1
        for s in rede.sinapses:
            rec = -recompensa if s.inibitoria else recompensa
            s.aplicar_elegibilidade(rec)
        snap_rede(rede, unidade)
        historico.append((i, "bom" if bom else "chuva", r["n_decisao"],
                          r["vai"], certo, recompensa,
                          {s.pre.nome: round(s.w, 2) for s in rede.sinapses}))
    return historico


def pesos(rede):
    return {s.pre.nome: round(s.w, 2) for s in rede.sinapses}


def avaliar(rede, rotulo):
    # Congela só durante a avaliação e restaura depois: a continuação
    # do treino (17-40) precisa do STDP ligado, como no original.
    flags = [s.stdp for s in rede.sinapses]
    congelar(rede)
    r = teste_final(rede)
    margem, acertos = escore_acerto(r)
    ab = ablacao(rede)
    for s, f in zip(rede.sinapses, flags):
        s.stdp = f
    validas = [e for e, (_, d) in ab.items() if d < 0]
    nocivas = [e for e, (_, d) in ab.items() if d > 0]
    inertes = [e for e, (_, d) in ab.items() if d == 0]
    print(f"  {rotulo}: pesos={pesos(rede)}")
    print("    teste: " + "; ".join(
        f"{c}: {n} {'VAI' if v else 'FICA'}" for c, (n, v) in r.items()))
    print(f"    acertos {acertos}/4, margem {margem} | "
          f"C3 válidas {validas} | nocivas {nocivas} | inertes {inertes}")
    print("    deltas ao ablar: " +
          "; ".join(f"{e}: {d:+.0f}" for e, (_, d) in ab.items()))
    return {"pesos": pesos(rede), "acertos": acertos, "margem": margem,
            "validas": validas, "nocivas": nocivas, "inertes": inertes}


def main():
    print("== Exp 3 contínuo vs quantizado (R-STDP com crédito) ==")
    resultados = {}
    for unidade in UNIDADES:
        rot = "contínuo" if unidade is None else f"unidade {unidade}"
        print(f"\n-- {rot} --")
        rede = montar_q(unidade)
        print(f"  início comum: {pesos(rede)}")
        treinar_credito_q(rede, ensaios=16, unidade=unidade)
        r16 = avaliar(rede, "16 ensaios")
        treinar_credito_q(rede, ensaios=24, inicio=17, unidade=unidade)
        r40 = avaliar(rede, "40 ensaios")
        resultados[rot] = (r16, r40)

    print("\n== Comparação direta ==")
    print("  (acertos/4 | margem | nocivas)")
    for rot, (r16, r40) in resultados.items():
        print(f"  {rot:14s} 16: {r16['acertos']}/4 | margem "
              f"{r16['margem']:3d} | nocivas {r16['nocivas']}   ||   "
              f"40: {r40['acertos']}/4 | margem {r40['margem']:3d} | "
              f"nocivas {r40['nocivas']}")


if __name__ == "__main__":
    main()
