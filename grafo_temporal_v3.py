"""Grafo temporal v0.3 — moldagem de recompensa: o laço pode
APRENDER a cruzar o limiar?

A pergunta que o v0.2 deixou: o laço nasce em 22 e resolve tudo,
mas a memória está na condição inicial, não no aprendizado. O v0.1
mostrou por que o treino não cruza o limiar sozinho: a recompensa
binária ±1 pune todo dia bom (sempre errado, 0 disparos no atraso)
e corrói o laço (8 -> 6,8). É uma armadilha de bootstrap.

A única mudança aqui é a recompensa — o resto é o v0.1 (mesma
arquitetura v0.2, mesma tarefa temporal, mesmo Exp 3 com crédito):
em vez de ±1 pelo acerto final, uma recompensa GRADUADA pela
atividade que a rede sustenta:

  A = n_h1 + n_h2 + 2 * n_decisao_no_atraso
      (h conta no ensaio inteiro; a decisão só no atraso)
  dia bom / só sol (alvo VAI):  r = min(A, 6) / 6
  chuva / só calor (alvo FICA): r = +1 se A == 0,
                                senão -min(A, 6) / 6

A ideia: no início, um dia bom tem h1 1x + h2 1x na pista e 0 no
atraso -> A = 2, r = +0,33. A rede já não é punida por não chegar
lá — é recompensada por se aproximar. A elegibilidade do laço vem
dos disparos coincidentes de h1/h2 na pista, por isso o sinal
positivo tem de o fazer crescer em vez de corroer. Nos dias de
chuva o h está calado (A = 0, r = +1), o que reforça a inibição.

O teto 6 é um parâmetro de moldagem, não um valor derivado: com
A inicial 2, dá r = 0,33 — um gradiente real, sem saturar. Se o
laço crescer, A cresce e r -> 1.

Como a taxa de crescimento por ensaio é pequena (o v0.1 corroeu
só 1,2 em 80 ensaios), o treino vai a 400 ensaios, com avaliação
em 16/40/80/160/400. Perguntas: o laço inverte a erosão e cresce?
Cruza o limiar 14-20? A decisão acende no atraso — e o ISC sobe,
ou a tarefa continua a resolver-se por um só caminho?

Rode: python3 grafo_temporal_v3.py
"""
import sys

sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

from rede_pulsos.tarefa_temporal import rodar_cenario_temporal
from grafo_temporal import (
    avaliar_temporal,
    diagnostico_inicial,
    teste_final_temporal,
)
from grafo_maior_v2 import montar_grafo_v2, pesos

TETO_MOLDAGEM = 6.0


def recompensa_moldada(r, vai_alvo):
    """r é o resultado de rodar_cenario_temporal. A é a atividade
    sustentada: h1+h2 no ensaio inteiro + decisão no atraso (x2)."""
    n_h = len(r["disparos"]["h1"]) + len(r["disparos"]["h2"])
    a = n_h + 2 * r["n_decisao"]
    if vai_alvo:
        return min(a, TETO_MOLDAGEM) / TETO_MOLDAGEM
    return 1.0 if a == 0 else -min(a, TETO_MOLDAGEM) / TETO_MOLDAGEM


def treinar_moldado(rede, inicio, fim):
    """Exp 3 com crédito, recompensa graduada, alternando dia bom /
    dia de chuva como no v0.1."""
    for i in range(inicio, fim + 1):
        bom = (i % 2 == 1)
        kw = dict(sol=1.0, calor=0.7, chuva=0.0) if bom else \
            dict(sol=1.0, calor=0.7, chuva=1.0)
        r = rodar_cenario_temporal(rede, **kw)
        recompensa = recompensa_moldada(r, vai_alvo=bom)
        for s in rede.sinapses:
            rec = -recompensa if s.inibitoria else recompensa
            s.aplicar_elegibilidade(rec)


def main():
    print("== Grafo temporal v0.3 (v0.1 + moldagem de recompensa) ==")
    rede = montar_grafo_v2()
    print(f"  início: {pesos(rede)}")
    diagnostico_inicial(rede)
    for inicio, fim in [(1, 16), (17, 40), (41, 80), (81, 160),
                        (161, 400)]:
        treinar_moldado(rede, inicio, fim)
        avaliar_temporal(rede, f"{fim} ensaios")
        print(f"  laço aos {fim}: h1->h2 "
              f"{[round(s.w, 2) for s in rede.sinapses if s.pre.nome == 'h1' and s.pos.nome == 'h2'][0]}, "
              f"h2->h1 "
              f"{[round(s.w, 2) for s in rede.sinapses if s.pre.nome == 'h2' and s.pos.nome == 'h1'][0]}")
    print(f"  pesos aos 400: {pesos(rede)}")


if __name__ == "__main__":
    main()
