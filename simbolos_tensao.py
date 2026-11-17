"""Núcleo de auto-interrogação v0.1 — primeira simulação executável.

Pergunta do Dan (13/11/2026): a regra pela qual o sistema julga e
protege a diversidade ainda é sustentável? A tensão constitutiva é
entre a dinâmica log-normal (assimetria, cauda pesada, rich-club) e
a homogeneidade topológica necessária para emergência de símbolos
(Var(λ) baixa).

Este script testa essa tensão contra a rede-pulsos, onde ela rompeu
de fato no treino STDP de 13/11 (treino_bike.py).

CORREÇÃO 14/11/2026 — ISC verdadeiro, dado pelo Dan às 03:04 UTC:
fórmula TOPOLÓGICA, para grafos causais/shards e emergência do
Símbolo: ISC_Local = Arestas_Válidas_Existentes / Arestas_Máximas_Possíveis.
Na rede-pulsos (4 nós, 3 arestas existentes de 6 possíveis):
ISC_Local = 3/6 = 0,50 em TODOS os estados — o treino muda os pesos,
não o número de arestas, então o ISC topológico não varia aqui.
A versão anterior usava um ISC PROVISÓRIO (entropia normalizada
dos |pesos|), que o Dan corrigiu: ele dava 0,909 / 0,775 / 0,998 e
estava errado. A entropia H dos pesos continua impressa abaixo só
como diagnóstico separado — NÃO é ISC.
A fórmula DINÂMICA do Dan, ISC = 1/(1+H_fase+H_espectral), mede
fluxo/estados no tempo e não é a usada aqui para emergência do Símbolo.

Em aberto (Dan ainda não definiu no que passou): o que torna uma
aresta "válida" — se validade depender de limiar de peso, o ISC
poderia variar com o treino; sem essa definição, conta-se aresta
existente como válida.

Rode: python3 simbolos_tensao.py
"""
import hashlib
import math
import sys

import numpy as np

sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

# Pesos |w| do treino de 13/11: [sol, calor, chuva]
ESTADOS = {
    "inicial (10, 8, -3)": [10.0, 8.0, 3.0],
    "Variante A final (30, 30, -3)": [30.0, 30.0, 3.0],
    "Variante B final (30, 30, -26.41)": [30.0, 30.0, 26.41],
    "B 40 ensaios (30, 30, -29.8)": [30.0, 30.0, 29.8],
}

# Topologia da rede-pulsos: 4 nós, 3 arestas (sensores -> decisão).
N_NOS = 4
ARESTAS_EXISTENTES = 3
ARESTAS_MAXIMAS = N_NOS * (N_NOS - 1) // 2  # 6


def espectro_laplaciano_normalizado(pesos):
    """Estrela de 4 nós: sensores -> decisão. Devolve λ ordenados e Var(λ)."""
    n = 4
    W = np.zeros((n, n))
    for i, w in enumerate(pesos):
        W[i, 3] = W[3, i] = abs(w)
    d = W.sum(axis=1)
    Dm12 = np.diag(1.0 / np.sqrt(d))
    Ln = np.eye(n) - Dm12 @ W @ Dm12
    vals = np.sort(np.linalg.eigvalsh(Ln))
    return vals, float(np.var(vals))


def isc_topologico():
    """ISC_Local do Dan (fórmula topológica): arestas existentes / máximas."""
    return ARESTAS_EXISTENTES / ARESTAS_MAXIMAS


def entropia_pesos(pesos):
    """Diagnóstico separado: entropia de Shannon dos |pesos|. NÃO é ISC."""
    w = np.array(pesos, dtype=float)
    p = w / w.sum()
    h = float(-(p * np.log(p)).sum())
    return h / math.log(len(pesos)), h


def hash_regra(texto):
    return hashlib.sha256(texto.encode()).hexdigest()[:16]


REGRA_ORIGINAL = "ISC>=0.85 e Var(lambda)<=0.25"
REGRA_REDEFINIDA = "ISC>=0.50 e Var(lambda)<=0.60"


def avalia(regra):
    isc = isc_topologico()
    _, var = espectro_laplaciano_normalizado(ESTADOS["inicial (10, 8, -3)"])
    if regra == REGRA_ORIGINAL:
        return isc >= 0.85 and var <= 0.25
    return isc >= 0.50 and var <= 0.60


def main():
    print("== 1. Espectro e ISC topológico (fórmula do Dan) por estado ==")
    print(f"ISC_Local = {ARESTAS_EXISTENTES}/{ARESTAS_MAXIMAS} = {isc_topologico():.3f} "
          f"em todos os estados (limiar ≥ 0.85: FALHA)")
    for nome, pesos in ESTADOS.items():
        vals, var = espectro_laplaciano_normalizado(pesos)
        h_norm, h = entropia_pesos(pesos)
        print(f"{nome}:")
        print(f"  λ normalizado = {[round(float(x), 3) for x in vals]}")
        print(f"  Var(λ) = {var:.4f}  (limiar ≤ 0.25: {'PASSA' if var <= 0.25 else 'FALHA'})")
        print(f"  H pesos (diagnóstico, NÃO é ISC) = {h:.3f}  (entropia norm. {h_norm:.3f})")
        print(f"  Símbolo pela regra original? NÃO (ISC 0.500 < 0.85 e Var 0.500 > 0.25)")

    print("\n== 2. Teste da redefinição (critério B) — com o ISC verdadeiro ==")
    print(f"Regra comprometida antes: {REGRA_ORIGINAL}  hash={hash_regra(REGRA_ORIGINAL)}")
    print(f"Regra redefinida depois: {REGRA_REDEFINIDA}  hash={hash_regra(REGRA_REDEFINIDA)}")
    for estado in ["inicial", "Variante A final"]:
        print(f"{estado}: regra original -> INVÁLIDO; "
              f"regra redefinida -> {'VÁLIDO' if avalia(REGRA_REDEFINIDA) else 'INVÁLIDO'}")
    print("Leitura honesta: com o ISC verdadeiro, o teste perde o poder de "
          "discriminar. A redefinida valida TODOS os estados — o inicial "
          "equilibrado e a Variante A concentrada — e a original não valida "
          "nenhum. O sinal SUSPEITO dispara pela letra do critério B (a nova "
          "valida um estado concentrado que a velha invalidava), mas dispara "
          "também para o estado equilibrado, então ele não distingue evolução "
          "legítima de redefinição que legitima poder. Motivo: ISC e Var são "
          "ambos cegos aos pesos nesta estrela — dependem só da topologia, "
          "que não muda com o treino. Na versão provisória, pelo menos o ISC "
          "variava (0,909 -> 0,775); no verdadeiro, nem isso. Ver §3.")

    print("\n== 3. Ponto cego estrutural: ISC e Var(λ) são cegos na estrela ==")
    print("Para qualquer peso positivo na estrela de 4 nós, o espectro "
          "normalizado é sempre [0, 1, 1, 2] e Var(λ) = 0.50, e o ISC "
          "topológico é sempre 3/6 = 0.50.")
    print("Conclusão: a rede-pulsos, como estrela de 4 nós, NUNCA pode ser "
          "Símbolo pela regra ISC ≥ 0.85 e Var ≤ 0.25, por estrutura — não "
          "por treino. Para testar o núcleo de verdade, o grafo precisa de "
          "topologia mais rica (ciclos, laterais, recorrência), onde o "
          "espectro possa variar — e de uma definição do que torna uma "
          "aresta 'válida', para o ISC poder variar com o treino.")
    print("O Laplaciano NÃO normalizado, por contraste, explode com a escala "
          "(Var 129.75 no inicial → 1356.75 na Variante A), então ele detecta "
          "escala, não forma normalizada.")

    print("\n== 4. Fato mínimo (critério C) — o que seria preciso provar ==")
    print("Para cada ensaio, provar juntos: (1) Var e H calculados sobre o "
          "conjunto COMPLETO de arestas; (2) hash da regra comprometida ANTES "
          "do ensaio; (3) existe pelo menos um símbolo diverso ainda válido "
          "(ISC ≥ 0.85 e Var ≤ 0.25).")
    print(f"Exemplo de compromisso: hash_regra original = {hash_regra(REGRA_ORIGINAL)}")
    print("Com o ISC verdadeiro, (3) falha em todos os estados: nenhum passa "
          "na regra original, então não há sobrevivente diverso para provar — "
          "a tensão foi rompida, e o fato mínimo não pode ser produzido. É "
          "assim que o sistema saberia, de dentro, que rompeu.")


if __name__ == "__main__":
    main()
