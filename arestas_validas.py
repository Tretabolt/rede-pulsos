"""Aresta válida v1.0 — proposta da Alice (rascunho, para o Dan corrigir).

Pergunta aberta desde 14/11/2026: o que torna uma aresta "válida"?
Sem essa definição, o ISC topológico do Dan (arestas válidas / máximas)
não varia com o treino na estrela de 4 nós: é sempre 3/6 = 0,50, e o
critério B perde o poder de discriminar evolução legítima de redefinição
que legitima poder (ver simbolos_tensao.py §2-3).

Três candidatos, testados contra os estados reais do treino (13-14/11):
  C1 — piso de peso: válida se |w| >= w_min. Varre w_min e procura a
       faixa em que o ISC discrimina os estados.
  C2 — ablação (eficácia causal): válida se zerar a aresta muda a
       contagem de disparos em pelo menos um contexto de avaliação.
  C3 — eficácia específica (crédito): válida se zerar a aresta PIORA
       o escore de acerto da rede; NOCIVA se zerar MELHORA o escore;
       inerte se não muda nada. O escore é a soma das margens por
       contexto: contexto que deve dar VAI contribui (n - 3); contexto
       que deve dar FICA contribui (3 - n). Correto por contexto
       (treino_memoria_especifica.py, linha 26): bom VAI, chuva FICA,
       só sol VAI, só calor FICA.

Discriminação exigida: o ISC deve variar com o treino E não deve
premiar mero ganho global — a Variante A (sol/calor saturados, chuva
fraca, dia de chuva com 18 VAI) não pode parecer mais "integrada" que
a Variante B ou o Exp 3 (que tentam especificidade).

Achado metodológico (primeira rodada deste script): avaliar uma rede
com STDP ligado TREINA a rede — o "inicial" saiu da avaliação como
(13.1, 11.55, -3) em vez de (10, 8, -3), e repetir a ablação deu
números diferentes. Por isso cada estado é CONGELADO (stdp=False em
todas as sinapses) logo após o treino, antes de qualquer avaliação:
medir não pode ser treinar.

Rode: python3 arestas_validas.py
"""
import sys

sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

from treino_bike import montar, treinar, teste_final
from treino_memoria_especifica import montar_rstdp, treinar_credito

# Contexto -> decisão correta (True = VAI, False = FICA).
CORRETO = {"bom": True, "chuva": False, "só sol": True, "só calor": False}
ARESTAS_MAXIMAS = 6  # K4: 4 nós, 6 pares possíveis.


def congelar(rede):
    """Desliga o STDP: a partir daqui, avaliar não treina mais."""
    for s in rede.sinapses:
        s.stdp = False


def construir_estados():
    """Reconstrói cada estado treinando de verdade (determinístico) e
    congela logo em seguida."""
    estados = {}

    rede = montar()
    congelar(rede)
    estados["inicial"] = rede

    rede = montar(stdp_chuva=False)
    treinar(rede, ensaios=16)
    congelar(rede)
    estados["A 16 ensaios"] = rede

    rede = montar(stdp_chuva=False)
    treinar(rede, ensaios=40)
    congelar(rede)
    estados["A 40 ensaios"] = rede

    rede = montar(stdp_chuva=True)
    treinar(rede, ensaios=16)
    congelar(rede)
    estados["B 16 ensaios"] = rede

    rede = montar(stdp_chuva=True)
    treinar(rede, ensaios=40)
    congelar(rede)
    estados["B 40 ensaios"] = rede

    rede = montar_rstdp()
    treinar_credito(rede, ensaios=16)
    congelar(rede)
    estados["Exp3 16 ensaios"] = rede

    rede = montar_rstdp()
    treinar_credito(rede, ensaios=40)
    congelar(rede)
    estados["Exp3 40 ensaios"] = rede

    return estados


def pesos(rede):
    return {s.pre.nome: round(s.w, 2) for s in rede.sinapses}


def escore_acerto(resultado):
    """Margem total na direção certa + nº de contextos corretos."""
    margem = 0
    acertos = 0
    for nome, (n, vai) in resultado.items():
        certo = CORRETO[nome]
        margem += (n - 3) if certo else (3 - n)
        if vai == certo:
            acertos += 1
    return margem, acertos


def ablacao(rede):
    """Para cada aresta: zera o peso, reavalia, restaura. Devolve
    (mudou_contagem, delta_escore) por aresta. Rede congelada: sem
    deriva entre uma ablação e outra."""
    base = teste_final(rede)
    base_escore, _ = escore_acerto(base)
    out = {}
    for s in rede.sinapses:
        w0 = s.w
        s.w = 0.0
        r = teste_final(rede)
        s.w = w0
        mudou = any(r[c][0] != base[c][0] for c in base)
        esc, _ = escore_acerto(r)
        out[s.pre.nome] = (mudou, round(esc - base_escore, 2))
    return out


def main():
    estados = construir_estados()
    # Avalia cada estado UMA vez e guarda tudo: repetir a avaliação
    # sem congelar mudava os números (ver docstring).
    dados = {}
    for nome, rede in estados.items():
        r = teste_final(rede)
        margem, acertos = escore_acerto(r)
        dados[nome] = {"pesos": pesos(rede), "teste": r,
                       "margem": margem, "acertos": acertos,
                       "ablacao": ablacao(rede)}

    print("== 1. Estados reconstruídos (treino real, congelados) ==")
    for nome, d in dados.items():
        print(f"{nome}: pesos={d['pesos']}")
        print("   teste: " + "; ".join(
            f"{c}: {n} {'VAI' if v else 'FICA'}"
            for c, (n, v) in d["teste"].items()))
        print(f"   acertos {d['acertos']}/4, margem total {d['margem']}")

    print("\n== 2. C1 — piso de peso (|w| >= w_min): ISC = válidas/6 ==")
    for w_min in [3, 5, 8, 10, 12, 15, 20, 25]:
        print(f"  w_min={w_min:2d}: " + " | ".join(
            f"{nome}: {sum(1 for w in d['pesos'].values() if abs(w) >= w_min)}/6="
            f"{sum(1 for w in d['pesos'].values() if abs(w) >= w_min) / 6:.2f}"
            for nome, d in dados.items()))

    print("\n== 3. C2 — ablação: válida se zerar muda alguma contagem ==")
    print("== 4. C3 — eficácia específica: válida se zerar PIORA o escore; "
          "NOCIVA se zerar MELHORA; inerte se nada muda ==")
    for nome, d in dados.items():
        ab = d["ablacao"]
        c2_validas = [e for e, (mudou, _) in ab.items() if mudou]
        c3_validas = [e for e, (_, dl) in ab.items() if dl < 0]
        c3_nocivas = [e for e, (_, dl) in ab.items() if dl > 0]
        c3_inertes = [e for e, (_, dl) in ab.items() if dl == 0]
        print(f"{nome}:")
        print(f"   C2 válidas {len(c2_validas)}/3 {c2_validas} -> "
              f"ISC {len(c2_validas) / 6:.2f}")
        print(f"   C3 válidas {len(c3_validas)}/3 {c3_validas} -> "
              f"ISC {len(c3_validas) / 6:.2f} | nocivas {c3_nocivas} | "
              f"inertes {c3_inertes}")
        print("   deltas de escore ao ablar: " +
              "; ".join(f"{e}: {dl:+.0f}" for e, (_, dl) in ab.items()))

    print("\n== 5. Veredito de discriminação (calculado, não afirmado) ==")
    for w_min in [3, 5, 8, 10, 12, 15, 20, 25]:
        iscs = sorted({sum(1 for w in d["pesos"].values()
                           if abs(w) >= w_min) / 6 for d in dados.values()})
        print(f"  C1 w_min={w_min:2d}: ISC distintos entre estados: "
              f"{[round(x, 2) for x in iscs]}")
    for nome, d in dados.items():
        ab = d["ablacao"]
        c2 = sum(1 for _, (mudou, _) in ab.items() if mudou)
        c3 = sum(1 for _, (_, dl) in ab.items() if dl < 0)
        noc = sum(1 for _, (_, dl) in ab.items() if dl > 0)
        print(f"  {nome}: C2 ISC {c2 / 6:.2f} | C3 ISC {c3 / 6:.2f} | "
              f"nocivas {noc}")


if __name__ == "__main__":
    main()
