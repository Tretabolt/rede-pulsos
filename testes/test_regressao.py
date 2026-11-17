"""Regressão: a refatoração tem que reproduzir os números dos scripts soltos.

Referências (~/workspace/neuronio-hh/):
  simulacao_lif.py:      I=10 -> 6 disparos, I=2 -> 0
  dois_lif_sinapse.py:   w=18 -> pós 5 disparos, w=4 -> pós 0
  stdp_dois_lif.py:      w 14 -> 24.57 após 8 ensaios, pós 5 disparos no fim
Rode: python3 testes/test_regressao.py
"""
import sys
sys.path.insert(0, "/home/hatch/workspace/rede-pulsos")

from rede_pulsos import LIF, Sinapse, Rede


def estimulo(i_amp, t0=20.0, t1=120.0):
    return lambda t: i_amp if t0 <= t <= t1 else 0.0


def dois_lif(w, stdp=False, ensaios=1):
    rede = Rede()
    pre = rede.adicionar_neuronio(LIF("pre"))
    pos = rede.adicionar_neuronio(LIF("pos"))
    sin = rede.ligar(Sinapse(pre, pos, w, stdp=stdp))
    historico = []
    for _ in range(ensaios):
        rede.resetar_ensaio(manter_pesos=True)
        out = rede.rodar({"pre": estimulo(10.0)})
        historico.append((sin.w, out["pre"], out["pos"]))
    return historico


def main():
    # 1. Um LIF sozinho
    rede = Rede()
    n = rede.adicionar_neuronio(LIF("sozinho"))
    out = rede.rodar({"sozinho": estimulo(10.0)})
    assert len(out["sozinho"]) == 6, out
    assert out["sozinho"] == [33.8, 49.6, 65.4, 81.2, 97.0, 112.8], out
    rede.resetar_ensaio()
    out = rede.rodar({"sozinho": estimulo(2.0)})
    assert len(out["sozinho"]) == 0, out
    print("1 LIF sozinho: OK — I=10 dá 6 disparos, I=2 dá 0")

    # 2. Dois LIFs, sinapse forte e fraca
    hist = dois_lif(18.0)
    assert len(hist[0][2]) == 5, hist
    assert hist[0][2] == [51.9, 68.7, 84.8, 100.8, 116.6], hist
    hist = dois_lif(4.0)
    assert len(hist[0][2]) == 0, hist
    print("2 LIFs: OK — w=18 dá 5 disparos no pós, w=4 dá 0")

    # 3. STDP, 8 ensaios, w 14 -> ~24.57
    hist = dois_lif(14.0, stdp=True, ensaios=8)
    w_final = hist[-1][0]
    assert abs(w_final - 24.57) < 0.05, hist
    assert len(hist[-1][2]) == 5, hist
    assert len(hist[0][2]) == 2, hist
    print(f"STDP: OK — w 14.00 -> {w_final:.2f} em 8 ensaios, "
          f"pós de 2 para {len(hist[-1][2])} disparos")

    print("Tudo certo: a refatoração reproduz os scripts soltos.")


if __name__ == "__main__":
    main()
