# Rede de pulsos — construir, de verdade

O passo depois dos scripts soltos em `~/workspace/neuronio-hh/`.
Aqui o LIF, a sinapse e a STDP viraram peças reutilizáveis, com teste
de regressão — em vez de um script novo para cada ideia.

## O que tem

```
rede_pulsos/
  lif.py          LIF: E_L=-65, limiar -50, reset -65, tau_m=10 ms, R=2, refratário 2 ms
  sinapse.py      s decai com tau_syn=8 ms; STDP A+=0.6, A-=0.4, tau=20 ms, w em [0, 30]
  rede.py         Rede: neurônios na ordem, sinapses entre eles, rodar() e resetar_ensaio()
  tarefa_bike.py  3 sensores (sol, calor, chuva) -> 1 decisão
testes/
  test_regressao.py   reproduz os números dos scripts soltos
plot_bike.py      gráfico V da decisão: dia bom x dia de chuva
rede_bike.png     o gráfico gerado
invariancia_rede.py   carga sináptica total Q, contínua x discreta
veto_custodia.py      veta só a aresta nociva, medida por ablação
homeostase_exp3.py    três freios no Exp 3, até 40 ensaios
veto_durante_treino.py  veto por custódia a cada 8 ensaios, até 80 ensaios
arestas_validas.py    definição C3 de aresta válida (v1.0)
```

## Como rodar

```bash
cd ~/workspace/rede-pulsos
python3 testes/test_regressao.py   # tem que dar tudo OK
python3 -m rede_pulsos.tarefa_bike
python3 plot_bike.py
```

## Resultado da tarefa da bicicleta (12/11/2026)

Do vídeo original: sol 1,0×0,8, calor 0,7×0,5, chuva 0×−1,5, viés −0,4.
Aqui cada entrada é um sensor que dispara em pulsos; a decisão é um LIF
que só recebe as sinapses. Pesos: sol 18, calor 12, chuva −30 (inibitória).
Corrente do sensor: I = 2 + 10×valor, de 20 a 120 ms. Regra: 3+ disparos
da decisão → "vai", senão "fica".

| cenário | sensores (disparos) | decisão | resultado |
|---|---|---|---|
| dia bom (sol, calor, sem chuva) | sol 8, calor 5 | 11 disparos | VAI |
| dia de chuva (sol, calor, chuva) | sol 8, calor 5, chuva 8 | 0 disparos | FICA |
| só sol | sol 8 | 7 disparos | VAI |
| só calor | calor 5 | 0 disparos | FICA |

Dois cuidados honestos, já no código:

- **O "+2" na corrente é um remendo, e está marcado como tal.** O LIF só
  dispara se I > 7,5, então I = 10×valor deixava o calor (I=7) com
  0 disparos — a rede decidiria só pelo sol, e o sensor de calor seria
  enfeite. Com I = 2 + 10×valor, calor vira I=9 e dispara 5 vezes.
- **A chuva é inibitória fixa, sem STDP.** Ela segura a decisão no zero
  mesmo com sol+calor disparando. Aprender esse peso sozinha é um próximo
  passo, não está feito.

## Teste de regressão — os números antigos batem

- 1 LIF: I=10 → 6 disparos (33,8…112,8 ms); I=2 → 0
- 2 LIFs: w=18 → pós 5 disparos; w=4 → pós 0
- STDP, 8 ensaios: w 14,00 → 24,57; pós de 2 para 5 disparos

## Treino com STDP — dias bons x chuvosos (13/11/2026)

Pergunta do Dan: treinar dias bons e chuvosos alternados e ver se a
chuva fica inibitória enquanto sol e calor sobem — memória específica,
não só ganho global? Arquivo: `treino_bike.py`.

Ponto de partida fraco: sol 10, calor 8, chuva −3.
Sem treino: dia bom 5 disparos → VAI; dia de chuva 3 → VAI (errado).

- **Variante A, chuva fixa em −3, STDP só em sol/calor, 16 ensaios:**
  sol 10 → 30, calor 8 → 30 (saturam no ensaio 7). Dia bom 20 VAI,
  dia de chuva 18 VAI, só calor 6 VAI. Ganho global, não memória.
- **Variante B, chuva com STDP inibitório de sinal corrigido:**
  chuva −3 → −26,41 (aprende, fica mais inibitória a cada dia de chuva
  em que o pós dispara por engano). Mas sol/calor também saturam em 30,
  e o dia de chuva cai só de 18 para 8 VAI — continua VAI, errado.
  Em 40 ensaios a chuva chega a −29,8 e o dia de chuva estabiliza
  em 6 VAI: cabo de guerra nos tetos, sem separação.

Por quê: sol e calor disparam antes do pós nos dois contextos — dia bom
(correto) e dia de chuva (errado) — então o STDP hebbiano os potencializa
nos dois. O pós disparar é o único professor, e ele não distingue os
contextos. Conclusão honesta: **não emerge espontaneamente**. Falta um
ingrediente: sinal de erro/recompensa (R-STDP) ou plasticidade
inibitória que fortaleça a chuva também quando ela segura o pós.

## As duas correções testadas (14/11/2026)

Arquivo: `treino_memoria_especifica.py`. Mesmos parâmetros do treino
original (pesos iniciais 10 / 8 / −3, A+ 0,6, A− 0,4, 16 ensaios
alternados). A `Sinapse` ganhou o modo `acumular_elegibilidade`, que
guarda o Δw do STDP e só aplica no fim do ensaio com a recompensa —
o STDP clássico e o treino original não mudaram (regressão passa).

- **Experimento 1 — R-STDP, recompensa global ±1:** falha feio. No dia
  de chuva errado a recompensa −1 reverte também o Δw da chuva, que
  estava tentando inibir — a chuva volta de −3 para 0 já no ensaio 8
  (apagada) e no ensaio 15 até o dia bom vira FICA. Em 40 ensaios tudo
  vira 0 FICA. Lição: recompensa global não distingue quem causou o
  erro de quem tentou impedi-lo — é o problema da atribuição de crédito.
- **Experimento 2 — plasticidade inibitória no silêncio:** nunca dispara.
  A regra só reforça a chuva se o pós ficar em silêncio no dia de chuva,
  e o pós nunca fica (mínimo 5 disparos no dia 2). Resultado idêntico à
  Variante B: chuva −26,41 em 16 ensaios, dia de chuva 8 VAI; em 40,
  −29,8 e 6 VAI. A regra só ajudaria se a rede já estivesse quase em
  silêncio — ovo e galinha.
- **Experimento 3 — R-STDP com atribuição de crédito (acréscimo da
  Alice, não estava nos dois desenhos):** a inibitória recebe a
  recompensa com o sinal trocado. Melhor resultado até aqui: em 16
  ensaios, sol 15,76, calor 9,95, chuva −10,1; dia bom 9 VAI, só sol 6
  VAI, só calor 0 FICA — mas o dia de chuva dá 3 VAI, errado por
  exatamente 1 disparo (limiar 3). Em 40 ensaios regride: chuva −30
  (teto), sol 29,17, calor 27,22, dia de chuva 5 VAI e só calor 5 VAI
  (errado). O cabo de guerra continua: o excitatório cresce sem teto
  útil e ultrapassa a inibição. Falta provavelmente homeostase — ou um
  teto excitatório — para a separação se manter.

## Invariância da rede — o peso r* (17/11/2026)

Arquivo: `invariancia_rede.py`. Carga sináptica total Q = ∫ I_sin dt no
neurônio de decisão, contínua (soma I_sin·dt) x discreta
(Σ w·τ_syn·(1−exp(−(T−t0)/τ_syn))). Dia bom, dt=0,1: Q = 1638,70
contínua x 1628,50 discreta (+0,6%, convergindo: +1,3% em dt=0,2 até
+0,1% em dt=0,02). Sem o w, Q = 103,76 — errado por um fator 15,79 e
sem convergir nunca: o análogo de esquecer o r*. A decisão é
invariante em dt (0,2 / 0,1 / 0,05 / 0,02) nos 4 cenários da bicicleta.

## Veto por custódia (17/11/2026)

Arquivo: `veto_custodia.py`. Ablata cada aresta dos 7 estados
congelados e veta só a nociva — aquela cuja ablação melhora o escore —
contra o veto por rótulo, que zera a aresta de maior |w|. A aresta
nociva é sempre o calor, em 5/7 estados (exceto inicial e Exp3-16, onde
ele é inerte, delta 0); a margem melhora +7/+9/+7/+4 e os acertos vão
de 2/4 para 3/4 em A16/A40 e de 2/4 para 4/4 em B16/B40/Exp3-40, sem
nunca piorar. O veto por rótulo piora a margem em 6/7 estados — o sol
é a aresta mais válida (delta −13 a −22); o único acerto dele, B40, é
coincidência (lá o maior |w| é o calor, 30,0 x sol 29,8). Limites
honestos: o VAI/FICA correto vem de fora; cada estado tem no máximo 1
aresta nociva, então o veto simultâneo é exato aqui (um grafo maior
precisaria de veto iterativo); arestas inertes ficam, então o ISC C3
continua 0,33.

## Homeostase no Exp 3 (17/11/2026)

Arquivo: `homeostase_exp3.py`. Três freios no R-STDP com crédito do
Exp 3, até 40 ensaios. Vencedor: **H1, STDP dependente do peso** — o
Δw que aumenta o peso é multiplicado por (30−|w|)/30; perto do teto a
potenciação some, enfraquecer continua normal. H1 em 40 ensaios: sol
18,13, calor 8,89, chuva −12,39; bom 10 VAI, chuva 2 FICA, só sol 7
VAI, só calor 0 FICA — **4/4**, e já em 16 ensaios também 4/4 (margem
9). É a primeira vez que a rede acerta tudo treinando sozinha, sem veto
depois. H2, teto excitatório em 20: 2/4 (o calor alcança o sol do mesmo
jeito). H3, normalização sol+calor=18: 3/4, margem 2 (fraca demais para
disparar só com o sol). Dois cuidados honestos: a margem sozinha diria
que o baseline é o melhor (22 contra 15 do H1), porque premia
disparar muito, não acertar — o que decide é acertos; e o H1 não é
para sempre: em 80 ensaios regride para 3/4 (chuva 4 VAI, sol 23,97).
Ele compra uma janela, de 16 a 40 ensaios com 4/4, não estabilidade
eterna. Números conferidos rodando o script de novo em 17/11.

## Veto durante o treino (17/11/2026)

Arquivo: `veto_durante_treino.py`. Exp 3 com um veto por custódia a
cada 8 ensaios: abla cada aresta com a rede congelada (medir não pode
treinar) e veta de vez só a nociva, congelando a aresta vetada (w=0 e
STDP desligado nela, para o STDP não fazer ela crescer de novo).

- **T1, veto com os 4 contextos:** o veto dispara uma vez só, no ensaio
  32 (calor delta +4), e a rede trava em sol 30, calor 0, chuva −25,27;
  bom 13 VAI, chuva 0 FICA, só sol 13 VAI, só calor 0 FICA — **4/4,
  margem 26, em 40 e em 80 ensaios, igualzinho.** Estável de verdade.
- **T2, H1 + veto:** o veto nunca dispara (o calor nunca fica nocivo
  com a homeostase) e fica igual ao H1 sozinho, 4/4 em 40, 3/4 em 80.
  A homeostase torna o veto redundante.
- **T3, veto só com bom+chuva:** nunca dispara e fica igual ao
  baseline, 2/4. Cuidado honesto: a nocividade do calor só aparece no
  contexto só calor, que não é ensaio de treino — o veto durante o
  treino depende de usar os contextos de teste (só sol, só calor) no
  escore da ablação. Isso é vazamento do teste para o treino, do mesmo
  tipo da recompensa do R-STDP, mas mais amplo, e está escrito no
  script, não escondido.

## Próximos passos — um de cada vez

1. **Treinar com STDP:** ligar STDP nas sinapses excitatórias e repetir
   dias bons e dias de chuva, para os pesos aprenderem sozinhos — em vez
   de eu ter fixado sol 18 / calor 12 / chuva −30 na mão.
   *Feito em 13/11, ver seção acima: não basta STDP puro.*
2. **Homeostase:** sem ela, o STDP aditivo satura w em 30 e a decisão
   dispara sem parar (já visto no teste de 2 LIFs).
   *Feito em 17/11 (H1, ver seção acima): 4/4 de 16 a 40 ensaios.
   Subiu para o GitHub em 17/11, com o relatório completo (seções
   de invariância, veto e homeostase neste README).*
3. **Arquitetura quântica sintética:** átomos como neurônios, acoplamento
   como sinapse — camada separada, não mistura com esta rede clássica.
4. **Veto durante o treino:** aplicar a custódia no meio do aprendizado,
   não só no fim.
   *Feito em 17/11 (T1, ver seção acima): 4/4 em 40 e em 80 ensaios.
   Subiu para o GitHub em 17/11, com o relatório completo.*

Os scripts soltos em `~/workspace/neuronio-hh/` não foram mexidos.
