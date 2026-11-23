# Semana 10 · Pronosticando la NFL con un modelo de ELO

## ¿Cómo va el modelo? (semanas 1–9 · pick si prob. > 50%)

- Récord del modelo en la temporada (aciertos-fallos-empates): **78-57-1 (57.7%)**
- Semana 9: 10-5 (66.7%)
- Aciertos que el propio modelo esperaba: 85.4 → reales: 78.5 (empate = ½)
- Brier: 0.225 (moneda al aire = 0.250) · Log loss: 0.643 (moneda = 0.693)
- Cuando eligió al local: 49-39-1 · al visitante: 29-18
- En esos mismos partidos, siempre el local: 67-68-1
- Favorito de Las Vegas: 92-43-1 (68.0%) · el modelo coincide con Las Vegas en 80.9% de los picks · Brier Las Vegas: 0.214
- El batacazo de la semana 9: **GB** ganó con solo 35% de probabilidad (GB 45–27 NE)

### Calibración (temporada)

| Confianza | Partidos | Récord | Prob. media | % real |
|---|---|---|---|---|
| 50–60% | 58 | 25-32-1 | 55% | 44% |
| 60–70% | 46 | 26-20 | 65% | 57% |
| 70–80% | 30 | 25-5 | 75% | 83% |
| 80%+ | 2 | 2-0 | 81% | 100% |

## Modelo 1 (Elo) vs Modelo 2 (Elo + ausencias)

Pronósticos congelados antes de cada kickoff · semanas 1–9 · 136 partidos · pick si prob. > 50%

| Modelo | Récord | % aciertos | Brier | Log loss |
|---|---|---|---|---|
| Modelo 2 · Elo + ausencias | 88-47-1 | 65.1% | 0.218 | 0.630 |
| Modelo 1 · Elo | 78-57-1 | 57.7% | 0.225 | 0.643 |
| Las Vegas (referencia) | — | 68.0% | 0.214 | — |

- Mejor Brier hasta ahora: **Modelo 2 · Elo + ausencias** (más bajo = mejor).
- Eligieron distinto favorito en 12 partidos → acertó el Modelo 1 en 1 y el Modelo 2 en 11.
- Semanas 1–9: reconstruidas (backfill) con el reporte final de lesiones de cada semana.

### Semana 10: dónde difieren (ausencias según provisional: reporte sem. 9)

- LAC @ BAL: Modelo 1 BAL 67% · Modelo 2 BAL 59%
    - BAL (-80 Elo): Lamar Jackson (QB, inicia Tyler Huntley)
- WAS @ NYG: Modelo 1 NYG 67% · Modelo 2 NYG 61%
    - NYG (-82 Elo): Jaxson Dart (QB, inicia J.J. McCarthy)
- SEA @ LV: Modelo 1 SEA 69% · Modelo 2 SEA 64%
    - SEA (-50 Elo): Sam Darnold (QB, Out sem. 9)
- KC @ ATL: Modelo 1 KC 58% · Modelo 2 KC 54%
- LA @ ARI: Modelo 1 LA 65% · Modelo 2 LA 69%
    - LA (-43 Elo): Davante Adams (WRTE, IR)
    - ARI (-82 Elo): Jacoby Brissett (QB, inicia Gardner Minshew)
- CAR @ NO: Modelo 1 NO 56% · Modelo 2 NO 58%

_Ojo: el reporte de lesiones de esta semana aún no sale. Vuelve a correr `python3 src/weekly_content.py --solo-pronosticos` el viernes/sábado para congelar la versión final del Modelo 2._

## Picks de la semana 10 (hora CDMX)

- jue 12 nov 19:15 · WAS @ NYG → **NYG 67%**
- dom 15 nov 08:30 · NE @ DET → **DET 58%**
- dom 15 nov 12:00 · BUF @ NYJ → **BUF 83%**
- dom 15 nov 12:00 · CAR @ NO → **NO 56%**
- dom 15 nov 12:00 · HOU @ CLE → **HOU 51%**
- dom 15 nov 12:00 · JAX @ TEN → **JAX 78%**
- dom 15 nov 12:00 · KC @ ATL → **KC 58%**
- dom 15 nov 12:00 · MIA @ IND → **IND 57%**
- dom 15 nov 12:00 · MIN @ GB → **GB 54%**
- dom 15 nov 15:05 · LA @ ARI → **LA 65%**
- dom 15 nov 15:05 · SEA @ LV → **SEA 69%**
- dom 15 nov 15:25 · SF @ DAL → **SF 68%**
- dom 15 nov 19:20 · PIT @ CIN → **CIN 51%**
- lun 16 nov 19:15 · LAC @ BAL → **BAL 67%**

## Destacados

- **El más claro** · BUF @ NYJ: El favorito más claro de la semana: BUF gana con 82.9%
- **La moneda al aire** · HOU @ CLE: Solo 2.0 puntos de diferencia: el partido más parejo de la semana
- **La sorpresa** · LA @ ARI: Las Vegas ve muy favorito a LA (78.2%). El modelo lo ve más parejo: 65.0%.
