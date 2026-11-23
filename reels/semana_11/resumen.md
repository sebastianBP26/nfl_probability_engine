# Semana 11 · Pronosticando la NFL con un modelo de ELO

## ¿Cómo va el modelo? (semanas 1–10 · pick si prob. > 50%)

- Récord del modelo en la temporada (aciertos-fallos-empates): **87-62-1 (58.3%)**
- Semana 10: 9-5 (64.3%)
- Aciertos que el propio modelo esperaba: 94.3 → reales: 87.5 (empate = ½)
- Brier: 0.223 (moneda al aire = 0.250) · Log loss: 0.639 (moneda = 0.693)
- Cuando eligió al local: 52-43-1 · al visitante: 35-19
- En esos mismos partidos, siempre el local: 71-78-1
- Favorito de Las Vegas: 101-47-1 (68.1%) · el modelo coincide con Las Vegas en 82.6% de los picks · Brier Las Vegas: 0.211
- El batacazo de la semana 10: **LAC** ganó con solo 33% de probabilidad (LAC 23–20 BAL)

### Calibración (temporada)

| Confianza | Partidos | Récord | Prob. media | % real |
|---|---|---|---|---|
| 50–60% | 65 | 29-35-1 | 55% | 45% |
| 60–70% | 51 | 29-22 | 65% | 57% |
| 70–80% | 31 | 26-5 | 75% | 84% |
| 80%+ | 3 | 3-0 | 82% | 100% |

## Modelo 1 (Elo) vs Modelo 2 (Elo + ausencias)

Pronósticos congelados antes de cada kickoff · semanas 1–10 · 150 partidos · pick si prob. > 50%

| Modelo | Récord | % aciertos | Brier | Log loss |
|---|---|---|---|---|
| Modelo 2 · Elo + ausencias | 97-52-1 | 65.0% | 0.216 | 0.626 |
| Modelo 1 · Elo | 87-62-1 | 58.3% | 0.223 | 0.639 |
| Las Vegas (referencia) | — | 68.1% | 0.211 | — |

- Mejor Brier hasta ahora: **Modelo 2 · Elo + ausencias** (más bajo = mejor).
- Eligieron distinto favorito en 12 partidos → acertó el Modelo 1 en 1 y el Modelo 2 en 11.
- Semanas 1–9: reconstruidas (backfill) con el reporte final de lesiones de cada semana.

### Semana 11: dónde difieren (ausencias según provisional: reporte sem. 10)

- PIT @ PHI: Modelo 1 PIT 63% · Modelo 2 PIT 74%
    - PHI (-100 Elo): Jalen Hurts (QB, inicia Joe Milton III); Dontayvion Wicks (WRTE, Out sem. 9)
- ARI @ KC: Modelo 1 KC 76% · Modelo 2 KC 85%
    - ARI (-110 Elo): Jacoby Brissett (QB, inicia Gardner Minshew); Michael Wilson (WRTE, IR)
- BAL @ CAR: Modelo 1 CAR 57% · Modelo 2 CAR 65%
    - BAL (-82 Elo): Lamar Jackson (QB, inicia Tyler Huntley)
- JAX @ NYG: Modelo 1 JAX 62% · Modelo 2 JAX 68%
    - NYG (-82 Elo): Jaxson Dart (QB, inicia J.J. McCarthy)
- MIA @ BUF: Modelo 1 BUF 88% · Modelo 2 BUF 92%
    - MIA (-92 Elo): Malik Willis (QB, inicia Kyle McCord); Chris Bell (WRTE, Doubtful sem. 10)
- IND @ HOU: Modelo 1 HOU 67% · Modelo 2 HOU 71%
    - IND (-30 Elo): Josh Downs (WRTE, otro equipo)

_Ojo: el reporte de lesiones de esta semana aún no sale. Vuelve a correr `python3 src/weekly_content.py --solo-pronosticos` el viernes/sábado para congelar la versión final del Modelo 2._

## Picks de la semana 11 (hora CDMX)

- jue 19 nov 19:15 · IND @ HOU → **HOU 67%**
- dom 22 nov 12:00 · ARI @ KC → **KC 76%**
- dom 22 nov 12:00 · BAL @ CAR → **CAR 57%**
- dom 22 nov 12:00 · JAX @ NYG → **JAX 62%**
- dom 22 nov 12:00 · MIA @ BUF → **BUF 88%**
- dom 22 nov 12:00 · NO @ CHI → **CHI 80%**
- dom 22 nov 12:00 · TB @ DET → **DET 76%**
- dom 22 nov 12:00 · TEN @ DAL → **DAL 72%**
- dom 22 nov 15:05 · NYJ @ LAC → **LAC 68%**
- dom 22 nov 15:25 · LV @ DEN → **DEN 84%**
- dom 22 nov 15:25 · PIT @ PHI → **PIT 63%**
- dom 22 nov 19:20 · MIN @ SF → **SF 61%**
- lun 23 nov 19:15 · CIN @ WAS → **CIN 61%**

## Destacados

- **El más claro** · MIA @ BUF: El favorito más claro de la semana: BUF gana con 87.6%
- **La moneda al aire** · BAL @ CAR: Solo 14.4 puntos de diferencia: el partido más parejo de la semana
- **La sorpresa** · IND @ HOU: Las Vegas ve muy favorito a HOU (84.0%). El modelo lo ve más parejo: 67.1%.
