# Semana 12 · Modelo 2: Elo + ausencias

## ¿Cómo va el modelo? (semanas 1–11 · pick si prob. > 50%)

_Historial publicado · Semanas 1–9: modelo Elo · semanas 10–11: Elo + ausencias (pronósticos congelados antes de cada partido)._

- Récord del modelo en la temporada (aciertos-fallos-empates): **96-66-1 (59.2%)**
- Semana 11: 9-4 (69.2%)
- Aciertos que el propio modelo esperaba: 103.7 → reales: 96.5 (empate = ½)
- Brier: 0.220 (moneda al aire = 0.250) · Log loss: 0.630 (moneda = 0.693)
- Cuando eligió al local: 61-44-1 · al visitante: 35-22
- En esos mismos partidos, siempre el local: 83-79-1
- Favorito de Las Vegas: 110-51-1 (68.2%) · el modelo coincide con Las Vegas en 84.0% de los picks · Brier Las Vegas: 0.209
- El batacazo de la semana 11: **PHI** ganó con solo 26% de probabilidad (PIT 16–19 PHI)

### Calibración (temporada)

_Historial publicado · Semanas 1–9: modelo Elo · semanas 10–11: Elo + ausencias (gráfica 11_calibracion)._

| Confianza | Partidos | Récord | Prob. media | % real |
|---|---|---|---|---|
| 50–60% | 66 | 29-36-1 | 55% | 45% |
| 60–70% | 55 | 31-24 | 65% | 56% |
| 70–80% | 36 | 30-6 | 75% | 83% |
| 80%+ | 6 | 6-0 | 84% | 100% |

## Modelo 1 (Elo) vs Modelo 2 (Elo + ausencias)

Pronósticos congelados antes de cada kickoff · semanas 1–11 · 163 partidos · pick si prob. > 50%

| Modelo | Récord | % aciertos | Brier | Log loss |
|---|---|---|---|---|
| Modelo 2 · Elo + ausencias | 106-56-1 | 65.3% | 0.214 | 0.619 |
| Modelo 1 · Elo | 96-66-1 | 59.2% | 0.220 | 0.631 |
| Las Vegas (referencia) | — | 68.2% | 0.209 | — |

- Mejor Brier hasta ahora: **Modelo 2 · Elo + ausencias** (más bajo = mejor).
- Eligieron distinto favorito en 12 partidos → acertó el Modelo 1 en 1 y el Modelo 2 en 11.
- Semanas 1–9: reconstruidas (backfill) con el reporte final de lesiones de cada semana.

### Semana 12: dónde difieren (ausencias según provisional: reporte sem. 11)

- WAS @ ARI: Modelo 1 WAS 55% · Modelo 2 WAS 67%
    - ARI (-110 Elo): Jacoby Brissett (QB, inicia Gardner Minshew); Michael Wilson (WRTE, IR)
- NYG @ IND: Modelo 1 NYG 58% · Modelo 2 NYG 67%
    - NYG (-84 Elo): Jaxson Dart (QB, inicia J.J. McCarthy)
    - IND (-110 Elo): Daniel Jones (QB, inicia Anthony Richardson); Josh Downs (WRTE, otro equipo)
- NYJ @ MIA: Modelo 1 MIA 56% · Modelo 2 NYJ 53% · **cambia el favorito**
    - NYJ (-23 Elo): Omar Cooper (WRTE, Out sem. 11)
    - MIA (-84 Elo): Malik Willis (QB, inicia Kyle McCord)
- PHI @ DAL: Modelo 1 DAL 56% · Modelo 2 DAL 64%
    - PHI (-80 Elo): Jalen Hurts (QB, inicia Joe Milton III)
- BAL @ HOU: Modelo 1 HOU 66% · Modelo 2 HOU 73%
    - BAL (-80 Elo): Lamar Jackson (QB, inicia Jacoby Brissett)
- SEA @ SF: Modelo 1 SF 62% · Modelo 2 SF 68%
    - SEA (-52 Elo): Sam Darnold (QB, Out sem. 10)
- GB @ LA: Modelo 1 LA 61% · Modelo 2 LA 56%
    - LA (-43 Elo): Puka Nacua (WRTE, IR)
- ATL @ MIN: Modelo 1 MIN 76% · Modelo 2 MIN 73%
    - MIN (-20 Elo): Jordan Addison (WRTE, Out sem. 11)

_Ojo: el reporte de lesiones de esta semana aún no sale. Vuelve a correr `python3 src/weekly_content.py --solo-pronosticos` el viernes/sábado para congelar la versión final del Modelo 2._

## Picks de la semana 12 (hora CDMX)

- mié 25 nov 19:00 · GB @ LA → **LA 56%**
- jue 26 nov 12:00 · CHI @ DET → **DET 56%**
- jue 26 nov 15:30 · PHI @ DAL → **DAL 64%**
- jue 26 nov 19:20 · KC @ BUF → **BUF 69%**
- vie 27 nov 14:00 · DEN @ PIT → **DEN 60%**
- dom 29 nov 12:00 · ATL @ MIN → **MIN 73%**
- dom 29 nov 12:00 · BAL @ HOU → **HOU 73%**
- dom 29 nov 12:00 · NO @ CIN → **CIN 72%**
- dom 29 nov 12:00 · NYG @ IND → **NYG 67%**
- dom 29 nov 12:00 · NYJ @ MIA → **NYJ 53%**
- dom 29 nov 12:00 · TEN @ JAX → **JAX 87%**
- dom 29 nov 15:05 · NE @ LAC → **NE 68%**
- dom 29 nov 15:25 · SEA @ SF → **SF 68%**
- dom 29 nov 15:25 · WAS @ ARI → **WAS 67%**
- dom 29 nov 19:20 · LV @ CLE → **CLE 70%**
- lun 30 nov 19:15 · CAR @ TB → **TB 51%**

## Destacados

- **El más claro** · TEN @ JAX: El favorito más claro de la semana: JAX gana con 87.5%
- **La moneda al aire** · CAR @ TB: Solo 2.8 puntos de diferencia: el partido más parejo de la semana
- **La sorpresa** · NE @ LAC: Las Vegas va con LAC (51.2%). El modelo va con NE (68.4%).
