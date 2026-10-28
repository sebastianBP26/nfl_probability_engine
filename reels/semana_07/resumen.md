# Semana 7 · Pronosticando la NFL con un modelo de ELO

## ¿Cómo va el modelo? (semanas 1–6 · pick si prob. > 50%)

- Récord del modelo en la temporada (aciertos-fallos-empates): **53-39-1 (57.5%)**
- Semana 6: 8-6 (57.1%)
- Aciertos que el propio modelo esperaba: 58.2 → reales: 53.5 (empate = ½)
- Brier: 0.232 (moneda al aire = 0.250) · Log loss: 0.660 (moneda = 0.693)
- Cuando eligió al local: 35-27-1 · al visitante: 18-12
- En esos mismos partidos, siempre el local: 47-45-1
- Favorito de Las Vegas: 57-35-1 (61.8%) · el modelo coincide con Las Vegas en 82.8% de los picks · Brier Las Vegas: 0.231
- El batacazo de la semana 6: **ARI** ganó con solo 22% de probabilidad (ARI 18–16 LA)

### Calibración (temporada)

| Confianza | Partidos | Récord | Prob. media | % real |
|---|---|---|---|---|
| 50–60% | 43 | 20-22-1 | 55% | 48% |
| 60–70% | 30 | 17-13 | 65% | 57% |
| 70–80% | 18 | 14-4 | 75% | 78% |
| 80%+ | 2 | 2-0 | 81% | 100% |

## Picks de la semana 7 (hora CDMX)

- Jueves 22 oct 18:15 · NE @ CHI → **NE 52%**
- Domingo 25 oct 07:30 · PIT @ NO → **PIT 60%**
- Domingo 25 oct 11:00 · CIN @ BAL → **BAL 62%**
- Domingo 25 oct 11:00 · CLE @ TEN → **CLE 63%**
- Domingo 25 oct 11:00 · IND @ MIN → **MIN 78%**
- Domingo 25 oct 11:00 · MIA @ NYJ → **NYJ 57%**
- Domingo 25 oct 11:00 · NYG @ HOU → **HOU 66%**
- Domingo 25 oct 11:00 · SF @ ATL → **SF 70%**
- Domingo 25 oct 11:00 · TB @ CAR → **CAR 67%**
- Domingo 25 oct 14:05 · DEN @ ARI → **DEN 72%**
- Domingo 25 oct 14:25 · GB @ DET → **DET 56%**
- Domingo 25 oct 14:25 · LA @ LV → **LA 58%**
- Domingo 25 oct 18:20 · KC @ SEA → **SEA 61%**
- Lunes 26 oct 18:15 · DAL @ PHI → **PHI 64%**

## Destacados

- **El más seguro** · IND @ MIN: El pick con más confianza de la semana: MIN gana con 78.2%
- **La moneda al aire** · NE @ CHI: Solo 4.8 puntos de diferencia: el partido más parejo de la semana
- **La sorpresa** · DAL @ PHI: Las Vegas va con DAL (66.4%). El modelo va con PHI (64.2%).
