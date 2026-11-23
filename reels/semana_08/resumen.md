# Semana 8 · Pronosticando la NFL con un modelo de ELO

## ¿Cómo va el modelo? (semanas 1–7 · pick si prob. > 50%)

- Récord del modelo en la temporada (aciertos-fallos-empates): **60-46-1 (56.5%)**
- Semana 7: 7-7 (50.0%)
- Aciertos que el propio modelo esperaba: 67.0 → reales: 60.5 (empate = ½)
- Brier: 0.236 (moneda al aire = 0.250) · Log loss: 0.668 (moneda = 0.693)
- Cuando eligió al local: 38-32-1 · al visitante: 22-14
- En esos mismos partidos, siempre el local: 52-54-1
- Favorito de Las Vegas: 67-39-1 (63.1%) · el modelo coincide con Las Vegas en 82.2% de los picks · Brier Las Vegas: 0.231
- El batacazo de la semana 7: **ATL** ganó con solo 30% de probabilidad (SF 20–29 ATL)

### Calibración (temporada)

| Confianza | Partidos | Récord | Prob. media | % real |
|---|---|---|---|---|
| 50–60% | 47 | 22-24-1 | 55% | 48% |
| 60–70% | 37 | 20-17 | 64% | 54% |
| 70–80% | 21 | 16-5 | 74% | 76% |
| 80%+ | 2 | 2-0 | 81% | 100% |

## Picks de la semana 8 (hora CDMX)

- jue 29 oct 18:15 · CAR @ GB → **GB 64%**
- dom 1 nov 12:00 · ARI @ DAL → **DAL 65%**
- dom 1 nov 12:00 · ATL @ TB → **ATL 51%**
- dom 1 nov 12:00 · BAL @ BUF → **BUF 72%**
- dom 1 nov 12:00 · CLE @ PIT → **PIT 65%**
- dom 1 nov 12:00 · IND @ JAX → **JAX 74%**
- dom 1 nov 12:00 · LV @ NYJ → **NYJ 52%**
- dom 1 nov 12:00 · MIN @ DET → **MIN 53%**
- dom 1 nov 12:00 · TEN @ CIN → **CIN 78%**
- dom 1 nov 15:05 · LAC @ LA → **LA 70%**
- dom 1 nov 15:25 · KC @ DEN → **DEN 67%**
- dom 1 nov 15:25 · NE @ MIA → **NE 68%**
- dom 1 nov 19:20 · PHI @ WAS → **PHI 52%**
- lun 2 nov 19:15 · CHI @ SEA → **SEA 62%**

## Destacados

- **El más claro** · TEN @ CIN: El favorito más claro de la semana: CIN gana con 78%
- **La moneda al aire** · ATL @ TB: Solo 2 puntos de diferencia: el partido más parejo de la semana
- **La sorpresa** · KC @ DEN: Las Vegas va con KC (53%). El modelo va con DEN (67%).
