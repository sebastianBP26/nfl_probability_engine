# Semana 9 · Pronosticando la NFL con un modelo de ELO

## ¿Cómo va el modelo? (semanas 1–8 · pick si prob. > 50%)

- Récord del modelo en la temporada (aciertos-fallos-empates): **68-52-1 (56.6%)**
- Semana 8: 8-6 (57.1%)
- Aciertos que el propio modelo esperaba: 76.0 → reales: 68.5 (empate = ½)
- Brier: 0.231 (moneda al aire = 0.250) · Log loss: 0.656 (moneda = 0.693)
- Cuando eligió al local: 45-35-1 · al visitante: 23-17
- En esos mismos partidos, siempre el local: 62-58-1
- Favorito de Las Vegas: 78-42-1 (64.9%) · el modelo coincide con Las Vegas en 81.8% de los picks · Brier Las Vegas: 0.225
- El batacazo de la semana 8: **ARI** ganó con solo 35% de probabilidad (ARI 33–30 DAL)

### Calibración (temporada)

| Confianza | Partidos | Récord | Prob. media | % real |
|---|---|---|---|---|
| 50–60% | 51 | 22-28-1 | 55% | 44% |
| 60–70% | 43 | 24-19 | 65% | 56% |
| 70–80% | 25 | 20-5 | 74% | 80% |
| 80%+ | 2 | 2-0 | 81% | 100% |

## Picks de la semana 9 (hora CDMX)

- jue 5 nov 19:15 · JAX @ BAL → **JAX 55%**
- dom 8 nov 08:30 · CIN @ ATL → **CIN 53%**
- dom 8 nov 12:00 · CLE @ NO → **CLE 54%**
- dom 8 nov 12:00 · DAL @ IND → **IND 53%**
- dom 8 nov 12:00 · DEN @ CAR → **DEN 70%**
- dom 8 nov 12:00 · DET @ MIA → **DET 68%**
- dom 8 nov 12:00 · LA @ WAS → **LA 60%**
- dom 8 nov 12:00 · NYG @ PHI → **PHI 52%**
- dom 8 nov 12:00 · NYJ @ KC → **KC 78%**
- dom 8 nov 15:05 · HOU @ LAC → **HOU 55%**
- dom 8 nov 15:05 · LV @ SF → **SF 80%**
- dom 8 nov 15:25 · ARI @ SEA → **SEA 77%**
- dom 8 nov 15:25 · GB @ NE → **NE 65%**
- dom 8 nov 19:20 · TB @ CHI → **CHI 74%**
- lun 9 nov 19:15 · BUF @ MIN → **MIN 50%**

## Destacados

- **El más claro** · LV @ SF: El favorito más claro de la semana: SF gana con 80%
- **La moneda al aire** · BUF @ MIN: Prácticamente 50-50: el partido más parejo de la semana
- **La sorpresa** · CLE @ NO: Las Vegas va con NO (66%). El modelo va con CLE (54%).
