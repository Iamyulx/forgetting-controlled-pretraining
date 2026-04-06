# Pretraining continuo con Forgetting Control

Este proyecto simula **catastrophic forgetting** durante **pretraining continuo** de un language model pequeno y compara cuatro estrategias:

- `sequential_baseline`: sigue pretraining solo con datos nuevos.
- `replay_only`: mezcla ejemplos viejos desde un `replay buffer`.
- `ewc_only`: aplica regularizacion tipo EWC.
- `replay_plus_ewc`: combina replay y EWC.

La idea central no es hacer fine-tuning de una tarea final, sino mantener el **mismo objetivo de language modeling** mientras cambia la distribucion del corpus. Eso lo vuelve un experimento de **continual pretraining**.

## Que incluye

- Generacion de dos corpora sinteticos con cambio de dominio:
  - dominio antiguo: cientifico/tecnico
  - dominio nuevo: financiero/regulatorio
- Modelo causal pequeno basado en Transformer
- `ReplayBuffer` con reservoir sampling
- Estimacion diagonal de Fisher para EWC
- Runner de experimentos con metricas por epoca
- Exportacion de:
  - `history.csv`
  - `summary.csv`
  - `dataset_preview.json`
  - graficos `.png`

## Metricas que reporta

- `old_val_loss` y `old_val_perplexity`
- `new_val_loss` y `new_val_perplexity`
- `old_val_accuracy` y `new_val_accuracy`
- `forgetting_score`: cuanto empeora el dominio viejo tras pretraining continuo
- `retention_ratio`: que fraccion de la performance vieja se mantiene
- `plasticity_gain`: cuanto mejora el dominio nuevo durante la etapa 2

## Estructura

```text
run_experiment.py
src/forgetting_control/data.py
src/forgetting_control/model.py
src/forgetting_control/strategies.py
src/forgetting_control/experiment.py
```

## Como correrlo

Modo rapido para validar pipeline:

```bash
python run_experiment.py --quick
```

Modo completo:

```bash
python run_experiment.py --output-dir outputs/full_run
```

## Salidas esperadas

Al terminar, en la carpeta de salida veras:

- `history.csv`: evolucion completa por experimento y epoca
- `summary.csv`: comparacion final de estrategias
- `validation_curves.png`: curvas de perplexity viejo/nuevo
- `strategy_comparison.png`: barras con forgetting, retencion y plasticidad

## Interpretacion

- Si `sequential_baseline` empeora mucho en `old_val_perplexity`, hay forgetting claro.
- Si `replay_only` baja ese dano, el buffer esta ayudando a retener conocimiento viejo.
- Si `ewc_only` tambien mejora retencion, EWC esta frenando cambios destructivos en pesos importantes.
- Si `replay_plus_ewc` logra buena retencion sin destruir la adaptacion al dominio nuevo, tienes la mejor evidencia del enfoque.
