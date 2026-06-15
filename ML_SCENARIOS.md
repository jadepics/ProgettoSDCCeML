# Scenari sperimentali per accuratezza classificazione

Questi scenari sono gestiti dal master in `DataPreparationService` tramite il campo `dataset_scenario` della `SubmitTrainingRequest`.

## Ordine consigliato

1. `baseline_no_leakage`
   - Target: `diagnosed_diabetes`
   - Drop: `diabetes_stage`, `diabetes_risk_score`
   - Serve come baseline corretta senza leakage evidente.

2. `diagnostic_noise_10pct`, `diagnostic_noise_25pct`, `diagnostic_noise_50pct`
   - Target: `diagnosed_diabetes`
   - Drop: `diabetes_stage`, `diabetes_risk_score`
   - Aggiunge rumore gaussiano deterministico ai marker diagnostici:
     `hba1c`, `glucose_fasting`, `glucose_postprandial`, `insulin_level`.
   - Serve a misurare quanto il modello dipende da misure cliniche precise.

3. `imbalance_positive_80`, `imbalance_positive_90`, `imbalance_negative_80`
   - Target: `diagnosed_diabetes`
   - Drop: `diabetes_stage`, `diabetes_risk_score`
   - Cambia il rapporto tra classi via downsampling deterministico.
   - Serve a evidenziare perché accuracy da sola può essere fuorviante.

4. `stage_multiclass_no_leakage`
   - Target: `diabetes_stage`
   - Drop: `diagnosed_diabetes`, `diabetes_risk_score`
   - Esperimento multi-classe: `No Diabetes`, `Pre-Diabetes`, `Type 1`, `Type 2`, `Gestational`.
   - Da leggere soprattutto con `balanced_accuracy` e `macro_f1`, perché alcune classi sono molto rare.

## Metriche aggiunte

Per ogni classificazione vengono ora salvate anche:

- `balanced_accuracy`
- `macro_f1`
- `weighted_f1`

Queste metriche vengono salvate in `validation_metrics` e `test_metrics`, insieme ad accuracy, classification report e confusion matrix.

## Selezione modello

`ModelSelector` supporta ora anche:

- `balanced_accuracy`
- `macro_f1`
- `weighted_f1`

Di default rimane `auto`, cioè accuracy per classification e R2 per regression. Per esperimenti sbilanciati puoi avviare il master con:

```bash
export MODEL_SELECTION_METRIC=macro_f1
```

oppure:

```bash
export MODEL_SELECTION_METRIC=balanced_accuracy
```
