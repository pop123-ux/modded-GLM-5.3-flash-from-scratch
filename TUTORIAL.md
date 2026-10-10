# A practical first experiment

1. Create the Python environment and run tests using README instructions.
2. Open `configs/user_repaired.json` in your IDE. Its values preserve your residual scaling/dropout choices while using a valid head dimension.
3. Run ten tiny steps with `configs/tiny.json`. Inspect `train.jsonl`, `training-receipt.json`, and a saved `config.json`.
4. Open `experiments/run_ablation_matrix.py`. Choose two arms, initially `vuk` and `hybrid`. Use the same data/model seeds, context, batch size and device.
5. Run enough steps to see a meaningful loss curve. Tiny smoke runs cannot select the best architecture.
6. Change one factor—such as `sparse_sdpa`—and compare token-weighted loss and training time. Then run multiple paired seeds.
7. Choose a quality threshold from the baseline; rerun matched conditions with `--target-loss`. Do not report only a faster step time if quality drops.
8. Evaluate the final selected checkpoint once on `confirm`, keeping both cross entropy and executable success.

The supplied notebooks are your educational companion. Former empty notebooks are valid JSON placeholders, explicitly marked unfinished. Original annotated Python sources are saved as text for side-by-side IDE review; active code lives in `glm53_flash/`.
