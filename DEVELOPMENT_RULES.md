# DEVELOPMENT_RULES.md

Binding rules for all work on this project. See `PROJECT_CONTEXT.md` for the objective and
architecture these rules protect.

## Rule 1 — Work phase-by-phase
Implement PHASE 0 through PHASE 12 sequentially (see `PROJECT_CONTEXT.md`). Never skip the
baseline U-Net. Never start advanced/Transformer architectures before the Siamese U-Net baseline
is trained and evaluated.

## Rule 2 — Verify before moving forward
At the end of every phase:
1. Run appropriate tests.
2. Run the application/code where applicable.
3. Verify generated files.
4. Check for errors.
5. Fix errors before continuing.
6. Record the result in `DEVELOPMENT_LOG.md`.
7. Summarize what was completed.
8. Clearly identify the next phase.

A phase is never declared complete if it has not been tested.

## Rule 3 — Never fabricate results
Never invent IoU, Dice, Precision, Recall, F1, Accuracy, training time, inference time, number of
detected changes/buildings, changed area, or any other numeric result. All results come from actual
execution. Anything not yet measured is labeled `NOT YET MEASURED` — not estimated, not implied.

## Rule 4 — Do not claim unsupported capabilities
If the training data only supports binary building-change masks, the system does not claim to
classify roads/vegetation/water automatically. Every capability is labeled one of:
`Implemented` / `Experimentally supported` / `Planned` / `Future scope`.

## Rule 5 — Prefer simple working implementations
Simple baseline -> working Siamese U-Net -> evaluation -> improvement. Advanced techniques are
introduced only when they provide measurable value, established by comparing real metrics.

## Rule 6 — Preserve working code
Before modifying existing code: inspect current implementation, understand dependencies, make the
smallest reasonable change, run tests. No unnecessary rewrites of working modules.

## Rule 7 — Reproducibility
Fixed random seeds where appropriate, YAML configuration files, saved checkpoints, saved metrics,
named experiments, documented hyperparameters. Every experiment must be reproducible from its config.

## Error handling protocol
1. Read the complete traceback.
2. Identify the root cause.
3. Inspect the relevant files.
4. Fix the root cause (not a workaround).
5. Re-run the failed command.
6. Run related tests.
7. Document the fix if significant.
Never randomly modify unrelated files to make an error go away.

## Stop conditions
Stop and ask the user when:
1. A required dataset cannot be reliably obtained.
2. A download requires manual authentication or a license agreement.
3. Hardware limitations make the intended experiment impractical.
4. A major architectural decision cannot be made safely.
5. A dependency conflict cannot be resolved without changing the intended stack.
6. The dataset lacks labels required for a proposed feature.
7. Continuing would require inventing results.
8. A destructive change could damage working code.

## Git
Conventional commits per phase (`phase-0: ...`, `phase-1: ...`, etc.). Never commit virtual
environments, large raw datasets, secrets, `.env` files, temporary files, or unnecessary model
artifacts.

## Security
Never hardcode API keys, tokens, passwords, or credentials. Use `.env` (never committed) with
`.env.example` documenting required variables.

## Documentation honesty
Documentation describes the actual implemented system. Future functionality is never documented
as if it already exists.
