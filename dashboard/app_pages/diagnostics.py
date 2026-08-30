"""Diagnostics — technical detail for users who want it, kept out of the main workflow. This is
where full scientific disclaimers, model internals, and known limitations live in the app; deeper
development history remains in the repository's documentation files (README, DEVELOPMENT_LOG,
docs/), not surfaced here.
"""
import streamlit as st

from data import PROJECT_ROOT, load_test_metrics
from theme import ICONS, section_header

section_header("Diagnostics", "Technical detail behind every result in this application.", ICONS["diagnostics"])

sel = st.session_state.get("model_selection", {})

with st.expander("Model information", icon=":material/model_training:", expanded=True):
    if sel:
        metrics = load_test_metrics(sel["experiment_name"])
        st.write(f"**Selected model:** {sel['display_name']}")
        st.write(f"**Configuration:** `{sel['config_path']}`")
        st.write(f"**Checkpoint:** `{sel['checkpoint_path']}`")
        if metrics:
            tm = metrics["test_metrics"]
            st.write(
                f"**Benchmark metrics** (held-out test set, {metrics.get('checkpoint_epoch', 'N/A')} "
                f"training epochs): IoU={tm['iou']:.4f}, Dice={tm['dice']:.4f}, "
                f"Precision={tm['precision']:.4f}, Recall={tm['recall']:.4f}, "
                f"Accuracy={tm['accuracy']:.4f}"
            )
    else:
        st.caption("No model currently selected.")

with st.expander("Input validation", icon=":material/fact_check:"):
    st.markdown(
        """
Every uploaded image pair is checked automatically before inference:

- **Dimension match** — the before and after images must have identical dimensions.
- **Registration offset estimate** — a phase-correlation estimate of pixel shift between the two
  images. This is a diagnostic only; it does not correct or align the images, and a large estimate
  does not by itself prove misalignment (real large-scale change can also produce one).
- **Resolution plausibility** — flags imagery whose pixel size is much coarser than the model's
  training resolution, when a real pixel size is known.
- **Cloud / overexposure heuristic** — flags images with an unusually large share of bright,
  low-saturation pixels. This is a simple heuristic, not a validated cloud detector — it can miss
  real cloud cover and can flag bright surfaces (rooftops, sand, snow) that are not clouds.

None of these checks confirm prediction accuracy. They describe the input only.
        """
    )

with st.expander("Prediction probability", icon=":material/percent:"):
    st.markdown(
        """
The probability map shown for each detection is the model's raw sigmoid output per pixel — the
network's own estimate that a pixel changed. **It is not a calibrated confidence score**: a pixel
at 0.8 is not independently verified to be correct roughly 80% of the time. No calibration study
has been run on this model.

The decision threshold (set in the sidebar) converts this probability map into a binary change
mask. The default value was selected via a sweep over the validation set only — it was never
tuned against the test set used for reported accuracy.
        """
    )

with st.expander("Failure cases", icon=":material/report:"):
    st.markdown(
        """
The model's real, measured test predictions show generally strong agreement with ground truth,
but not perfect agreement. One documented case: a scene with **no real change** produced a small
cluster of false-positive predictions with no counterpart in the ground truth — a concrete example
of the model occasionally reacting to lighting, seasonal, or registration differences rather than
genuine geographic change. This is disclosed rather than hidden, even though it is an imperfection
in the best-performing model.
        """
    )

with st.expander("Data compatibility", icon=":material/compare_arrows:"):
    st.markdown(
        """
This model was trained exclusively on the LEVIR-CD benchmark: 0.5 m/pixel satellite imagery of
building change in Texas suburbs, pre-registered before/after pairs. It predicts a **binary
building-change mask only** — it does not classify the type of change (road, vegetation, water,
etc.), since the training data contains no such labels.

Uploaded imagery that differs substantially from this — much coarser resolution, a different
sensor, unregistered images, or a different geographic/urban context — has not been independently
validated. Area statistics assume the LEVIR-CD effective resolution unless a real pixel size is
otherwise known (as it is for the Geospatial Intelligence page, which uses real georeferenced
imagery).
        """
    )

with st.expander("Known limitations", icon=":material/rule:"):
    st.markdown(
        """
- Benchmark accuracy figures are measured on the LEVIR-CD test set only and do not transfer
  directly to arbitrary uploaded imagery.
- Severity scores are an analytical ranking derived from model outputs — not ground truth or a
  validated physical damage assessment.
- Multi-temporal intervals are independent detections; no causal or object-tracking claim is made.
- Multi-class / damage-type classification is not implemented — no suitable labeled dataset for
  this task could be reliably obtained.
- A research-only Transformer-based architecture is available for comparison; it underperforms the
  recommended model and is not intended for production use.
- Training used a single run/seed per configuration — no variance estimate exists across repeated
  runs.

Full technical documentation — dataset methodology, training configuration, evaluation
methodology, and the complete limitations record — is maintained in the project repository
(`README.md`, `docs/`).
        """
    )
