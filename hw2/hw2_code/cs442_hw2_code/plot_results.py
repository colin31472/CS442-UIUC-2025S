import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(preds, truth, attrs):
    """
    Returns a triple: (overall_error, stat_parity_gap, eq_odds_gap)
    """
    assert len(preds) == len(truth) == len(attrs)

    # Overall classification error
    overall_err = np.mean(preds != truth)

    # Statistical parity gap
    idx0 = (attrs == 0)
    idx1 = (attrs == 1)
    p0 = np.mean(preds[idx0])
    p1 = np.mean(preds[idx1])
    sp_gap = abs(p0 - p1)

    # Equalized odds gap
    # We break it into Y=0 and Y=1 subsets, measure predictions among A=0 vs A=1
    idx_y0 = (truth == 0)
    idx_y1 = (truth == 1)
    # false-positive rate difference
    fpr_a0 = np.mean(preds[idx0 & idx_y0])
    fpr_a1 = np.mean(preds[idx1 & idx_y0])
    # true-positive rate difference
    tpr_a0 = np.mean(preds[idx0 & idx_y1])
    tpr_a1 = np.mean(preds[idx1 & idx_y1])
    # eq_odds_gap = 0.5(|fpr_a0 - fpr_a1| + |tpr_a0 - tpr_a1|)
    eo_gap = 0.5 * (abs(fpr_a0 - fpr_a1) + abs(tpr_a0 - tpr_a1))

    return overall_err, sp_gap, eo_gap

def load_experiment(dataset, model, mus):
    """
    Loads the npz files and returns lists of (error, spgap, eogap) for each mu.
    """
    errors, spgaps, eogaps = [], [], []
    for mu in mus:
        fname = f"{dataset}_{model}_{mu}.npz"
        data = np.load(fname)
        preds = data["prediction"]
        truth = data["truth"]
        attrs = data["attribute"]

        err, sp, eo = compute_metrics(preds, truth, attrs)
        errors.append(err)
        spgaps.append(sp)
        eogaps.append(eo)

    return errors, spgaps, eogaps


if __name__ == "__main__":
    mus = [0.1, 1.0, 10.0]
    # We'll do it for both 'adult' and 'compas'
    # and for both 'fair' and 'cfair-eo'
    # Then we can produce 3 plots per dataset,
    # each plot containing two lines: one for 'fair' and one for 'cfair-eo'.

    for dataset in ["adult", "compas"]:
        # Gather metrics
        fair_err, fair_sp, fair_eo = load_experiment(dataset, "fair", mus)
        cfair_err, cfair_sp, cfair_eo = load_experiment(dataset, "cfair-eo", mus)

        # --- Overall Error ---
        plt.figure()
        plt.plot(mus, fair_err, marker='o', label="FairNet")
        plt.plot(mus, cfair_err, marker='o', label="CFairNet (EO)")
        plt.xlabel("Adversarial weight (mu)")
        plt.ylabel("Overall Error")
        plt.xscale("log")      # so 0.1,1,10 are spaced nicely
        plt.title(f"{dataset.capitalize()} - Overall Error")
        plt.legend()
        plt.show()

        # --- Statistical Parity Gap ---
        plt.figure()
        plt.plot(mus, fair_sp, marker='o', label="FairNet")
        plt.plot(mus, cfair_sp, marker='o', label="CFairNet (EO)")
        plt.xlabel("Adversarial weight (mu)")
        plt.ylabel("SP Gap")
        plt.xscale("log")
        plt.title(f"{dataset.capitalize()} - Statistical Parity Gap")
        plt.legend()
        plt.show()

        # --- Equalized Odds Gap ---
        plt.figure()
        plt.plot(mus, fair_eo, marker='o', label="FairNet")
        plt.plot(mus, cfair_eo, marker='o', label="CFairNet (EO)")
        plt.xlabel("Adversarial weight (mu)")
        plt.ylabel("Equalized Odds Gap")
        plt.xscale("log")
        plt.title(f"{dataset.capitalize()} - Equalized Odds Gap")
        plt.legend()
        plt.show()
