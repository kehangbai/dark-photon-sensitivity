"""Generate production cross-section tables on a fine dark-photon mass grid.

Compute Cross Sections times BRs, scaning over m_Ap in
10 MeV steps and writes one results file per mass.

It only relied on the darkcast package: https://gitlab.com/philten/darkcast
which should be installed first.

Usage:
    python make_xsec_tables.py
    python make_xsec_tables.py --mmin 0.2 --mmax 20.0 --step 0.01 --br 0.01
"""
import argparse
import numpy as np
import darkcast

# ggF production cross section for mH = 125 GeV (fb)
SIGMA_GGF = {
    "Run2": 48580,    # 13 TeV
    "Run3": 52230,    # 13.6 TeV
}

CHANNEL = "mu_mu"


def get_model():
    # Load dark photon model from darkcast/models (and DARKCAST_MODEL_PATH).
    return darkcast.Models()["dark_photon"]


def xsec_4mu(model, m_Ap, sigma_ggF, br_hApAp):
    # Both A' -> mu mu
    br = model.bfrac(CHANNEL, m_Ap)
    return sigma_ggF * br_hApAp * br * br


def xsec_2mu(model, m_Ap, sigma_ggF, br_hApAp):
    # Exactly one A' -> mu mu
    br = model.bfrac(CHANNEL, m_Ap)
    return sigma_ggF * br_hApAp * 2 * br * (1 - br)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mmin", type=float, default=0.2, help="min m_Ap (GeV)")
    p.add_argument("--mmax", type=float, default=20.0, help="max m_Ap (GeV)")
    p.add_argument("--step", type=float, default=0.01, help="mass step (GeV); 0.01 = 10 MeV")
    p.add_argument("--br", type=float, default=0.01, help="Br(h -> A'A')")
    args = p.parse_args()

    model = get_model()

    # Build the mass grid in integer multiples of `step` to avoid float drift.
    n = int(round((args.mmax - args.mmin) / args.step)) + 1
    masses = args.mmin + args.step * np.arange(n)

    for run, sigma_ggF in SIGMA_GGF.items():
        fname = f"results_{run}_10MeV.txt"
        with open(fname, "w") as f:
            f.write("m_Ap\txsec_2mu\txsec_4mu\n")
            for m in masses:
                xs2 = xsec_2mu(model, m, sigma_ggF, args.br)
                xs4 = xsec_4mu(model, m, sigma_ggF, args.br)
                # mass written in MeV, matching results_Run*.txt convention
                f.write(f"{m * 1000:.0f}\t{xs2:.6f}\t{xs4:.6f}\n")
        print(f"Wrote {fname} ({len(masses)} mass points, {args.step*1000:.0f} MeV steps)")


if __name__ == "__main__":
    main()
