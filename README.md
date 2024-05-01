# dark-photon-sensitivity
Making plots for projected sensitivity to dark photon models

# Setup

## Git clone this repo

Run
```
git clone git@github.com:kehangbai/dark-photon-sensitivity.git
```

## Setup darkcast

Git clone the `darkcast` [repo](https://gitlab.com/philten/darkcast) first to this directory
```
cd dark-photon-sensitivity
git clone git@gitlab.com:philten/darkcast.git
```

In `darkcast/models`, remove all other model besides `dark_photon.py`. 

This allows us to calculate the correct $Br(A' \to \mu\mu)$ and total width of the $A'$.

## Define model and plot

For example
```
Ap_model = DarkPhotonModel(BR_hApAp=1e-4, Lmin=0.001, Lmax=0.3)
Ap_model.plot_num_sig(title_list)
```

This model define the $Br(H \to A' A')$ to be $10^{-4}$, with an accepted decay distance of $1-300$ mm.

