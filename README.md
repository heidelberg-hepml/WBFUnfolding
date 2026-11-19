<h1 align="center">WBFUnfolding</h1>

<p align="center">
<a href="https://pytorch.org"><img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.2+-DC583A.svg?style=flat&logo=pytorch"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Hydra-1.3-78a9c2"></a>
<a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</p>

Code for generative unfolding of Higgs production in weak boson fusion, with the decay
$H\to WW^*\to e\nu\mu\nu$, to parton level. It contains

- a transformer conditional flow matching (TraCFM) unfolding network with the transverse-mass
  phase-space parameterization, in which the parton-level cuts hold by construction;
- conditional training on several SM and EFT hypotheses, used to estimate the simulation dependence;
- classifier-based background subtraction ($t\bar t$ and $WWjj$) by reweighting the data before unfolding.

## Installation

```sh
git clone https://github.com/heidelberg-hepml/WBFUnfolding.git
cd WBFUnfolding
python -m venv venv
source venv/bin/activate
pip install --editable .
```

Tested with Python 3.11, PyTorch 2.12, Hydra 1.3 and OmegaConf 2.3. Training runs on a GPU.

## Data

The code reads `.npy` event files with reco level and truth level (parton: `hard`, particle: `gen`)
events, row-aligned:

| Process | Files |
|---|---|
| SM signal | `reco-hard_hww_{hard,rec}.npy`, `reco-gen_hww_{gen,rec}.npy` |
| CP-even, $c_{HW}=\pm0.5$ | `reco-hard_hww_chw0p5_*`, `reco-hard_hww_chwm0p5_*` |
| CP-odd, $c_{H\tilde W}=5,-2$ | `reco-hard_hww_chwtil5p0_*`, `reco-hard_hww_chwtilm2p0_*` |
| $t\bar t$ | `reco-{hard,gen}_ttbar_nobjet_nocjv_noolv_*` |
| $WWjj$ | `reco-{hard,gen}_wwjj_*` |

Set the sample directory once, as an absolute path:

```sh
export WBF_DATA_DIR=/abs/path/to/samples
export WBF_TRIMMED_DIR=$WBF_DATA_DIR/trimmed   # default; output of trim_background_files.py
```

or pass `data_dir=...` / `trimmed_dir=...` on the command line.

## Usage

All runs go through `run.py` with a [Hydra](https://hydra.cc) config from `config/`. A run writes its
config, logs, model and plots to `results/<model>/<MMDD_HHMMSS>-<run.name>/`.

```sh
python run.py -cn unfold                 # train an unfolder (config/unfold.yaml)
python run.py -cn classify               # train a classifier (config/classify.yaml)
python run.py -cn config -cp <run_dir> run.type=plot   # re-evaluate and re-plot a run
```

Any config entry can be overridden on the command line, e.g. `train.nits=500_000`.

## Reproducing the paper

The configs in `config/paper/` reproduce the trained networks of the paper.

**Unfolding the signal.** The parton-level unfolder with the naive parameterization, and with the
transverse-mass parameterization:

```sh
python run.py -cn paper/unfold_naive                                # naive, trained on the SM
python run.py -cn paper/unfold_sm                                  # trained on the SM
python run.py -cn paper/unfold_sm dataset.process=EFT_chw0p5        # trained on CP-even
python run.py -cn paper/unfold_sm dataset.process=EFT_chwtil5       # trained on CP-odd
python run.py -cn paper/unfold_conditioned                          # all five hypotheses, conditioned
```

**Simulation dependence.** Evaluate a trained unfolder on several samples and, for the conditioned
network, with all labels, then plot:

```sh
RUN=results/TraCFM/<run>
python run.py -cn config -cp $RUN run.path=$RUN \
    '+evaluate.sim_dep=[{name:eval_sm,eval_process:signal},{name:eval_cpeven_pos,eval_process:EFT_chw0p5},{name:eval_cpodd_pos,eval_process:EFT_chwtil5}]'
python experiments/plot_simulation_dependence.py $RUN --trained-on SM

# conditioned network: label ids 0..4 = SM, CP-even(+), CP-even(-), CP-odd(+), CP-odd(-)
python run.py -cn config -cp $RUN run.path=$RUN \
    '+evaluate.sim_dep=[{name:eval_cpodd_pos,subset:3},{name:eval_cpodd_pos_wronglabels,subset:3,labels:[0,1,2,4]}]'
python experiments/plot_simulation_dependence.py $RUN --wrong-labels eval_cpodd_pos_wronglabels --correct eval_cpodd_pos
```

**Background subtraction.**

```sh
# 1. keep only the uniform bulk of the background files
python experiments/trim_background_files.py

# 2. reweight the particle-cut ttbar sample to the parton-level cuts
python experiments/ttbar_chain_reweighting.py dataset.use_njets=true \
    dataset.features.dphi_jj=false dataset.features.dphi_jj_signed_eta=true
TTW=results/TransformerClassifier/<ttbar_run>/weights_particle_to_parton.npy

# 3. signal/background classifier, 3-fold cross-fitting, five fold assignments
for s in 0 1 2 3 4; do
    python run.py -cn paper/bkg_classifier train.kfold_seed=$s dataset.ttbar_event_weights=$TTW
done

# 4. merge them, with each member normalized to the background yield
python experiments/merge_kfold_dumps.py --bkg-yield-norm --out results/TransformerClassifier/bkg-clf-merged \
    results/TransformerClassifier/*-bkg-clf-kseed{0,1,2,3,4}
python experiments/replot_classifier_observables.py results/TransformerClassifier/bkg-clf-merged --bkg-yield-norm

# 5. unfold the background-subtracted data with the SM unfolder, in a sub-directory of its run
CLF=$(readlink -f results/TransformerClassifier/bkg-clf-merged)
UNF=$(readlink -f results/TraCFM/<sm_unfolder>)
python experiments/make_eval_block.py --classifier-dir $CLF --unfolder-dir $UNF
OUT=$UNF/reweight_bkgsub && mkdir -p $OUT
cp $UNF/config.yaml $OUT/ && ln -sfn $UNF/model $OUT/model
python run.py -cn config -cp $OUT run.path=$OUT run.type=plot \
    evaluate.unfold_data_weights=$CLF/eval_block/unfolding_weights_trn.pt \
    dataset.p_data.data_path=$CLF/eval_block
python experiments/replot_unfolded_data.py $OUT --classifier-dir $CLF   # with bootstrap + ensemble band
```

## Code structure

```
run.py                      entry point (task: unfolding | classification)
config/                     Hydra configs; config/paper/ for the paper runs
src/unfolding/              CFM / TraCFM models, training, evaluation, plots
src/classification/         classifier models, training, background-subtraction plots
src/datasets/               data loading, phase-space parameterizations, preprocessing
src/utils/physics.py        kinematics, transverse-mass parameterization and its inverse
experiments/                background-file trimming, ttbar reweighting, k-fold merging, paper plots
```

## Citation

If you use this code, please cite the paper.
