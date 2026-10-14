# U-Nets on turbulent channel-flow fields: super-resolution and time stepping

These four notebooks put one convolutional network, a four-level U-Net with 1.94 million parameters, to two different jobs on the same data: 5,777 planes of the streamwise velocity u from a turbulent channel-flow simulation. In Stage 1 the U-Net restores fine detail in space: each plane is coarsened by a factor of 4, interpolated back, and the network learns to recover the original. Stage 2 turns the planes into data for the second job, pairs of consecutive snapshots written as sharded TFRecord files, and Stage 3 streams those files to train the same U-Net to advance the field in time, u(t) → u(t + 1). Stage 4 asks whether either result beats what a careful engineer would try first: nearest-neighbour and bicubic upscaling for the first job; persistence (the next field equals the current one) and the mean velocity profile for the second, at one step and over rollouts as long as the data allow.

Portfolio page: https://vamsigudipati.github.io/unet-turbulent-fields.html

## Notebooks

| Notebook | Stage | Results | Models | Animations |
| --- | --- | --- | --- | --- |
| `notebooks/01_unet_turbulent_superresolution.ipynb` | Stage 1 — U-Net super-resolution of turbulent channel-flow fields | `results/01_unet_turbulent_superresolution/` | `models/01_unet_turbulent_superresolution/` | `animations/01_*` |
| `notebooks/02_prepare_temporal_pairs_tfrecords.ipynb` | Stage 2 — Temporal pairs as TFRecords | `results/02_prepare_temporal_pairs_tfrecords/` | — | — |
| `notebooks/03_unet_temporal_evolution.ipynb` | Stage 3 — U-Net temporal evolution from TFRecords | `results/03_unet_temporal_evolution/` | `models/03_unet_temporal_evolution/` | `animations/03_*` |
| `notebooks/04_baselines_and_rollout.ipynb` | Stage 4 — Baselines and rollout | `results/04_baselines_and_rollout/` | — | — |

Each notebook is also exported as HTML next to it, with all outputs. In `results/`, the `exercises/` folder of Stages 1 and 3 holds the tables (CSV) and figures (PNG) of the exercises and of the three-seed repeats, and `figures/` the architecture drawings; Stage 4 writes its tables and figures directly into its results folder. `models/` holds the trained networks as Keras files (the per-epoch checkpoints stay out of git); `animations/` holds the training videos and their final frames.

## Data

The planes are y–x slices (201 × 384 points, u only, one HDF5 file each, gzip-compressed) of a direct numerical simulation of turbulent channel flow, 60 spanwise positions with up to 107 snapshots each, named `P125_83pi.<snapshot>.<plane>.h5`. They are too large for git and come with the repository's first release:

| Release file | Size | Goes to | Read by |
| --- | --- | --- | --- |
| `u_yx_planes_h5.tar` | 1.60 GB | extract into `release_assets/`, giving `release_assets/u_yx_planes_h5/` (5,777 files) | Stages 1, 2, 3 (rollouts) and 4 |
| `pretrained_models.tar` (optional) | 47 MB | extract into `release_assets/`, giving `release_assets/pretrained_models/` | Stage 4 only, as an extra row; it runs without them |

`release_assets/` is ignored by git. Stage 2 writes the TFRecord shards (3.62 GB) into `release_assets/temporal_pairs_tfrecords/`, so it must run before Stage 3. SHA-256 of `u_yx_planes_h5.tar`: `410518b833edc2fd5bf1b5c569fd2ed35ea15824ed924117958ef8c1edefadd5`. The optional pretrained models are the two networks distributed with the original notebooks; Stage 4 scores them next to mine.

## Setting up and running

I ran everything on a laptop (Apple M3 Max, 48 GB) with Python 3.12, TensorFlow 2.18.1, Keras 3.6 and the `tensorflow-metal` plug-in for the Apple GPU. The frozen environment is in `requirements.txt`; the videos use the ffmpeg binary that `imageio-ffmpeg` installs.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name py312-tf-metal-studio
tar -xf u_yx_planes_h5.tar -C release_assets/
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 02_prepare_temporal_pairs_tfrecords.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=5400 01_unet_turbulent_superresolution.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=5400 03_unet_temporal_evolution.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_baselines_and_rollout.ipynb
```

All paths inside the notebooks are relative to `notebooks/`, so run them from there. Every model is trained when its notebook runs; Stage 4 trains nothing and loads the models of Stages 1 and 3. On the laptop, Stage 2 took under a minute, Stages 1 and 3 side by side on the GPU about 2 h 50 min and 1 h 55 min, and Stage 4 under two minutes.

**Why the GPU.** I timed one training step of each U-Net inside `model.fit`: 90 ms on the Apple GPU against 184 ms on the CPU for the super-resolution network (batch 4), 96 ms against 351 ms for the time-stepping one (batch 8). Training on the GPU is not bit-for-bit repeatable: everything that involves no training reproduces exactly, but a re-run gives slightly different trained numbers, which is one more reason for the seed repeats.

**Why 25 epochs in Stage 1.** The original notebook trains for up to 50 epochs, about 82 minutes here. I cap one training at 45 minutes, so the main training runs 25 epochs; in the original 50-epoch run the validation loss after 25 epochs was within 0.3 % of its final value.

**Why three seeds.** A single training gives one draw of the initial weights and the batch order. Every training that takes 15 minutes or less is repeated with the seeds 0, 1 and 2, and I report the mean and the standard deviation; a difference smaller than that spread is not an effect. The Stage 1 main training (47 minutes) is a single run.

## What the four stages show

- Stage 1's U-Net is far better than the interpolation it starts from, and Stage 4 shows it also beats bicubic upscaling on every one of the 868 test planes, with about a third of bicubic's error, and restores about half of the energy at the scales the coarse grid cannot carry, where bicubic keeps a fifth.
- Stage 3's U-Net beats persistence at one step, but Stage 4 shows why that is not the same as following the flow: the snapshots are 10 time units apart and the velocity fluctuation at one snapshot is essentially uncorrelated with the next. A reference that ignores the current field and predicts the mean velocity profile comes within about 3 % of the U-Net, and fed its own predictions every U-Net falls behind the mean profile from the second step, behind persistence after a number of steps that changes from one training to the next (between 3 and 20 in my runs), and leaves the data entirely over long rollouts.
