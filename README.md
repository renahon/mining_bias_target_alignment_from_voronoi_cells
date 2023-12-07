# Mining bias-target Alignment from Voronoi Cells

This is an example implementation of : Mining bias-target Alignment from Voronoi Cells.

This repository is tailored for Biased-MNIST but can be adapted to other datasets.

## Installations

If you want to work with conda :

```bash
conda create -n voronoi_cells_bias_alignment python=3.9
conda activate voronoi_cells_bias_alignment
pip3 install -r pip_requirements.txt
```

If not, simply install the requirement with :

```bash
pip3 install -r pip_requirements.txt
```

## Run our method on Biased-MNIST

1. In a terminal, get inside the voronoi_cells_bias_alignment folder

2. Run : python3 main.py

### Possible extra arguments #####

- --dev : to specify the device you want to work on (ex: cpu or cuda:0)

- --rho : to select a level of digit-color correlation in Biased-MNIST (ex : 0.997)

- other extra arguments detailed in utils/configs.py
