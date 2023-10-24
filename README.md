# nerfstudio fuzzy metaball reference
This is a [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) implementation of the [Fuzzy Metaball](leonidk.github.io/fmb-plus) rendering methods.

## Registering with Nerfstudio
First, [install nerfstudio](https://docs.nerf.studio/en/latest/quickstart/installation.html). Second, Clone or fork this repository and run the commands:

```
pip install -e .
ns-install-cli
```

## Running the new method
This repository creates a new Nerfstudio method named "fmb". To train with it, run the command:
```
ns-train fmb --data [PATH]
```
