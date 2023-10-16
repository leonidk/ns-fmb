# nerfstudio fuzzy metaball reference
just a straightforward implementation of the fuzzy metaballs work in nerfstudio


## Registering with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands:

```
pip install -e .
ns-install-cli
```

## Running the new method
This repository creates a new Nerfstudio method named "method-template". To train with it, run the command:
```
ns-train fmb --data [PATH]
```
