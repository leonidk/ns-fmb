# nerfstudio fuzzy metaball reference
This is a [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) implementation of the [Fuzzy Metaball](leonidk.github.io/fmb-plus) rendering methods. The enable optimization with purely color-based losses, this includes a far-field NeRF to model the background. The Gaussians then organize into modeling just the foreground object. 

## Result
Example reconstruction of Nerfstudio plane and sculpture sequences with **40 Gaussians**. This using posed color images (no masks, no flow) and random initialization from a small sphere, with a few thousand iterations of optimization. 

https://github.com/leonidk/ns-fmb/assets/645763/bd7bee47-83f3-4f7f-835f-19e99ff49204

https://github.com/leonidk/ns-fmb/assets/645763/a94f5f25-7c1c-4439-8262-3d62ece113fe

You can see how the forty Gaussians distribute themselves to model the foreground. Of note, some Gaussians are used to model the shadows and the specularities and other color-specific features. 


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

## Notes

* **Good**: Color-based optimization works! This wasn't tested in either [prior](https://leonidk.github.io/fuzzy-metaballs/) [paper](https://leonidk.github.io/fmb-plus/).
* **Good**: Many background models work. NeRF interpolates well, but even a texture atlas works well on i.i.d. evaluation. 
* **Bad**: It is slow. The current torch implementation seems imperfect. On the same machine, this might get 150-300K rays per second (with simple background), but the JAX [codebase](https://leonidk.github.io/fmb-plus/) gets around 2M train ray/s (even with extra losses like flow!).  
