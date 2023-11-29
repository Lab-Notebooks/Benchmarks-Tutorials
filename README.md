## Reproducibility for Multiphase Fluid Dynamics Benchmarks

This exceedingly simple two dimensional simulation of a rising bubble took 15 hours 
and 350 processors on a supercomputer, representing 4 seconds of physical time.

**Why did it cost so much?**

__ANSWER__: The resolution of this simulation is approximately 1 mm which corresponds to 
500 points covering the bubble (Diameter = 0.5m). Additionally the Reynolds Number (Re), 
which quantifies relationship between inertial and viscous forces, is very low (35 and 
3.5 in liquid and gas respectively), requiring in a time-step of 0.5 microseconds. This 
results in a configuration dominated by diffusion and is designed to test limitations of 
our open-source software, Flash-X, which uses a fully explicit method for solving Multiphase 
Navier-Stokes equations.

<p align="center"> <img src="analysis/RisingBubble/animation.gif" width="600" style="border:none;background:none;"/> </p>


### Citation

```
@software{akash_dhruv_2023_10215417,
  author       = {Akash Dhruv},
  title        = {{Lab-Notebooks/Outflow-Forcing-Revised: zenodo 
                   archive}},
  month        = nov,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {zenodo},
  doi          = {10.5281/zenodo.10215417},
  url          = {https://doi.org/10.5281/zenodo.10215417}
}
```
