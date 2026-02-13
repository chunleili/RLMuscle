# Train volumetric muscle with reinforcement learning

It is based on NVIDIA's [Newton](https://github.com/newton-physics/newton) physical engine, but with [my own fork](https://github.com/chunleili/newton) so there might be some differences. 

# Run
Install [uv](https://docs.astral.sh/uv/getting-started/installation/). 

Then install the package with:

```
uv sync --extra viewer
```
If you do not need the gui, you can just run `uv sync` without the extra.

Then run the example with:
```
uv run main.py 
```

# Roadmap
- physical engine
    - [x] Implement a minimal joint demo using newton
    - [x] USD IO with layering
    - [ ] Add muscle coupling solver
- reinforcement learning
    - [ ] Implement a simple RL task

# Caution
- up-axis: USD and Houdini use Y up by default. But Newton uses Z up by default. See [here](https://newton-physics.github.io/newton/latest/concepts/conventions.html#coordinate-system-and-up-axis-conventions) for newton's convention. We use transfer the asset to Z up when loading it (turn off by switch off "y_up_to_z_up"). Be careful when importing other assets.