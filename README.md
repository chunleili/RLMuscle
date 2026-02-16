# Train volumetric muscle with reinforcement learning

It is based on NVIDIA's [Newton](https://github.com/newton-physics/newton) physical engine, but with [my own fork](https://github.com/chunleili/newton) so there might be some differences. 

# Run
Firstly, git clone this repo and [my fork](https://github.com/chunleili/newton) of newton. 

```
git clone https://github.com/chunleili/RLMuscle
cd RLMuscle
git clone https://github.com/chunleili/newton external/newton
```

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

Or run the USD IO teaching example directly:
```
.\.venv\Scripts\python.exe examples\example_usd_io.py --viewer gl
```


# Roadmap
- physical engine
    - [x] Implement a minimal joint demo using newton
    - [x] USD IO with layering
    - [ ] Add muscle coupling solver
- reinforcement learning
    - [ ] Implement a simple RL task


# Note

## visualization & USD IO

### Layered USD
Use **"--use-layered-usd"** to enable the layered USD export. This is better than the newton's usd viewer because it just adding layers on top of the original usd file, which is the correct way to use usd. So it is incompatible with the "--viewer usd" and has to be used with usd as input. 

You can also specify "--copy-usd" to copy the input usd file to the output directory, which is useful when you want to move and share the usd since the usd use relative path to reference the input usd file.


### Headless mode
You can run the USD IO example in headless mode:
```
.\.venv\Scripts\python.exe examples\example_usd_io.py --viewer null --headless --num-frames 100 --use-layered-usd
```
It will automatically save the layered usd file after 100 frames.

# Caution
## up-axis
 USD and Houdini use Y up by default. But Newton uses **Z up** by default. See [here](https://newton-physics.github.io/newton/latest/concepts/conventions.html#coordinate-system-and-up-axis-conventions) for newton's convention. We will **transfer the asset to Z up when loading it** (turn off by switching off "y_up_to_z_up"). Be careful when importing other assets.
