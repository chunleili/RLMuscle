# Train volumetric muscle with reinforcement learning

It is based on NVIDIA's [Newton](https://github.com/newton-physics/newton) physical engine, but with [my own fork](https://github.com/chunleili/newton) so there might be some differences. 

## Install & Run
Firstly, git clone this repo with submodule. 

```
git clone https://github.com/chunleili/RLMuscle 
git submodule update --init --recursive
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) with following script:

``` sh
# Linux or macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install the package with:

```
uv sync 
```

or with optional dependencies (for OpenSim comparison):

```
uv sync --extra optional
```

Then run the example with:
```
uv run main.py 
```

(Optional) To run a different example, set environment variable `RUN` or change .env file. The RUN case names are exactly the same as those in `examples/*`. E.g.,
```sh
# Linux or macOS
RUN=example_couple3 uv run main.py

# Windows (PowerShell)
$env:RUN = "example_couple3"; uv run main.py
```

(Optional) You can also use `uv run -m examples.example_XXX` to run an example.


Output (if any) will be saved in the "output" directory.

## Assets download
We use Git LFS to manage large assets. If you have git-lfs installed and LFS smudge is enabled (default), then the assets will be downloaded automatically when you git clone. Otherwise you should first install git lfs run
```
git lfs pull
```

## Roadmap
- physical engine
    - [x] Implement a minimal joint demo using newton
    - [x] USD IO
    - [x] Add muscle coupling solver
    - [x] Hill-type volumetric muscle model 
      - [x] VBD: DeGrooteFregly2016. Two examples: slidingBall and simpleArm. Compared against OpenSim.
      - [x] XPBD: DeGrooteFregly2016; Millard2012. Three examples: slidingBall、simpleArm and bicep (couple3). The first two compared against OpenSim. Prefer XPBD over VBD.
- reinforcement learning
    - [ ] Implement a simple RL task (simpleArm)
- final stage
    - [ ] Full body simulation with RL control

## Examples & Experiments

### Importing a Full Human Skeleton/Muscle from USD
![import_human](docs/imgs/import_human.png)

`uv run -m examples.example_human_import` 

### Sliding Ball Comparison against OpenSim
Run the sliding ball muscle experiment and generate force/displacement comparison curves:
```
uv run python scripts/run_sliding_ball_comparison.py
```
Results (plots and data) are saved to `output/`. If `pyopensim` is installed (`uv sync --extra optional`), the script also runs an OpenSim reference simulation for comparison. Sample output plots are shown below:
![sliding_ball_comparison](docs/imgs/sliding_ball_curve.png)

This example corresponds to a sliding ball lifted by a single muscle.
<p float="left">
  <img src="./docs/imgs/sliding_ball_osim.gif" width="40%" />
  <img src="./docs/imgs/sliding_ball_vbd.gif" width="49%" />
</p>

### Simple Arm Comparison against OpenSim
```
uv run python scripts/run_simple_arm_comparison.py --mode xpbd-millard
uv run python scripts/run_simple_arm_comparison.py --mode xpbd-dgf
```

![simple_arm_comparison](docs/imgs/simple_arm_xpbd_millard_vs_osim.png)

<p float="left">
  <video src="./docs/imgs/simpleArm-osim-motion.mp4" width="47%" autoplay loop muted playsinline></video>
  <video src="./docs/imgs/simpleArm-xpbd-millard.mp4" width="49%" autoplay loop muted playsinline></video>
</p>

XPBD coupled simple arm (Millard Hill-type muscle + MuJoCo rigid body):

```
RUN=example_xpbd_coupled_simple_arm uv run main.py
```
<p>
  <video src="./docs/imgs/example_xpbd_coupled_simple_arm.5s.mp4" width="49%" autoplay loop muted playsinline></video>
</p>

### Bicep: Muscle-Bone Coupling (couple3)

XPBD-Millard volumetric bicep coupled with Newton MuJoCo elbow joint:

```
RUN=example_couple3 uv run main.py --auto --steps 300
```
<p>
  <video src="./docs/imgs/example_couple3.anim.5s.mp4" width="49%" autoplay loop muted playsinline></video>
</p>

## Test
You can run all the tests with:
```
uv run pytest -v
```

Or you can run a specific test file with:
```
uv run python tests/xxx.py
```


## Note

### Symbol Table
See [docs/notes/symbols.md](docs/notes/symbols.md) for the unified symbol table used across code and documentation.

### Docs
`/docs` is not a user guide but a collection of documentation files during development. We keep all changes and progress in `docs/` to make it easier to track. The structure is as follows:
- `docs/progress/*.md`: logs (experiment / plan execution progress; completed + next steps)
- `docs/plans/*.md`: plans
- `docs/specs/`: detailed specifications
- `docs/experiments/*.md`: experiment summaries
- `docs/notes/*.md`: theory (formula) notes   


### Layered USD
There are two ways to export USD. 1: Adding layers on top of an existing usd file. 2: construct a new usd file from scratch. The layered approach is better when you want to do a non-destructive editing, or adding different animations to the same source. See `examples/example_usd_io.py` for an example. But the second one is also useful when your scene is completely new.


### up-axis
 USD and Houdini use Y up by default. But Newton uses **Z up** by default. See [here](https://newton-physics.github.io/newton/latest/concepts/conventions.html#coordinate-system-and-up-axis-conventions) for newton's convention. We will **transfer the asset to Z up when loading it** (turn off by switching off "y_up_to_z_up"). Be careful when importing other assets.

## macOS Related
If you are simultaneously using Taichi and Warp, you have to first initialize Warp (`wp.init()`) then import Taichi, otherwise their LLVM will conflict with each other. Taichi is going to be deprecated. We will freeze the code related to Taichi and focus on Warp.
