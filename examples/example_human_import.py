from pathlib import Path
import warp as wp
import newton

# 可选：如果你 USD 里用到了 newton:* schema（例如 NewtonSceneAPI 等）
# 注意要尽早 import
try:
    import newton_usd_schemas 
except Exception:
    pass

# 如果你的场景里有 Mesh，需要 newton.usd 来做 mesh ingestion（推荐直接 import）
import newton.usd

INPUT_USD = (Path(__file__).resolve().parents[1] / "data/Human/Human.usd")  # 改成你的 USD
OUTPUT_USD = "newton_sim_out.usd"     # 模拟输出 USD（带动画）

# 1) 导入 USD -> Newton Model
builder = newton.ModelBuilder()

builder.add_usd(
    source=INPUT_USD.as_posix(),  # USD 文件路径
    root_path="/Human/Ragdoll",                    # 如果你的场景主层级在 /World，就写 "/World"
    apply_up_axis_from_stage=True,    # 按 USD 的 upAxis 处理（Houdini/Omni 的 upAxis 不一致时很有用）
)

model = builder.finalize()

# 2) 分配状态/接触/控制
state_0 = model.state()
state_1 = model.state()
control = model.control()
contacts = model.collide(state_0)

# 3) 选一个 solver（最简单先用 XPBD）
solver = newton.solvers.SolverXPBD(model, iterations=10)  

# 4) 每帧多子步推进
fps = 60
frame_dt = 1.0 / fps
substeps = 10
dt = frame_dt / substeps

# 5) 导出为 time-sampled USD（可用 usdview / Omniverse / Houdini 打开回放）
viewer = newton.viewer.ViewerUSD(output_path=OUTPUT_USD, fps=fps, up_axis="Z")
viewer.set_model(model)

t = 0.0
num_frames = 240  # 4 秒
for _ in range(num_frames):
    for _ in range(substeps):
        state_0.clear_forces()
        contacts = model.collide(state_0)
        solver.step(state_in=state_0, state_out=state_1, control=control, contacts=contacts, dt=dt)
        state_0, state_1 = state_1, state_0

    viewer.begin_frame(t)
    viewer.log_state(state_0)
    viewer.end_frame()
    t += frame_dt

viewer.close()
print(f"Saved: {Path(OUTPUT_USD).resolve()}")
