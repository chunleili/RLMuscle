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
    root_path="/Human",                    # 如果你的场景主层级在 /World，就写 "/World"
    # apply_up_axis_from_stage=True,    # 按 USD 的 upAxis 处理（Houdini/Omni 的 upAxis 不一致时很有用）
    skip_mesh_approximation = True, #This will show detailed bone mesh, but it will slow down the program dramatically.
)
builder.add_ground_plane()
model = builder.finalize()

# 2) 分配状态/接触/控制
state_0 = model.state()
state_1 = model.state()
control = model.control()
contacts = model.collide(state_0)

# 3) 选一个 solver（最简单先用 XPBD）
solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True)  
newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

initial_state = model.state()
initial_state.assign(state_0)

# 4) 每帧多子步推进
fps = 60
frame_dt = 1.0 / fps
substeps = 10
dt = frame_dt / substeps

# 5) 导出为 time-sampled USD（可用 usdview / Omniverse / Houdini 打开回放）
# viewer = newton.viewer.ViewerUSD(output_path=OUTPUT_USD, fps=fps, up_axis="Z")
viewer = newton.viewer.ViewerGL()  
viewer.set_model(model)
viewer._paused = True  # 先暂停，等渲染窗口打开了再开始模拟

t = 0.0
num_frames = 10000000  # 4 秒
for _ in range(num_frames):
    for _ in range(substeps):
        state_0.clear_forces()
        contacts = model.collide(state_0)
        if not viewer.is_paused():
            solver.step(state_in=state_0, state_out=state_1, control=control, contacts=contacts, dt=dt)
        if viewer.is_key_down("r"):
            state_0.assign(initial_state)
            state_1.assign(initial_state)
            viewer._paused = True  
        state_0, state_1 = state_1, state_0

    viewer.begin_frame(t)
    viewer.log_state(state_0)
    viewer.end_frame()
    t += frame_dt

viewer.close()
print(f"Saved: {Path(OUTPUT_USD).resolve()}")
