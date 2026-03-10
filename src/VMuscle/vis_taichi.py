"""Taichi-based visualizer for muscle simulation."""

import taichi as ti

from VMuscle.muscle_common import get_bbox


@ti.data_oriented
class Visualizer:
    def __init__(self, cfg, muscle=None):
        self.cfg = cfg
        self.muscle = muscle
        self.num_attach_lines = 0
        self.attach_lines_vertices = None

        if self.cfg.render_mode is None or not self.cfg.gui:
            self.cfg.gui = False
            return

        self.res = (1080, 720)
        self.window = ti.ui.Window(self.cfg.name, self.res, vsync=True,
                                   fps_limit=getattr(self.cfg, 'render_fps', 60))
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self.camera.position(0.5, 1.0, 1.95)
        self.camera.lookat(0.5, 0.3, 0.5)
        self.camera.fov(45)
        self.gui = self.window.get_gui()

        if muscle is not None:
            bbox = get_bbox(muscle.pos0_np)
            self._focus_camera_on_model(bbox)

    def _focus_camera_on_model(self, bbox):
        center = (bbox[0] + bbox[1]) * 0.5
        self.camera.position(center[0], center[1], center[2] + 0.5)
        self.camera.lookat(center[0], center[1], center[2])

    def _render_frame(self, step: int, save_image: bool = False):
        if self.cfg.render_mode is None or self.muscle is None:
            return

        self.camera.track_user_inputs(self.window, movement_speed=0.01, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)

        # Render muscle mesh
        self.scene.mesh(vertices=self.muscle.pos, indices=self.muscle.surface_tris_field,
                       color=(0.78, 0.28, 0.26), two_sided=True,
                       show_wireframe=getattr(self.cfg, 'show_wireframe', False))

        # Render bone mesh
        if hasattr(self.muscle, 'bone_indices_field') and self.muscle.bone_indices_field is not None:
            self.scene.mesh(vertices=self.muscle.bone_pos_field, indices=self.muscle.bone_indices_field,
                           color=(0.1, 0.4, 0.8), two_sided=True,
                           show_wireframe=getattr(self.cfg, 'show_wireframe', False))
        elif hasattr(self.muscle, 'bone_pos_field') and self.muscle.bone_pos_field is not None:
            self.scene.particles(self.muscle.bone_pos_field, radius=0.005, color=(0.1, 0.4, 0.8))

        # Render attach lines
        if self.num_attach_lines > 0:
            self.muscle._update_attach_targets_kernel()
            self._update_attach_vis(self.muscle.cons, self.muscle.pos, self.attach_cidx)
            self.scene.lines(vertices=self.attach_lines_vertices, width=1, color=(1.0, 0.0, 0.0))

        with self.gui.sub_window("Options", 0, 0, 0.25, 0.3) as w:
            self.gui.text(f"Step: {step}")
            if self.muscle is not None and self.muscle.step_cnt > 1:
                self.gui.text(f"FPS: {self.muscle.get_fps():.1f}")
                self.gui.text(f"Volume error: {self.muscle.calc_vol_error() * 100:.2f} %")
            if w.button("Pause" if not getattr(self.cfg, 'pause', False) else "Resume"):
                self.cfg.pause = not getattr(self.cfg, 'pause', False)
            if w.button("Toggle Wireframe"):
                self.cfg.show_wireframe = not getattr(self.cfg, 'show_wireframe', False)
            if w.button("Reset Simulation"):
                self.cfg.reset = True
            self.cfg.activation = w.slider_float("activation", self.cfg.activation, 0.0, 1.0)
            if self.muscle is not None:
                self.muscle.activation.fill(self.cfg.activation)

        self.scene.ambient_light((0.4, 0.4, 0.4))
        self.scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.6, 0.6, 0.6))
        self.scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.6, 0.6, 0.6))

        self.canvas.scene(self.scene)
        if self.cfg.render_mode == "human":
            self.window.show()
        if save_image:
            self.window.save_image(f"output/{step:04d}.png")

    def _render_control(self):
        if not self.cfg.gui or self.cfg.render_mode != "human":
            return
        for e in self.window.get_events(ti.ui.PRESS):
            if e.key in [ti.ui.ESCAPE]:
                exit()
            elif e.key == 'r' or e.key == 'R':
                self.cfg.reset = True
            elif e.key == ti.ui.SPACE:
                self.cfg.pause = not getattr(self.cfg, 'pause', False)
            elif e.key == 'g' or e.key == 'G':
                if self.muscle is not None:
                    bbox = get_bbox(self.muscle.pos0_np)
                    self._focus_camera_on_model(bbox)

    @ti.kernel
    def _update_attach_vis(self, cons: ti.template(), pos: ti.template(), attach_cidx: ti.template()):
        for i in range(self.num_attach_lines):
            cidx = attach_cidx[i]
            pts = cons[cidx].pts
            src_pt = pts[0]
            self.attach_lines_vertices[2 * i] = pos[src_pt]
            self.attach_lines_vertices[2 * i + 1] = cons[cidx].restvector.xyz

    def _init_attach_vis(self, attach_constraints):
        if not attach_constraints:
            return
        self.num_attach_lines = len(attach_constraints)
        if self.num_attach_lines > 0:
            self.attach_lines_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.num_attach_lines * 2)
            self.attach_cidx = ti.field(dtype=ti.i32, shape=self.num_attach_lines)
            for i, c in enumerate(attach_constraints):
                self.attach_cidx[i] = c['cidx']
