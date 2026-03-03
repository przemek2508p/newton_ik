# Command:
# uv run -m newton.examples basic_urdf
#

import os
import warp as wp
import newton
import newton.examples
import newton.ik as ik

# Ścieżka relatywna do pliku URDF - działa u każdego po sklonowaniu repo
URDF_PATH = os.path.join(os.path.dirname(__file__), "URDF", "A-2085-06.urdf")

class Example:
    def __init__(self, viewer, world_count, args=None):

        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        # -------------------------------------------------
        # Build model
        # -------------------------------------------------
        builder = newton.ModelBuilder()

        builder.add_urdf(
            URDF_PATH,
            xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()),
            floating=False,
        )

        builder.add_ground_plane()

        self.model = builder.finalize()
        self.viewer.set_model(self.model)

        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        # -------------------------------------------------
        # Znajdź indeks końcówki (ostatni link)
        # -------------------------------------------------
        self.ee_index = self.model.body_count - 1

        print(f"[INFO] End-effector index: {self.ee_index}")
        print(f"[INFO] Body label: {self.model.body_label[self.ee_index]}")

        # -------------------------------------------------
        # Gizmo target
        # -------------------------------------------------
        body_q_np = self.state.body_q.numpy()
        self.ee_tf = wp.transform(*body_q_np[self.ee_index])

        # -------------------------------------------------
        # IK Objectives
        # -------------------------------------------------
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0, 0, 0),
            target_positions=wp.array(
                [wp.transform_get_translation(self.ee_tf)],
                dtype=wp.vec3,
            ),
        )

        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array(
                [wp.vec4(*wp.transform_get_rotation(self.ee_tf))],
                dtype=wp.vec4,
            ),
        )

        self.joint_q = wp.array(
            self.model.joint_q,
            shape=(1, self.model.joint_coord_count),
        )

        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj],
            optimizer=ik.IKOptimizer.LBFGS,
        )

    # -------------------------------------------------
    # Update IK target from gizmo
    # -------------------------------------------------
    def _update_target(self):
        self.pos_obj.set_target_position(
            0, wp.transform_get_translation(self.ee_tf)
        )

        q = wp.transform_get_rotation(self.ee_tf)
        self.rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

    # -------------------------------------------------
    def step(self):
        self._update_target()

        self.ik_solver.reset()
        self.ik_solver.step(self.joint_q, self.joint_q, iterations=10)

        self.sim_time += self.frame_dt

    # -------------------------------------------------
    def render(self):
        self.viewer.begin_frame(self.sim_time)

        # gizmo TCP
        self.viewer.log_gizmo("target_tcp", self.ee_tf)

        newton.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            self.state,
        )

        self.viewer.log_state(self.state)
        self.viewer.end_frame()
        wp.synchronize()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--world-count", type=int, default=1)

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args.world_count, args)
    newton.examples.run(example, args)