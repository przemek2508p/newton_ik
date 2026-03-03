###########################################################################
# Example First Newton Project
#
# Command:
# uv run -m newton.examples first_newton_project
#
###########################################################################

import math

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder()

        # obie bryły tej samej długości
        link_0 = builder.add_link()
        builder.add_shape_cylinder(link_0, radius=0.5, half_height=1.0)

        link_1 = builder.add_link()
        builder.add_shape_cylinder(link_1, radius=0.5, half_height=1.0)

        # Trzeci dodatkowy
        # link_3 = builder.add_link()
        # builder.add_shape_cylinder(link_1, radius=0.5, half_height=0.8)

        # link_0 — dolny, sztywny
        j0 = builder.add_joint_fixed(
            # parent = -1 (rodzicem jest swiat)
            parent=-1,
            child=link_0,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        )

        # link_1 — górny, zgina się względem dolnego
        j1 = builder.add_joint_revolute(
            parent=link_0,
            child=link_1,
            axis=wp.vec3(1.0, 0.0, 0.0),  # zgięcie w przód/tył
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, -1.0), q=wp.quat_identity()),
        )

        # link_3 — trzeci testowy
        # j3 = builder.add_joint_fixed(
        #     parent=-1,
        #     child=link_0,
        #     parent_xform=wp.transform(p=wp.vec3(2.0, 10.0, 1.0), q=wp.quat_identity()),
        #     child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        # )

        builder.add_articulation([j0, j1], label="cylinders")

        builder.add_ground_plane()

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()

        # zgięcie sinusoidalne — zakres +-90 stopni
        q = self.model.joint_q.numpy()
        q[0] = math.sin(self.sim_time * 2.0) * (math.pi / 2.0)
        self.model.joint_q.assign(q)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)