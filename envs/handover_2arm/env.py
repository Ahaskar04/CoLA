import numpy as np
import mujoco
import os

os.environ.setdefault('MUJOCO_GL', 'glfw')


class Handover2ArmEnv:
    """
    2-arm Aloha handover task environment.

    Wraps the Aloha dual-arm MJCF scene (scene.xml).
    reset() randomizes the box's spawn position on the table.
    step() advances physics by one control step given an action.
    """

    def __init__(self, xml_path=None):
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), 'scene.xml')

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # cache the qpos address for the box's freejoint once, rather than
        # re-looking it up by name every reset
        self._box_qpos_addr = self.model.joint('middle_box_joint').qposadr[0]

        # bounds for random box spawn, in meters, centered on the table
        # (table collision geom is size="0.61 0.37 0.1" centered at origin --
        #  keep some margin so the box never spawns off the edge or inside an arm base)
        self._box_x_range = (-0.25, 0.25)
        self._box_y_range = (-0.20, 0.20)
        self._box_spawn_height = 0.03  # sits flush on table surface (z ~ 0)

        self.n_actuators = self.model.nu

    def reset(self, seed=None):
        """Reset the simulation and randomize the box's spawn position."""
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)

        rand_x = np.random.uniform(*self._box_x_range)
        rand_y = np.random.uniform(*self._box_y_range)

        # qpos for a freejoint is [x, y, z, qw, qx, qy, qz] (7 values) --
        # we only touch position (first 3); leave orientation at its XML default
        addr = self._box_qpos_addr
        self.data.qpos[addr:addr + 3] = [rand_x, rand_y, self._box_spawn_height]

        # propagate the manually-set qpos into derived quantities
        # (geom_xpos etc.) before anything reads or renders the state
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs()

    def step(self, action):
        """
        Apply an action (one target value per actuator, matching
        self.n_actuators / model.nu) and advance physics by one control step.
        """
        assert len(action) == self.n_actuators, (
            f'action length {len(action)} does not match n_actuators {self.n_actuators}'
        )
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward()
        done = False  # task-specific termination logic goes here later
        info = {}

        return obs, reward, done, info

    def _get_obs(self):
        """
        Build the observation vector. Placeholder for now -- fill in with
        whatever proprioceptive + object-state features COLA's policy needs
        (joint qpos/qvel per arm, box position, gripper site positions, etc.)
        """
        return {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
            'box_pos': self.data.qpos[self._box_qpos_addr:self._box_qpos_addr + 3].copy(),
        }

    def _compute_reward(self):
        """Placeholder -- task-specific success/reward logic goes here."""
        return 0.0


if __name__ == '__main__':
    # quick manual sanity check: reset a few times, print box spawn positions
    env = Handover2ArmEnv()
    obs = env.reset()
    print(f'box spawned at: {obs["box_pos"]}')
    import mujoco.viewer
    # mujoco.viewer.launch(env.model, env.data)
    print('joint names:', [env.model.joint(i).name for i in range(env.model.njnt)])
    print('total joints (njnt):', env.model.njnt)