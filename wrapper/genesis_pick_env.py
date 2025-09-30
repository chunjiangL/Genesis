# genesis_pick_env.py
import math
import time
import numpy as np

import genesis as gs


class GenesisPickEnv:
    """
    Minimal, Gym-like wrapper around Genesis for a pick task with a 7-DoF arm + 2-DoF gripper.
    - Asset: Franka Panda (MJCF) by default; adjust ROBOT_MJCF if needed.
    - Object: a small box to pick.
    - Control: joint position targets (7 arm joints + 2 finger openings).
    - Obs: joint state, EE pose, object pose; camera RGB available via render().

    API:
      env = GenesisPickEnv()
      obs = env.reset()
      obs, reward, done, info = env.step(action)  # action: np.ndarray (9,)
      rgb = env.render()                           # returns np.uint8 [H,W,3]
    """

    # Adjust this path if your local Genesis install uses a different sample asset layout.
    ROBOT_MJCF = "xml/franka_emika_panda/panda.xml"

    def __init__(
        self,
        show_viewer: bool = False,
        dt: float = 0.01,
        cam_size=(640, 480),
        object_size=(0.04, 0.04, 0.04),
        object_start=(0.55, 0.00, 0.05),
        seed: int = 0,
    ):
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        # ---- Robust backend init: try GPU, fall back to CPU
        try:
            gs.init(backend=gs.gpu)
        except Exception as e:
            print("[WARN] GPU backend failed; falling back to CPU. Reason:", e)
            gs.init(backend=gs.cpu)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt),
            rigid_options=gs.options.RigidOptions(dt=dt),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.6, -0.9, 1.0),
                camera_lookat=(0.6, 0.0, 0.2),
                camera_fov=50,
                max_FPS=60,
            ),
            show_viewer=show_viewer,
        )

        # --- world & assets ---
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=self.ROBOT_MJCF))
        self.box = self.scene.add_entity(
            gs.morphs.Box(size=object_size, pos=object_start)
        )

        # Camera for RGB capture (independent of viewer)
        # Genesis 0.3.x: no 'name' kwarg; some builds use width/height, others 'res'.
        try:
            self.cam = self.scene.add_camera(
                pos=(1.4, -0.6, 0.85),
                lookat=(0.6, 0.0, 0.2),
                fov=55,
                width=cam_size[0],
                height=cam_size[1],
            )
        except TypeError:
            # Fallback signature with 'res=(w,h)'
            self.cam = self.scene.add_camera(
                pos=(1.4, -0.6, 0.85),
                lookat=(0.6, 0.0, 0.2),
                fov=55,
                res=(cam_size[0], cam_size[1]),
            )

        self.scene.build()

        # indices: 7 arm joints + 2 fingers
        self.arm_dofs = np.arange(7)
        self.grip_dofs = np.arange(7, 9)

        # PD gains & torque limits (tuned for Franka; adjust if you swap robots)
        self.robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self.robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        self.robot.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
        )

        # link handle for end-effector (Franka hand link name in sample MJCF)
        self.ee = self.robot.get_link("hand")

        # book-keeping
        self._step_count = 0

    # ---------- utilities ----------
    def _step_sim(self, n=1):
        for _ in range(n):
            self.scene.step()

    def _home_pose(self):
        """Nominal comfortable joint configuration + open gripper."""
        return np.array([0.0, -0.4, 0.0, -1.8, 0.0, 1.5, 0.0, 0.045, 0.045], dtype=np.float32)

    def _randomize_object_xy(self):
        # jitter XY a bit on the table
        x = 0.52 + self.rng.uniform(-0.05, 0.05)
        y = 0.00 + self.rng.uniform(-0.05, 0.05)
        self.box.set_pos(np.array([x, y, 0.02], dtype=np.float32))

    def _ik_to(self, pos_xyz, quat_wxyz=(0, 1, 0, 0), settle=120):
        qpos = self.robot.inverse_kinematics(link=self.ee, pos=np.array(pos_xyz), quat=np.array(quat_wxyz))
        # keep current fingers
        cur = self.robot.get_dofs_position()
        qpos[-2:] = cur[-2:]
        # command arm joints only
        self.robot.control_dofs_position(qpos[:-2], self.arm_dofs)
        self._step_sim(settle)

    def open_gripper(self, width=0.045, hold=60):
        q = self.robot.get_dofs_position().copy()
        q[-2:] = width
        self.robot.control_dofs_position(q)
        self._step_sim(hold)

    def close_gripper(self, force=0.8, hold=120):
        # negative force closes the two fingers in Genesis Panda example
        self.robot.control_dofs_force(np.array([-force, -force]), self.grip_dofs)
        self._step_sim(hold)

    # ---------- public API ----------
    def reset(self):
        """Reset robot to home, randomize box, return observation dict."""
        self.robot.control_dofs_position(self._home_pose())
        self._step_sim(300)

        self._randomize_object_xy()
        self._step_sim(60)

        self._step_count = 0
        return self._obs()

    def _obs(self):
        q = self.robot.get_dofs_position()
        dq = self.robot.get_dofs_velocity()
        ee_p = self.ee.get_pos()
        ee_q = self.ee.get_quat()  # wxyz
        obj_p = self.box.get_links_pos()[0]
        obj_v = self.box.get_links_vel()[0]
        return {
            "q": q.copy(),
            "dq": dq.copy(),
            "ee_pos": ee_p.copy(),
            "ee_quat": ee_q.copy(),
            "obj_pos": obj_p.copy(),
            "obj_vel": obj_v.copy(),
            "t": self._step_count * self.dt,
        }

    def step(self, action):
        """
        action: np.ndarray shape (9,) = [7 arm joint targets, 2 finger openings]
        """
        assert isinstance(action, np.ndarray) and action.shape == (9,)
        self.robot.control_dofs_position(action)
        self._step_sim(1)
        self._step_count += 1

        obs = self._obs()
        # Simple dense reward just for sanity checks (optional)
        dist = np.linalg.norm(obs["ee_pos"] - obs["obj_pos"])
        reward = -dist
        done = False
        info = {"ee_obj_dist": float(dist)}
        return obs, reward, done, info

    def render(self):
        """
        Returns an RGB frame from the offscreen camera as np.uint8 [H,W,3].
        Handles minor API differences across Genesis versions.
        """
        try:
            out = self.cam.render(["rgb"])      # common in 0.3.x
            rgb = out["rgb"]
        except Exception:
            try:
                rgb = self.cam.render("rgb")    # some builds accept a single string and return the array
            except Exception:
                # last resort: scene render (might not match camera pose exactly)
                out = self.scene.render(["rgb"])
                rgb = out["rgb"]
        return rgb

    # ---------- scripted pick (for the demo) ----------
    def scripted_pick_once(self, approach_h=0.25, grasp_h=0.10, lift_h=0.30):
        """
        A minimal open-loop pick sequence using IK + gripper force close:
        1) go home & open gripper
        2) go above object (approach height)
        3) descend to grasp height
        4) close gripper
        5) lift a bit
        Returns True if it likely grasped (very naive check via finger distance).
        """
        self.robot.control_dofs_position(self._home_pose())
        self._step_sim(200)
        self.open_gripper(0.045, hold=60)

        obj = self.box.get_links_pos()[0]
        above = (float(obj[0]), float(obj[1]), float(approach_h))
        self._ik_to(above, settle=160)

        near = (float(obj[0]), float(obj[1]), float(grasp_h))
        self._ik_to(near, settle=140)

        self.close_gripper(force=0.9, hold=140)

        # lift
        lift = (float(obj[0]), float(obj[1]), float(lift_h))
        self._ik_to(lift, settle=160)

        # naive success heuristic: finger distance smaller than open width and object z increased
        fingers = self.robot.get_dofs_position()[-2:]
        finger_gap = float(np.mean(fingers))
        obj_new = self.box.get_links_pos()[0]
        lifted = obj_new[2] > (grasp_h + 0.05)
        return (finger_gap < 0.03) and bool(lifted)
