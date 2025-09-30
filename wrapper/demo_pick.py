# demo_pick.py
import os
import time
import imageio
import numpy as np

from genesis_pick_env import GenesisPickEnv


def record(env: GenesisPickEnv, frames, steps=60):
    """Collect some frames from the offscreen camera while sim runs."""
    for _ in range(steps):
        # keep holding current target; just advance physics
        obs, r, d, info = env.step(env.robot.get_dofs_position())
        frames.append(env.render())


def main():
    env = GenesisPickEnv(show_viewer=False)   # set False if you need headless
    obs = env.reset()

    frames = []
    frames.append(env.render())

    print("Running scripted pick...")
    ok = env.scripted_pick_once()
    print("Grasp success (heuristic):", ok)

    # hang around for a second so you can watch it in the viewer, also collect frames
    record(env, frames, steps=120)

    # Save a short MP4 so you can confirm it worked
    out_path = "pick_demo.mp4"
    print(f"Writing video to {out_path} ...")
    imageio.mimsave(out_path, frames, fps=int(1.0 / env.dt))
    print("Done.")


if __name__ == "__main__":
    main()
