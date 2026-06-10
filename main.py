import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
from tqdm import tqdm
from humanoid_task import Humanoid
from cartpole_task import CartpoleSwingUp
from evosax.algorithms import CMA_ES, Sep_CMA_ES
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

# Config
N_ENVS = 128


task_name = "humanoid"


if task_name == "humanoid":
    task = Humanoid(N_ENVS)
elif task_name == "cartpole":
    task = CartpoleSwingUp(N_ENVS)


control_init = task.get_warm_start()

# Instantiate the search strategy
es = CMA_ES(population_size=N_ENVS, solution=control_init)
es.elite_ratio = 0.1
params = es.default_params
params = params.replace(std_init=0.25)

# Initialize state
key = jax.random.key(0)
state = es.init(key, control_init, params)


num_generations = 250
metrics_log = []
# Ask-Eval-Tell loop
for i in tqdm(range(num_generations)):
    key, key_ask, key_eval = jax.random.split(key, 3)

    # Generate a set of candidate solutions to evaluate
    population, state = es.ask(key_ask, state, params)

    # Evaluate the fitness of the population
    population_np = np.array(population)
    # print(np.max(population_np))
    fitness_np = task.cost_function(population_np)
    fitness = jnp.array(fitness_np)

    # Update the evolution strategy
    state, metrics = es.tell(key, population, fitness, state, params)
    metrics_log.append(metrics)



# Extract the best fitness values across generations
generations = [metrics["generation_counter"] for metrics in metrics_log]
best_fitness = [metrics["best_fitness"] for metrics in metrics_log]


print("Final fitness = ", best_fitness[-1])


plt.figure(figsize=(10, 5))
plt.plot(generations, best_fitness, label="Best Fitness", marker="o", markersize=3)

os.makedirs("Figures", exist_ok=True)
plt.title("Best fitness over generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Figures/cost.png")
# plt.show()







VISUALIZE = True

if VISUALIZE :
    control_traj = state.best_solution.reshape((task.N_STEPS, -1))
    control_traj = np.array(control_traj)


    import imageio
    import mujoco

    show_reference = True

    data = mujoco.MjData(task.mj_model)
    dt = task.dt
    fps = int(round(1.0 / dt))
    data.qpos = task.q_init.copy()
    mujoco.mj_forward(task.mj_model, data)
    cam = mujoco.MjvCamera()
    if task_name == "humanoid":
        cam.lookat[:] = data.qpos[:3] 
        cam.elevation = -15   

    q_traj = []
    v_traj = []

    with mujoco.Renderer(task.mj_model, width=640, height=480) as renderer:
        with imageio.get_writer("Figures/rollout.mp4", fps=fps) as writer:

            if show_reference:
                ref_data = mujoco.MjData(task.mj_model)
                ref_data.qpos[:] = task.q_desired.copy()
                mujoco.mj_forward(task.mj_model, ref_data)

                renderer.update_scene(ref_data, camera=cam)
                ghost_img = renderer.render()

            for t in range(task.N_STEPS):
                q_traj.append(data.qpos.copy())
                v_traj.append(data.qvel.copy())
                for _ in range(task.N_HOLD):
                    data.ctrl[:] = control_traj[t]
                    mujoco.mj_step(task.mj_model, data) 

                    renderer.update_scene(data, camera=cam)
                    pixels = renderer.render()

                    if show_reference:
                        alpha = 0.3
                        pixels = (
                            (1 - alpha) * pixels.astype(np.float32)
                            + alpha * ghost_img
                        )
                        pixels = np.clip(pixels, 0, 255).astype(np.uint8)

                    writer.append_data(pixels)

            q_traj.append(data.qpos.copy())
            v_traj.append(data.qvel.copy())

    q_traj = np.array(q_traj)
    v_traj = np.array(v_traj)
    if task_name == "cartpole":
        task.plot_solution(q_traj, v_traj, control_traj)