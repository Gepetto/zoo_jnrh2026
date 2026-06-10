{
  lib,
  buildPythonPackage,
  evosax,
  imageio,
  ipykernel,
  jax,
  jupyterlab,
  matplotlib,
  mujoco,
  numpy,
  tqdm,
}:
buildPythonPackage {
  name = "zoo-tuto";
  version = "0.1.0";
  pyproject = true;

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./cartpole_task.py
      ./class_notebook.ipynb
      ./humanoid_task.py
      ./main.py
      ./model
      ./pyproject.toml
    ];
  };

  dependencies = [
    evosax
    imageio
    ipykernel
    jax
    jupyterlab
    matplotlib
    mujoco
    numpy
    tqdm
  ]
  ++ imageio.optional-dependencies.ffmpeg;
}
