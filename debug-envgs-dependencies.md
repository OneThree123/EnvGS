# EnvGS Dependency Debug Session

- Session: `envgs-dependencies`
- Status: `[OPEN]`
- Symptom: EnvGS startup initially could not import `pkg_resources` and `plyfile`; dependent modules were not registered, leading to a `VolumetricVideoDataloader` registry failure.

## Evidence

- Target runtime: `/data/yibo/envs/envgs/bin/python`, Python 3.10.20, PyTorch `2.7.1+cu128`, CUDA `12.8`, NVIDIA GeForce RTX 5090.
- `setuptools==83.0.0` did not provide the legacy `pkg_resources` module required by EasyVolcap.
- `plyfile` was missing, so `EnvGSSampler` could not import.
- After resolving these Python dependencies, the first model build reached loading `points3D.ply` with `1,038,118` points and then failed because `simple_knn._C.distCUDA2` was missing.
- The host has no `nvcc`, so compiling a new CUDA extension in this environment is not viable.
- A prebuilt CPython 3.10 `simple_knn` extension is available at `/data/yibo/code/IRGS/submodules/simple-knn/simple_knn/_C.cpython-310-x86_64-linux-gnu.so`; it loads successfully with the EnvGS PyTorch runtime.

## Fix and Verification

1. Installed the missing Python dependencies in the target environment:
   ```bash
   /data/yibo/envs/envgs/bin/python -m pip install "setuptools<81" plyfile
   ```
   This installed `setuptools==80.10.2` and `plyfile==1.1.5`.

2. Added an environment-local Python path file:
   ```text
   /data/yibo/envs/envgs/lib/python3.10/site-packages/envgs_simple_knn.pth
   ```
   Its content points to `/data/yibo/code/IRGS/submodules/simple-knn`, making the compatible prebuilt `simple_knn._C` extension importable without copying binaries or changing EnvGS source.

3. Verified the CUDA extension import using the target environment:
   ```bash
   /data/yibo/envs/envgs/bin/python -c "from simple_knn._C import distCUDA2; print('simple_knn OK')"
   ```

4. Updated the Shitang launcher to set both scene and environment point-cloud paths. `EnvGSSampler` requires `env_preload_gs` in addition to `preload_gs`.

5. Completed a real EnvGS `train` construction with `dry_run=True`. It successfully loaded both Gaussian point clouds, created the visualizer, and created an optimizer for `124,574,160` parameters. No dependency or model-construction exception remained.

6. Fixed the no-reflection rendering path in `easyvolcap/models/samplers/envgs_sampler.py`:
   - `store_dif_gaussian_output()` now supplies a zero `spec_map` when `render_reflection=false`, so diffuse output does not access a missing field.
   - Environment reflection tracing is now gated by both `self.render_reflection` and `render_reflection_start_iter`; a reflection-disabled experiment cannot enter that branch after iteration 3000.

7. Installed `lpips==0.1.4`, which is required by the default `PSNR`, `SSIM`, and `LPIPS` evaluator and by the configured perceptual loss after iteration 21000. Confirmed imports of `torch`, `torchvision`, `lpips`, `diff_surfel_tracing`, `diff_surfel_rasterization_wet`, and `simple_knn._C` in the target environment.

8. Ran two real single-iteration training checks successfully:
   - The first verified forward/backward execution and checkpoint saving after the `spec_map` fix.
   - The second forced `render_reflection_start_iter=0` while keeping `render_reflection=false`; it completed and saved checkpoints, directly verifying that the disabled reflection branch is skipped.

## Remaining Scope

- `pytorch3d` remains unavailable, producing import warnings for optional modules. It did not block either verified Shitang EnvGS training run and is not required by the active `NoopNetwork` configuration.
- A full 120-epoch training run and its subsequent test/render evaluation have not yet been performed. They remain required to validate long-duration stability and the final `metrics.json` output.
