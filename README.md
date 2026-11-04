# WARP-MC

WARP-MC (Wiener-gain, Attention-over-offsets, Rate-gated, Per-hypothesis Motion Compensation) is a learned per-hypothesis luma interpolation model that replaces the 8-tap DCTIF of VVC/H.266 in the VTM-23.8 reference software, in both the final translational motion compensation and the fractional motion estimation of the encoder. Inference is fixed point and the encoder and decoder reconstructions are bit-exact.

## Layout

| Directory | Content |
| --- | --- |
| `codec/` | VTM-23.8 modifications. `vtm-23.8_warp-mc.patch` is a unified diff against the unmodified VTM-23.8 source tree. `source/` holds the full text of every changed or added file (`NNInterp.{h,cpp}` kernel, `RefCodeMap.{h,cpp}` coding-structure map, hooks in `InterPrediction.{h,cpp}`, `InterSearch.{h,cpp}`, `UnitTools.cpp`, `DeblockingFilter.cpp`, `Picture.h`). `changed_files.txt` lists them. |
| `model/` | PyTorch model (`qkv_mc.py`), fixed-point simulator (`qkv_fixed.py`), exporter to the C++ kernel format (`qkv_export.py`), training script (`nl_tile.py`), BDOF replica used in evaluation (`bdof_replica.py`), BD-rate tool (`bdrate.py`). |
| `kernel_tests/` | Bit-exactness tests for the C++ kernel: `nninterp_test_cls.cpp` (512 reference records), `nninterp_tile_test.cpp` (tile rows against the simulator output), `fuzz.cpp` (fast path against the scalar reference). |
| `export/` | Exported fixed-point models: `export_tile_c16` (16-channel L7, 13,914 parameters) and `export_tile_s0` (32-channel L7). Each directory has `manifest.txt` and little-endian `.bin` arrays. |
| `checkpoints/` | Training checkpoints of the two exported models (seed 0). |
| `scripts/` | Encoding runs: reduced CTC class D (`server19/`), 2-second CTC protocol (`server147/`), training launchers (`train/`). |

## GPU path and proxy evaluation (experimental)

`codec/source/Lib/CommonLib/NNInterpCuda.{h,cu}` is a CUDA batch evaluator that is bit-exact with the scalar reference. Build with `-DNNI_CUDA=ON` (CMake option in `source/Lib/CommonLib/CMakeLists.txt`, CUDA architectures 80 and 120) and enable at run time with `VTM_NNINTERP_GPU=1` and `VTM_NNINTERP_GPU_DEV=<device>`. Several processes sharing one GPU need the MPS daemon. `kernel_tests/gpu/gpu_test.cpp` checks the GPU path against the tile reference and the scalar reference and measures throughput per batch size. `scripts/server147/c2s_run_gpu.sh` runs the 2-second protocol with a mix of GPU and CPU processes.

`codec/proxy_rdo.patch` (against the sources in `codec/source/`) adds `VTM_NNINTERP_PROXY=1`, which evaluates only the linear stages of the model in the screening stages of the encoder search. It is under evaluation and is not used in reported results.

## Build

Apply the patch to a clean VTM-23.8 tree (or copy the files in `codec/source/` over it) and build with OpenMP:

```
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-fopenmp
make -j EncoderApp DecoderApp
```

The kernel is selected at run time through environment variables. `VTM_NNINTERP_DIR` points to an export directory and enables the model. `VTM_NNINTERP_THREADS` sets the number of tile threads per process. `VTM_NNINTERP_CACHE` sets the tile cache capacity (0 disables it). `VTM_NNINTERP_GEOBCW=0` restores the scope without geometric partitioning and non-default BCW blocks. `VTM_NNINTERP_PROF=1` prints the model time split between motion estimation and motion compensation. `NNI_SIMD` caps the SIMD level (0 scalar, 1 AVX2, 2 AVX2 with AVX-VNNI) and `NNI_VERBOSE=1` prints the selected path.

Encoder and decoder must use the same export directory. The decoder reconstruction is compared with the encoder reconstruction in the run scripts.

## Kernel tests

```
g++ -O3 -std=c++17 -msse4.1 -fopenmp -I<VTM>/source/Lib/CommonLib kernel_tests/nninterp_tile_test.cpp codec/source/Lib/CommonLib/NNInterp.cpp -o tile_test
./tile_test export/export_tile_c16 tile.dmp tile_ref.i32      # expects all rows exact
./fuzz export/export_tile_c16 3000 0                            # expects mismatches 0
```

The reference files are produced by `model/qkv_fixed.py --tile <ckpt> <tile.dmp> <export_dir> <ref.i32> --f64`. The dump files are produced by the instrumented decoder (`nninterp_dumpref.cpp`).

## Model export

```
CUDA_VISIBLE_DEVICES= python model/qkv_export.py checkpoints/net_TILE_s0_c16_d5_nd2.pt <calib.dmp> export/export_tile_c16
```

The channel count of L7 is read from the checkpoint. The calibration dump sets the per-layer activation scales.

## Training

`model/nl_tile.py` reads tile dump files (`TRAIN_GLOB`, `TEST_GLOB`) and takes its settings from environment variables (`CH`, `DEPTH`, `ND`, `STEP`, `FILT`, `SEED`, `EPOCHS`, `PATIENCE`, `BS`, `LR`, `TRAIN_PIX`, `STRATA_W`, `OUTDIR`). `scripts/train/runvar.sh CH DEPTH SEED GPU` launches one run.

## License

The VTM modifications follow the license of the VTM software (BSD 3-Clause, JVET). The remaining code is released under the same terms.
