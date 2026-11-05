/* yvvyee: CUDA batched tile evaluator for NNInterp (QKV-MC luma interpolation, L1..L8).
 * Bit-exact with NNInterp::predictRef for every output (y, ylin, xd). One thread block per tile.
 * Usage: NNInterpCuda g; g.load(export_dir); g.predictBatch(in, out, n, NNInterpCuda::OUT_Y);
 * in/out are host arrays; pinned memory (allocPinned) avoids a staging copy inside the driver.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include "NNInterp.h"

class NNInterpCuda
{
public:
  using BlockIn = NNInterp::BlockIn;
  using BlockOut = NNInterp::BlockOut;
  enum : int { OUT_Y = 1, OUT_YLIN = 2, OUT_XD = 4, OUT_ALL = 7 };   // yvvyee: which BlockOut fields are copied back (the encoder needs only y)

  NNInterpCuda() = default;
  ~NNInterpCuda();
  NNInterpCuda(const NNInterpCuda &) = delete;
  NNInterpCuda &operator=(const NNInterpCuda &) = delete;

  bool load(const std::string &dir, int device = 0);   // manifest.txt + .bin; false if missing arrays or a bit-exactness precondition fails
  bool enabled() const { return m_loaded; }
  // yvvyee: evaluates n tiles (one H2D copy, one kernel, one D2H copy, then stream sync). false on CUDA error or invalid tile geometry.
  bool predictBatch(const BlockIn *in, BlockOut *out, int n, int outMask = OUT_ALL, bool linearOnly = false);
  // yvvyee: linearOnly = proxy evaluation (NNInterp::predict(in, out, true)): L1-L6 only, out.y = tile area of y_lin (bit-exact), ylin/xd as in the full path
  size_t deviceBytes() const;      // device memory allocated by this object (parameters + batch buffers)
  const char *lastError() const { return m_err; }

  static void *allocPinned(size_t bytes);
  static void freePinned(void *p);

private:
  struct Impl;
  Impl *m = nullptr;
  bool m_loaded = false;
  char m_err[256] = { 0 };
};
