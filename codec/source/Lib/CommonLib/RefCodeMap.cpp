/* yvvyee: 참조 화면 부호화 구조 지도 구현. VTM-11.0_oracle DecPatchDump::buildMap 과 같은 규칙. */
#include "RefCodeMap.h"
#include "CodingStructure.h"
#include "Picture.h"
#include "UnitTools.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

static std::map<const Picture *, std::vector<uint32_t>> g_bs[2];   // 화면 -> (y4<<13 | x4<<2 | bS) 목록 (VER, HOR)

void RefCodeMap::beginPicture(const Picture *pic)
{
  g_bs[0][pic].clear();
  g_bs[1][pic].clear();
}

void RefCodeMap::noteBS(const Picture *pic, int edgeDir, int x, int y, int bs)
{
  g_bs[edgeDir & 1][pic].push_back((uint32_t) (((y >> 2) << 13) | ((x >> 2) << 2) | (bs & 3)));
}

uint16_t RefCodeMap::unit(int x, int y) const
{
  const int ux = std::min(std::max(x, 0), w4 * 4 - 1) >> 2, uy = std::min(std::max(y, 0), h4 * 4 - 1) >> 2;
  return u[(size_t) uy * w4 + ux];
}

uint16_t RefCodeMap::at(int x, int y) const
{
  if (!valid()) return 0;
  const int cx = std::min(std::max(x, 0), w4 * 4 - 1), cy = std::min(std::max(y, 0), h4 * 4 - 1);
  const uint16_t v0 = unit(cx, cy);
  uint16_t v = (uint16_t) (v0 & 0x7FC0);
  if ((cx & 3) == 0) v |= (uint16_t) (v0 & 0x000D);
  if ((cy & 3) == 0) v |= (uint16_t) (v0 & 0x0032);
  return v;
}

void RefCodeMap::build(const CodingStructure &cs)
{
  const Picture *pic = cs.picture;
  const int W = pic->lwidth(), H = pic->lheight();
  w4 = (W + 3) >> 2; h4 = (H + 3) >> 2;
  u.assign((size_t) w4 * h4, 0);
  for (const CodingUnit *cu: cs.cus)
  {
    if (!cu->Y().valid()) continue;
    const uint16_t intra = CU::isIntra(*cu) ? 1 : 0;
    const uint16_t qp    = (uint16_t) std::min(63, std::max(0, (int) cu->qp));
    int nTu = 0;
    auto fill = [&](const Area &a, uint16_t cbf) {
      const int x0 = a.x >> 2, y0 = a.y >> 2, x1 = std::min(w4, (int) (a.x + a.width + 3) >> 2), y1 = std::min(h4, (int) (a.y + a.height + 3) >> 2);
      for (int uy = y0; uy < y1; uy++)
        for (int ux = x0; ux < x1; ux++)
        {
          uint16_t v = (uint16_t) ((1 << 14) | (qp << 8) | (intra << 7) | (cbf << 6));
          if (ux == x0) v |= 1;
          if (uy == y0) v |= 2;
          u[(size_t) uy * w4 + ux] = v;
        }
    };
    for (const auto &tu: CU::traverseTUs(*cu))
    {
      if (!tu.Y().valid()) continue;
      fill(tu.Y(), TU::getCbf(tu, COMPONENT_Y) ? 1 : 0);
      nTu++;
    }
    if (nTu == 0) fill(cu->Y(), 0);
  }
  for (int d = 0; d < 2; d++)
  {
    auto it = g_bs[d].find(pic);
    if (it == g_bs[d].end()) continue;
    for (uint32_t e: it->second)
    {
      const int y4 = (int) (e >> 13), x4 = (int) ((e >> 2) & 0x7FF), bs = (int) (e & 3);
      if (x4 < w4 && y4 < h4) u[(size_t) y4 * w4 + x4] |= (uint16_t) (bs << (d == 0 ? 2 : 4));
    }
    g_bs[d].erase(it);
  }
}

void RefCodeMap::dumpIfRequested(const Picture *pic, const CodingStructure &cs)
{
  const char *dir = getenv("VTM_CODEMAP_DUMP");
  if (!dir || !pic->codeMap.valid()) return;
  const std::string fn = std::string(dir) + "/poc_" + std::to_string(cs.slice->getPOC()) + ".map";
  FILE *fp = fopen(fn.c_str(), "wb");
  if (!fp) return;
  const int32_t hdr[2] = { pic->codeMap.w4, pic->codeMap.h4 };
  fwrite(hdr, sizeof(int32_t), 2, fp);
  fwrite(pic->codeMap.u.data(), sizeof(uint16_t), pic->codeMap.u.size(), fp);
  fclose(fp);
}
