/* yvvyee: 참조 화면 부호화 구조 지도 (4x4 단위, 15비트). stage4_integration.md 3절, 비트 배치는 stage3_design.md 10.2절.
 *   [0] TU 좌측 경계  [1] TU 상단 경계  [2-3] 좌측 경계 휘도 bS  [4-5] 상단 경계 휘도 bS
 *   [6] 휘도 cbf      [7] 인트라        [8-13] CU QP(0..63)      [14] 유효
 * 디블로킹 중 bS 를 모으고(noteBS), 디블로킹이 끝나면 CU·TU 로 지도를 만들어 Picture 에 붙인다. 인코더·복호기가 같은 코드를 쓴다.
 */
#pragma once
#include <cstdint>
#include <vector>

class CodingStructure;
class Picture;

struct RefCodeMap
{
  int                   w4 = 0, h4 = 0;
  std::vector<uint16_t> u;

  bool     valid() const { return !u.empty(); }
  uint16_t unit(int x, int y) const;                     // 화소 좌표(절단) -> 단위 값
  uint16_t at(int x, int y) const;                       // 화소 좌표 -> 화소별 값 (경계 비트는 단위 첫 열·행에서만)
  void     build(const CodingStructure &cs);

  static void beginPicture(const Picture *pic);          // bS 수집 시작 (디블로킹 시작 시)
  static void noteBS(const Picture *pic, int edgeDir, int x, int y, int bs);
  static void dumpIfRequested(const Picture *pic, const CodingStructure &cs);   // VTM_CODEMAP_DUMP=dir 이면 poc_<POC>.map 기록
};
