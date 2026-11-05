/* yvvyee: QKV-MC 가설별 휘도 분수 보간(L1~L8, L9 없음)의 고정소수 추론 — stage4_integration.md 4.3절·10.3절.
 * 내보내기 디렉터리(model/qkv_export.py)의 양자화 파라미터·조회표를 적재하고, 4x4 블록 하나를 20x20 정수 창과 참조 지도 창으로 예측한다.
 * yvvyee: 타일 의미(stage4 11절)로 일반화 — tw x th 타일 (4의 배수, 16 이하)을 (th+16)x(tw+16) 창으로 예측한다. tw=th=4, stride 20 은 기존 블록 의미와 비트 동일.
 * 산술: 화소 Q10.4 (정규화 1.0 = 1024 화소), FIR Q1.14, floor(x+0.5) 반올림, int64/__int128 누산. Python 모사기(qkv_fixed.py)와 비트 일치.
 * 블록 단위 구현(창 중첩 25배)이라 느리다. PU 단위 재구성(10.4절)의 비트 일치 기준으로도 쓴다.
 */
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

class NNInterp
{
public:
  static NNInterp &get();
  bool load(const std::string &dir);                 // manifest.txt 적재. 실패 시 false
  bool enabled() const { return m_loaded; }

    static constexpr int TILE = 16;   // yvvyee: PU 타일 크기 (stage4 11절)
  struct BlockIn
  {
    int32_t win[1024];       // (th+16)x(tw+16) 참조 정수 화소, 행 간격 stride (좌상단 = 정수 MV 위치 - 8), 화면 밖은 가장자리 복제
    int32_t map[1024];       // 같은 배치의 참조 부호화 구조 지도 (RefCodeMap::at)
    int     stride = 20;     // yvvyee: win/map 행 간격 (>= tw+16, stride*(th+15)+tw+16 <= 1024)
    int     tw = 4, th = 4;  // yvvyee: 타일 크기 (4의 배수, 4..16)
    int     fx, fy;          // MV 분수부 (1/16)
    int     qp;              // 슬라이스 QP
    int     imv;             // AMVR 등급 (0..3)
    int     merge;           // 병합 플래그
    int     dpoc;            // 참조 POC - 현재 POC
    int     w4, h4;          // 블록 크기 / 4
    int     st;              // 0 단방향, 1 양방향 BDOF 미적용, 2 양방향 BDOF 적용
  };
  struct BlockOut
  {
    int64_t y[256];          // yvvyee: th x tw 예측 (행 간격 tw), Q10.4 (화소 x16)
    int64_t ylin[400];       // yvvyee: (th+4)x(tw+4) 선형 경로 (타일 -2..t+1, 행 간격 tw+4), Q10.4
    int64_t xd[1024];        // yvvyee: L4b 뒤 창 (th+16)x(tw+16) (행 간격 tw+16), Q10.4 (시험용)
  };
  void predict(const BlockIn &in, BlockOut &out) const;      // yvvyee: 가속 구현. predictRef 와 모든 출력 비트가 같다 (적재 시 증명 조건을 못 맞추면 predictRef 로 넘김)
  // yvvyee: linearOnly = proxy evaluation for encoder screening (VTM_NNINTERP_PROXY). Runs L1-L6 only, skips L7/L8, and returns
  //         out.y = the tile area of y_lin, bit-exact to the y_lin the full path computes (out.y of the full path = y_lin + L8 correction).
  void predict(const BlockIn &in, BlockOut &out, bool linearOnly) const;
  void predictRef(const BlockIn &in, BlockOut &out) const;   // yvvyee: 가속 전 기준 구현 (비트 일치 검증용, 2026-10-21 이전 predict 그대로)
  struct Arr { std::vector<int64_t> v; std::vector<int64_t> shape; };
  const Arr &A_(const std::string &n) const { return A(n); }
  int64_t S_(const std::string &n, int i = 0) const { return S(n, i); }

private:
  NNInterp() {}
  std::map<std::string, Arr> m_t;
  bool m_loaded = false;
  struct Fast; std::shared_ptr<Fast> m_fast;   // yvvyee: 적재 시 만든 가속용 상수·가중치 배치
  void buildFast();
  const Arr &A(const std::string &n) const;
  int64_t S(const std::string &n, int i = 0) const { return A(n).v[i]; }
};
