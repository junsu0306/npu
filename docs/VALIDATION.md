# 제출 패키지 검증

검증일: 2026-10-20 UTC. 최종 측정 기준은 `final_20261015`이며, 이 검증은 **새 NPU 성능 실험이 아니라 제출물 무결성·코드 정리 검증**이다.

## 수행한 검사

| 검사 | 결과 |
|---|---|
| 정리 전 `3e8bb5a…`의 모델/원본 결과와 비교 | 68개 파일 SHA-256 동일, 변경 0개 |
| 최종 모델 목록 | ONNX 12개 + MXQ 12개 |
| 측정 JSON에 기록한 모델 해시 | ONNX/MXQ 24개 모두 일치 |
| 전체 평가 기록 | 12개 × 50,000장, 오류 0, telemetry 12/12 정상 |
| Top-1 오프라인 재검산 | 저장된 60만 예측/정답으로 12개 모두 일치 |
| Top-5 검증 범위 | JSON/CSV 집계값 일치; Top-5 예측 전체는 저장되지 않아 독립 재계산하지 않음 |
| 독립 메모리 결과 | 통합 평가의 NPU peak와 12/12 일치, random 입력 해시 일치 |
| ONNX 검사 | checker 통과, FP32 NHWC batch 1, 출력 1×1000, opset 17 |
| 선택 그래프 검사 | Gather/TopK 계열 및 identity selection MatMul 없음 |
| 회귀 테스트 | 18개 통과 |
| 문서 내부 링크/파이썬 문법 | 통과 |
| Git 포함 대상 | 최종 자산/기록 유지, 로컬 데이터·SDK·past 제외 |

검증 명령:

```bash
python3 scripts/verify_release.py --onnx --predictions --git-index
python3 -m unittest discover -s tests -v
python3 -m compileall -q compile benchmark scripts tests
git diff --cached --check
```

모델 파일 합계는 약 **618.20 MiB**, 원본 결과 파일은 약 **4.60 MiB**다. `assets/RELEASE_MANIFEST.json`에는 이 68개 파일의 상대경로/바이트 크기/해시를 기록했다. 문서와 코드 변경은 manifest의 모델·원본 결과와 구분한다.

## 정리 과정의 실행 안전장치

- Label 파일의 새 위치를 runtime/ONNX reference에 반영했다.
- SDK 없는 서버에서도 benchmark `--help`/`--dry-run`과 파일 검증을 사용할 수 있게 했다.
- Calibration은 새 폴더에 생성하고 이미지 부족/잘못된 shape·dtype·정규화/메타데이터 부재를 검출한다.
- Compiler의 실제 인자는 `vision_transformer`, `global8`, `cpu_offload=True`를 유지하고, 비활성 수동 양자화 예제를 제거했다.
- Calibration/runtime/전체 평가 전처리가 synthetic 이미지의 서로 다른 종횡비에서 동일함을 검사했다.
- 보고서 생성기는 현재 입력 결과로 판단한다. 과거 실험 수치와 독립 측정 성공을 고정 출력하지 않는다.

현재 Python 환경에는 `qbruntime`이 없어 호스트 측 검증만 수행했다. **새 MXQ 컴파일이나 NPU 재평가는 수행하지 않았다.** SDK 호출은 테스트 대역으로 인자만 검증했고, 실제 SDK 호환성/성능은 별도 장치에서 [RUNBOOK](RUNBOOK.md) 절차로 확인한다. 배포된 MXQ의 성능 근거는 보존한 2026-10-15 실제 측정 기록이다.
