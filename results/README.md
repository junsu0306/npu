# 결과 보존 기준

제출 대상은 **[final_benchmark/final_20261015](final_benchmark/final_20261015/summary.md)** 하나다.

- `summary.md`, `summary.csv`: 최종 표 및 평탄화 지표.
- `benchmark.json`, `run_metadata.json`, `models/*.json`: 전체/모델별 원본, 실행 환경과 명령.
- `samples.tsv`, `predictions/*.npy`: 50,000장 순서/정답 및 모델별 Top-1 예측.
- `isolated_memory/`: 독립 프로세스 메모리 결과와 고정 random 입력.

원본 측정 파일은 절대경로/실행 당시 Git 상태를 포함해 그대로 보존한다. 새 smoke/full/memory 실행은 별도 폴더를 사용하며 기본적으로 Git에서 제외한다. 파일 동일성 검증은 `scripts/verify_release.py`, 해석은 [docs/RESULTS.md](../docs/RESULTS.md)를 참조한다.
