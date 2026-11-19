# 최종 NPU 벤치마크

최종 실측은 **`results/final_benchmark/final_20261015`**이며 대응 모델은 **`assets/mxq_final_20261006`의 12개**다. [결과 원본](../results/final_benchmark/final_20261015/summary.md)과 [해석](../docs/RESULTS.md)을 기준으로 한다.

| 진입점 | 용도 |
|---|---|
| `final/eval_final_models.py` | 50,000장 ImageNet Top-1/Top-5, inference/E2E latency, host/NPU 메모리, 전력, 정적 ONNX 분석 |
| `memory/profile_memory.py` | 고정 random 입력, 모델별 새 프로세스, idle/active NPU 메모리 재검증 |
| `runtime/run_npu.py` | 사용자가 준비한 단일 이미지로 NPU 실행 확인 |
| `onnx/infer_onnx.py` | 동일 프로필을 사용하는 ONNX CPU reference |

실제 명령은 [RUNBOOK](../docs/RUNBOOK.md)의 4~8절에 있다. 새 측정은 새 출력 폴더에 기록한다. 최종 저장된 원본 결과를 덮어쓰지 않는다.

## 측정 계약

- ImageNet validation 1,000 class × 50장, seed 42, 모델별 같은 이미지/정답/순서.
- RGB, OpenCV bicubic 단변 249, center crop 224, mean/std 모두 0.5.
- HWC float32, batch 1, `global8`, device 0, warmup 10회.
- 전체 평가: host sample 20ms, NPU sample 500ms.
- 독립 메모리: random seed 2022, warmup 10회, latency 100회, 별도 telemetry 반복 10초.

`model.infer()`는 이미지 전처리를 제외한 지연시간이며 E2E FPS는 전처리까지 포함한다. NPU memory는 장치 전체에서 관측한 peak MB다. Host RSS 등은 프로세스 MiB이며 레이어 activation 또는 CUDA allocator와 동일하지 않다.

평가 `PASS`는 실행 완료이지 정확도 보존을 뜻하지 않는다. Patch Slimming 정확도 손실까지 포함한 최종 수치를 그대로 보고한다.
