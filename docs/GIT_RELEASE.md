# 외부 과제 제출 및 Git 안내

## 포함 범위

최종 실행 코드, tests, requirements, README/docs, ONNX 12개, MXQ 12개, `final_20261015` 원본 기록을 포함한다. 모델과 원본 결과는 `assets/RELEASE_MANIFEST.json`으로 고정한다.

다음은 제외한다.

- ImageNet 이미지와 재생성 가능한 calibration NPY.
- SDK wheel/매뉴얼, 가상환경, 편집기 설정, 캐시, 실행 로그.
- 과거 방법론 문서/진단 코드 및 `past/`.
- `assets/mxq/` 임시 컴파일 결과와 신규 평가 결과 폴더.

`.gitignore`만 추가해도 이미 추적된 파일은 자동 제외되지 않는다. 이번 정리에서는 기존 추적 항목도 정리한다. Calibration 원본 이미지는 로컬에 유지하고 Git에서만 제외한다. 과거 코드·문서·SDK·식별되지 않은 calibration tensor는 `past/submission_cleanup_20261020/`에 보존한다.

## 검증과 커밋 전 확인

```bash
python3 scripts/verify_release.py --onnx --predictions --git-index
python3 -m unittest discover -s tests -v
git diff --cached --stat
git diff --cached --check
git status --short
```

모델 파일은 기존 Git 추적 방식을 유지한다. `.gitattributes`는 binary 처리와 원본 측정 파일의 줄바꿈 보존만 설정하며, **Git LFS 마이그레이션이나 과거 이력 재작성은 수행하지 않는다.** 일부 ONNX는 50 MiB보다 크고 최대 약 84.20 MiB다. 호스팅 용량 정책에 따라 향후 LFS 이관은 별도 작업으로 결정한다.

이 작업에서는 commit/push를 하지 않는다. 변경 내역을 검토하고 사용자가 직접 커밋한다.

## 외부 전달 방식

기존 Git 이력에는 과거에 추적했던 SDK/데이터가 남아 있다. 현재 파일을 제외하는 것만으로 이력이 지워지지는 않는다. **저장소 전체를 복사하거나 `past/`까지 압축해 제출하지 않는다.**

사용자가 최종 커밋한 후 현재 커밋 파일만 내보내는 예:

```bash
git archive --format=zip --output=/tmp/npu_final_submission.zip HEAD
```

이 명령은 **커밋 전 staging 변경을 포함하지 않는다.** 제출 커밋 후 실행한다. Git 이력이 포함되지 않은 스냅샷이며 모델과 원본 결과는 모두 들어간다. 새 저장소로 공개해야 할 경우 SDK·데이터·모델 배포 권한을 확인한 뒤 별도 clean-history 전환을 결정한다. 기존 원격에 force-push하거나 이력을 자동 삭제하지 않는다.

## 데이터·SDK·모델 권리

ImageNet 데이터와 Mobilint SDK는 별도 사용/배포 조건을 따른다. 본 프로젝트가 이들 자료에 새 라이선스를 부여하지 않는다. 모델 가중치와 class label 파일도 원본 배포 조건을 확인한다. 과제 제출은 자동으로 무제한 공개 배포 허가를 의미하지 않는다.

## 결과 표현

측정 `PASS`는 실행 성공이며 정확도 보장을 뜻하지 않는다. 원본 Top-1/Top-5, 모델 해시, NPU 메모리 지표의 정의와 Patch Slimming 정확도 손실을 함께 전달한다. 모델/측정 원본을 새 코드로 실행한 것처럼 고쳐 쓰지 않는다.
