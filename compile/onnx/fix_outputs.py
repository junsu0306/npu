#!/usr/bin/env python3
"""
ONNX 모델의 출력을 단일 분류 출력('output')으로 정리.

token pruning 모델은 export 시 TopK 중간 출력이 graph output으로 노출되는데,
qbcompiler quantizer가 출력 1개만 지원하므로 컴파일 전에 제거해야 함.

사용:
  python3 compile/onnx/fix_outputs.py assets/onnx/vit_tiny_c30_token70_npusafe.onnx
  python3 compile/onnx/fix_outputs.py assets/onnx/  # 디렉토리 일괄 처리
"""

import argparse
import glob
import os
import onnx


def fix_outputs(src: str, dst: str | None = None, target_output: str = "output") -> str:
    model = onnx.load(src)

    current = [o.name for o in model.graph.output]
    if len(current) == 1 and current[0] == target_output:
        print(f"[SKIP] 이미 단일 출력: {os.path.basename(src)}")
        return src

    keep = [o for o in model.graph.output if o.name == target_output]
    if not keep:
        names = ", ".join(current)
        raise ValueError(f"'{target_output}' 출력을 찾을 수 없음. 현재 출력: {names}")

    del model.graph.output[:]
    model.graph.output.extend(keep)
    onnx.checker.check_model(model)

    if dst is None:
        base, ext = os.path.splitext(src)
        dst = f"{base}_fixout{ext}"

    onnx.save(model, dst)
    removed = [n for n in current if n != target_output]
    print(f"[FIXED] {os.path.basename(src)}")
    print(f"  제거된 출력 {len(removed)}개: {removed}")
    print(f"  저장: {dst}")
    return dst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help=".onnx 파일 또는 디렉토리")
    parser.add_argument("--output-name", default="output", help="유지할 출력 텐서 이름 (기본: 'output')")
    parser.add_argument("--inplace", action="store_true", help="원본 파일 덮어쓰기")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*.onnx")))
    else:
        files = [args.path]

    for f in files:
        dst = f if args.inplace else None
        try:
            fix_outputs(f, dst=dst, target_output=args.output_name)
        except Exception as e:
            print(f"[ERROR] {os.path.basename(f)}: {e}")


if __name__ == "__main__":
    main()
