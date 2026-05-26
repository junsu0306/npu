import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from qbcompiler.calibration import make_calib_man
from qbcompiler import mxq_compile

# 1. 캘리브레이션 전처리 (가장 안전한 float32 및 NCHW 규격)
def preprocess_normal(img_path: str):
    img = Image.open(img_path).convert('RGB')
    out = F.pil_to_tensor(img)
    out = F.resize(out, size=[256, 256])
    out = F.center_crop(out, output_size=[224, 224])
    out = out.to(torch.float, copy=False) / 255.
    out = F.normalize(out, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return out.numpy().astype(np.float32)

if __name__ == "__main__":
    work_dir = "/workspace/npu"
    calib_raw_dir = os.path.join(work_dir, "imagenet_calib")
    calib_npy_dir = os.path.join(work_dir, "calib_nchw_final")
    
    if not os.path.exists(calib_npy_dir):
        print("=== 정상 규격(NCHW) 캘리브레이션 데이터 생성 ===")
        make_calib_man(
            pre_ftn=preprocess_normal,
            data_dir=calib_raw_dir,
            save_dir=calib_npy_dir,
            save_name="imagenet_nchw",
            max_size=100
        )
    
    calib_txt_path = os.path.join(calib_npy_dir, "imagenet_nchw.txt")
    models = ["efficientvit_b0_r224.onnx", "efficientvit_b1_r224.onnx"]
    
    # 2. [핵심] 컴파일러의 착각을 원천 차단하는 강제 주입 데이터
    # 로그에서 확인한 정확한 입력 노드 이름("input.1" 또는 "input")을 사용합니다.
    # 만약 Opset 13으로 재추출하면서 이름이 "input"으로 바뀌었다면 "input"으로 적어주세요.
    input_node_name = "input.1" 
    
    feed_dict = {
        input_node_name: np.random.randn(1, 3, 224, 224).astype(np.float32)
    }
    in_dformats = {
        input_node_name: "NCHW"
    }
    
    for model_name in models:
        model_path = os.path.join(work_dir, model_name)
        if not os.path.exists(model_path):
            continue
            
        save_name = model_name.replace(".onnx", ".mxq")
        print(f"\n=== {model_name} 컴파일 시작 ===")
        
        mxq_compile(
            model=model_path,
            calib_data_path=calib_txt_path,
            backend="onnx",
            save_path=save_name,
            device="cpu",
            target_device="aries2",
            cpu_offload=True,
            
            # 직접 찾아내신 매뉴얼의 치트키 파라미터 적용
            feed_dict=feed_dict,
            in_dformats=in_dformats,
            
            # Dead Channel 에러 방지를 위한 레이어 단위 양자화
            is_quant_ch=False,
            quantize_method="percentile",
            quantize_percentile=0.999
        )
        print(f"=== {save_name} 변환 완료 ===")
