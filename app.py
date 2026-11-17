"""YOLOv11-Pose 坐姿检测器。

用 YOLOv11n-pose 提取 17 个 COCO 关键点, 根据躯干角度、头部相对位置、双肩高度差,
判断当前坐姿属于: 正常 / 弯腰驼背 / 低头趴桌 / 身体侧歪。

界面用 Gradio: 支持摄像头实时检测与上传图片检测。

运行:
    pip install -r requirements.txt
    python app.py                 # 默认 http://127.0.0.1:7860
    python app.py --share         # 生成公网临时链接
"""

from __future__ import annotations

import argparse
import math

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 可调参数: 所有阈值集中在这里, 换设备/换分辨率时只需要改这一块
# ---------------------------------------------------------------------------
MODEL_NAME = "yolo11n-pose.pt"   # 首次运行自动下载(~6MB)
PERSON_CONF = 0.5                # 人体检测置信度
KEYPOINT_CONF = 0.4              # 关键点可用性阈值(低于则跳过该人)

TORSO_ANGLE_MAX = 25.0           # 躯干与竖直方向夹角上限(度)
HEAD_DROP_MAX = 30.0             # 鼻尖相对肩中点的最大下移距离(px)
SHOULDER_DIFF_MAX = 35.0         # 双肩最大高度差(px)

# COCO-17 关键点下标
NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 0, 5, 6, 11, 12

GREEN = (0, 200, 0)
RED = (0, 0, 220)
BLUE = (220, 120, 0)

# 画面上的英文标签 + 返回给界面的中文说明(OpenCV 的 putText 不支持中文)
LABELS = {
    "normal": ("Normal", "坐姿正常"),
    "hunchback": ("Hunchback", "弯腰驼背"),
    "head_down": ("Head down", "低头趴桌"),
    "skew": ("Skew", "身体侧歪"),
    "no_person": ("No person", "未检测到人体"),
}

model = YOLO(MODEL_NAME)


# ---------------------------------------------------------------------------
# 几何与判定(纯函数, 方便单独测试)
# ---------------------------------------------------------------------------
def calculate_torso_angle(shoulder_center: np.ndarray, hip_center: np.ndarray) -> float:
    """计算"肩中点 -> 髋中点"连线与竖直方向的夹角(0~180 度)。

    竖直方向取 (0, 1)(图像坐标系 y 轴向下), 用向量点积求夹角;
    除以模长前先判断退化情况(两点重合时无方向, 返回 0)。
    """
    dx = float(shoulder_center[0] - hip_center[0])
    dy = float(shoulder_center[1] - hip_center[1])
    vert_dx, vert_dy = 0.0, 1.0
    mag_torso = math.hypot(dx, dy)
    if mag_torso == 0:
        return 0.0
    cos_value = (dx * vert_dx + dy * vert_dy) / mag_torso
    return math.degrees(math.acos(float(np.clip(cos_value, -1.0, 1.0))))


def classify_posture(keypoints: np.ndarray) -> tuple[str, float]:
    """根据单个人的关键点判定坐姿。

    返回 (类别 key, 躯干角度)。类别 key 见 LABELS。
    判定顺序: 驼背 -> 低头 -> 侧歪 -> 正常; 先命中的优先,
    避免同一帧出现互相矛盾的结论。
    """
    nose = keypoints[NOSE]
    left_shoulder = keypoints[L_SHOULDER]
    right_shoulder = keypoints[R_SHOULDER]
    left_hip = keypoints[L_HIP]
    right_hip = keypoints[R_HIP]

    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    hip_center = (left_hip + right_hip) / 2.0

    torso_angle = calculate_torso_angle(shoulder_center, hip_center)
    shoulder_diff = abs(float(left_shoulder[1] - right_shoulder[1]))
    head_drop = float(nose[1] - shoulder_center[1])

    if torso_angle > TORSO_ANGLE_MAX:
        return "hunchback", torso_angle
    if head_drop > HEAD_DROP_MAX:
        return "head_down", torso_angle
    if shoulder_diff > SHOULDER_DIFF_MAX:
        return "skew", torso_angle
    return "normal", torso_angle


# ---------------------------------------------------------------------------
# Gradio 回调
# ---------------------------------------------------------------------------
def posture_detect(input_img: np.ndarray | None):
    """检测一张图片/一帧画面, 返回标注后的 RGB 图像与中文判定文本。"""
    if input_img is None:
        return None, "请先上传图片或打开摄像头"

    img = input_img.copy()
    results = model(img, conf=PERSON_CONF, verbose=False)

    status_texts: list[str] = []
    for result in results:
        if result.keypoints is None:
            continue

        keypoints_xy = result.keypoints.xy.cpu().numpy()
        keypoints_conf = result.keypoints.conf.cpu().numpy()

        for person_kpt, person_conf in zip(keypoints_xy, keypoints_conf):
            # 关键点不可信(遮挡/背影)时跳过, 否则会给出错误结论
            required = (NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
            if any(person_conf[index] < KEYPOINT_CONF for index in required):
                continue

            label_key, torso_angle = classify_posture(person_kpt)
            english, chinese = LABELS[label_key]
            color = GREEN if label_key == "normal" else RED

            shoulder_center = ((person_kpt[L_SHOULDER] + person_kpt[R_SHOULDER]) / 2.0)
            hip_center = ((person_kpt[L_HIP] + person_kpt[R_HIP]) / 2.0)

            # 画面上画英文(OpenCV 不支持中文), 中文由 textbox 返回
            cv2.putText(
                img,
                f"{english} ({torso_angle:.0f} deg)",
                (int(shoulder_center[0]), int(shoulder_center[1]) - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.line(
                img,
                (int(shoulder_center[0]), int(shoulder_center[1])),
                (int(hip_center[0]), int(hip_center[1])),
                BLUE,
                2,
            )
            status_texts.append(chinese)

    # OpenCV 用 BGR, Gradio 期望 RGB —— 不转换会导致红蓝互换
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    info = " | ".join(status_texts) if status_texts else LABELS["no_person"][1]
    return img_rgb, info


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="YOLOv11 坐姿检测器") as demo:
        gr.Markdown("# 🪑 YOLOv11-Pose 坐姿检测系统")
        gr.Markdown(
            "支持摄像头实时检测 / 上传图片检测。"
            "识别四类状态：正常、弯腰驼背、低头趴桌、身体侧歪。"
        )
        with gr.Row():
            input_image = gr.Image(
                type="numpy", label="输入画面", sources=["webcam", "upload"]
            )
            output_image = gr.Image(label="检测结果")
        output_text = gr.Textbox(label="检测信息")
        detect_button = gr.Button("开始检测", variant="primary")
        detect_button.click(posture_detect, inputs=input_image, outputs=[output_image, output_text])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv11-Pose 坐姿检测器")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--share", action="store_true", help="生成 Gradio 公网临时链接")
    args = parser.parse_args()

    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
