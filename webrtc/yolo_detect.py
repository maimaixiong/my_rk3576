#!/usr/bin/env python3
"""
yolo_detect.py — RK3576 NPU YOLOv8 检测（ctypes librknnrt，无 rknnlite 依赖）
VisDrone 11 类（pedestrian/people/car/bus 等），640x640 输入
"""
import numpy as np
import cv2

VISDRONE_LABELS = [
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor", "others",
]


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """YOLO 标准 letterbox，返回 (缩放宽高, pad)"""
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


class YoloDetector:
    def __init__(self, model_path="/usr/local/bin/webrtc/yolov8s_airborne.rknn",
                 labels=None, conf_thres=0.25, iou_thres=0.45):
        from rknn_api_ctypes import RKNN
        self.rknn = RKNN()
        self.rknn.load_rknn(model_path)
        self.labels = labels or VISDRONE_LABELS
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def detect(self, img_bgr):
        """
        输入任意尺寸 BGR 图，返回 [{label, conf, box:[x1,y1,x2,y2]}]
        box 坐标相对原图
        """
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # YOLO 训练用 RGB
        img, ratio, (dw, dh) = letterbox(img_bgr, (640, 640))
        # 推理：输入 NHWC uint8（自动转 FP16）
        out = self.rknn.inference(img)
        pred = out.reshape(15, -1).T   # (8400, 15): xywh + 11 classes
        return self._postprocess(pred, ratio, dw, dh)

    def _postprocess(self, pred, ratio, dw, dh):
        boxes, scores, cls_ids = [], [], []
        for i in range(pred.shape[0]):
            row = pred[i]
            score = float(row[4:].max())
            if score < self.conf_thres:
                continue
            cls_id = int(row[4:].argmax())
            cx, cy, w, h = row[:4]
            # 640 坐标系 → xyxy
            x1 = (cx - w / 2 - dw) / ratio
            y1 = (cy - h / 2 - dh) / ratio
            x2 = (cx + w / 2 - dw) / ratio
            y2 = (cy + h / 2 - dh) / ratio
            boxes.append([max(0, x1), max(0, y1), x2, y2])
            scores.append(score)
            cls_ids.append(cls_id)
        if not boxes:
            return []
        boxes = np.array(boxes)
        scores = np.array(scores)
        keep = self._nms(boxes, scores)
        results = []
        for i in keep:
            results.append({
                "class_id": cls_ids[i],
                "label": self.labels[cls_ids[i]] if cls_ids[i] < len(self.labels) else "?",
                "conf": round(float(scores[i]), 3),
                "box": [round(float(v), 1) for v in boxes[i]],
            })
        return results

    def _nms(self, boxes, scores):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= self.iou_thres]
        return keep


if __name__ == "__main__":
    import sys, time
    det = YoloDetector()
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "/tmp/yolo/bus.jpg")
    if img is None:
        print("图片加载失败"); sys.exit(1)
    t0 = time.time()
    results = det.detect(img)
    print(f"检测 {len(results)} 个目标, 推理耗时 {(time.time()-t0)*1000:.0f}ms")
    for r in results[:10]:
        print(f"  {r['label']} conf={r['conf']} box={r['box']}")
