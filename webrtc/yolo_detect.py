#!/usr/bin/env python3
"""
yolo_detect.py — RK3576 NPU YOLOv8 检测（ctypes librknnrt，无 rknnlite/torch 依赖）
支持：单输出（VisDrone 11 类）和多输出 DFL（COCO 80 类）两种模型
"""
import numpy as np
import cv2

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

VISDRONE_LABELS = [
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor", "others",
]


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
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


def _softmax(x, axis):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _dfl(position):
    """DFL 解码 (1,64,H,W) → (1,4,H,W)"""
    n, c, h, w = position.shape
    mc = c // 4                      # reg_max = 16
    y = position.reshape(n, 4, mc, h, w)
    y = _softmax(y, axis=2)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(axis=2)


def _box_process(position):
    """DFL box 解码 → xyxy（640 坐标）"""
    gh, gw = position.shape[2:4]
    col = np.arange(gw, dtype=np.float32).reshape(1, 1, 1, gw).repeat(gh, axis=2)
    row = np.arange(gh, dtype=np.float32).reshape(1, 1, gh, 1).repeat(gw, axis=3)
    grid = np.concatenate([col, row], axis=1)
    stride = np.array([640 // gw, 640 // gh], np.float32).reshape(1, 2, 1, 1)
    pos = _dfl(position)
    xy1 = grid + 0.5 - pos[:, 0:2]
    xy2 = grid + 0.5 + pos[:, 2:4]
    return np.concatenate([xy1 * stride, xy2 * stride], axis=1)


class YoloDetector:
    def __init__(self, model_path="/usr/local/bin/webrtc/yolov8s_airborne.rknn",
                 conf_thres=0.20, iou_thres=0.45):
        from rknn_api_ctypes import RKNN
        self.rknn = RKNN()
        self.rknn.load_rknn(model_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        # 输出数决定模型类型
        self.n_output = self.rknn.output_attr and 1 or 0
        import ctypes
        num = None
        # 简化：通过查询判断
        from rknn_api_ctypes import RknnInputOutputNum, RKNN_QUERY_IN_OUT_NUM
        num = RknnInputOutputNum()
        self.rknn.lib.rknn_query(self.rknn.ctx, RKNN_QUERY_IN_OUT_NUM,
                                 ctypes.byref(num), ctypes.sizeof(num))
        self.n_output = num.n_output
        if self.n_output > 1:
            self.labels = COCO_LABELS
            print(f"[yolo] COCO yolov8 模型 ({self.n_output} 输出, DFL)", flush=True)
        else:
            self.labels = VISDRONE_LABELS
            print(f"[yolo] VisDrone 单输出模型", flush=True)

    def detect(self, img_bgr):
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img, ratio, (dw, dh) = letterbox(img_bgr, (640, 640))
        outputs = self.rknn.inference(img)
        if self.n_output > 1:
            return self._postprocess_dfl(outputs, ratio, dw, dh)
        else:
            pred = outputs.reshape(15, -1).T
            return self._postprocess_simple(pred, ratio, dw, dh)

    # ---------- 单输出（VisDrone）----------
    def _postprocess_simple(self, pred, ratio, dw, dh):
        boxes, scores, cls_ids = [], [], []
        for i in range(pred.shape[0]):
            row = pred[i]
            score = float(row[4:].max())
            if score < self.conf_thres:
                continue
            cls_id = int(row[4:].argmax())
            cx, cy, w, h = row[:4]
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
        return [self._pack(i, boxes, scores, cls_ids) for i in keep]

    # ---------- 多输出 DFL（COCO）----------
    def _postprocess_dfl(self, outputs, ratio, dw, dh):
        branches = self.n_output // 3
        all_boxes, all_scores, all_cls = [], [], []
        for b in range(branches):
            box_out = outputs[b * 3]      # (1,64,H,W)
            cls_out = outputs[b * 3 + 1]  # (1,80,H,W)
            score_out = outputs[b * 3 + 2]  # (1,1,H,W)
            xyxy = _box_process(box_out)   # (1,4,H,W)
            n = xyxy.shape[2] * xyxy.shape[3]
            boxes = xyxy.reshape(4, n).T                 # (n,4)
            cls = cls_out.reshape(80, n).T               # (n,80)
            conf = score_out.reshape(n)                  # (n,)
            all_boxes.append(boxes)
            all_scores.append(cls)
            all_confs = conf if b == 0 else None
            # 合并置信度（用 score 输出）
            if b == 0:
                all_conf = conf
            else:
                all_conf = np.concatenate([all_conf, conf])
            all_cls.append(cls)
        boxes = np.concatenate(all_boxes)
        cls = np.concatenate(all_cls)

        class_max = cls.max(axis=1)
        class_id = cls.argmax(axis=1)
        final_conf = class_max * all_conf          # 参照官方 filter_boxes

        mask = final_conf >= self.conf_thres
        boxes, class_id, final_conf = boxes[mask], class_id[mask], final_conf[mask]

        if len(boxes) == 0:
            return []
        keep = self._nms(boxes, final_conf)
        results = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            results.append({
                "class_id": int(class_id[i]),
                "label": self.labels[int(class_id[i])] if int(class_id[i]) < len(self.labels) else "?",
                "conf": round(float(final_conf[i]), 3),
                "box": [round((x1 - dw) / ratio, 1), round((y1 - dh) / ratio, 1),
                        round((x2 - dw) / ratio, 1), round((y2 - dh) / ratio, 1)],
            })
        return results

    def _pack(self, i, boxes, scores, cls_ids):
        x1, y1, x2, y2 = boxes[i]
        return {
            "class_id": cls_ids[i],
            "label": self.labels[cls_ids[i]] if cls_ids[i] < len(self.labels) else "?",
            "conf": round(float(scores[i]), 3),
            "box": [round(float(v), 1) for v in boxes[i]],
        }

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
    t0 = time.time()
    res = det.detect(img)
    print("检测 {} 个目标, 耗时 {:.0f}ms".format(len(res), (time.time()-t0)*1000))
    for r in res[:10]:
        print("  {} conf={} box={}".format(r["label"], r["conf"], r["box"]))
