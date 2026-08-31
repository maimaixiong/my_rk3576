#!/usr/bin/env python3
"""
rknn_api_ctypes.py — ctypes 调用板载 librknnrt（无需 rknnlite/torch）
支持 rknn_init / query / inputs_set / run / outputs_get
"""
import ctypes
import ctypes.util
import numpy as np

# RKNN API 常量（rknn_api.h）
RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256
RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_FLOAT16 = 1
RKNN_TENSOR_INT8 = 2
RKNN_TENSOR_UINT8 = 3
RKNN_TENSOR_INT16 = 4
RKNN_TENSOR_INT32 = 5
RKNN_TENSOR_INT64 = 6
RKNN_TENSOR_BOOL = 7
RKNN_TENSOR_NHWC = 1


class RKNNContext(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p)]


class RknnTensorAttr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * RKNN_MAX_DIMS),
        ("name", ctypes.c_char * RKNN_MAX_NAME_LEN),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("qnt_type", ctypes.c_uint32),
        ("fl", ctypes.c_int32),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]


class RknnInput(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_uint32),
        ("fmt", ctypes.c_uint32),
    ]


class RknnOutput(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]


class RknnInputOutputNum(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class RKNN:
    """librknnrt 的轻量 ctypes 封装（只支持单输入/输出推理）"""

    def __init__(self, lib_path="/usr/lib/librknnrt.so"):
        self.lib = ctypes.CDLL(lib_path)
        self.ctx = RKNNContext()
        self.input_attr = None
        self.output_attr = None

    def load_rknn(self, model_path):
        self.lib.rknn_init.restype = ctypes.c_int
        ret = self.lib.rknn_init(
            ctypes.byref(self.ctx), model_path.encode(), 0, 0, None)
        if ret != 0:
            raise RuntimeError(f"rknn_init 失败: {ret}")
        # 查询输入输出
        num = RknnInputOutputNum()
        if self.lib.rknn_query(self.ctx, RKNN_QUERY_IN_OUT_NUM,
                               ctypes.byref(num), ctypes.sizeof(num)) != 0:
            raise RuntimeError("查询 IO 数量失败")
        print(f"[rknn] 输入 {num.n_input} 输出 {num.n_output}")

        self.input_attr = RknnTensorAttr()
        self.input_attr.index = 0
        self.lib.rknn_query(self.ctx, RKNN_QUERY_INPUT_ATTR,
                            ctypes.byref(self.input_attr), ctypes.sizeof(self.input_attr))
        self.output_attr = RknnTensorAttr()
        self.output_attr.index = 0
        self.lib.rknn_query(self.ctx, RKNN_QUERY_OUTPUT_ATTR,
                            ctypes.byref(self.output_attr), ctypes.sizeof(self.output_attr))
        in_dims = [self.input_attr.dims[i] for i in range(self.input_attr.n_dims)]
        out_dims = [self.output_attr.dims[i] for i in range(self.output_attr.n_dims)]
        print(f"[rknn] 输入 {in_dims} ({self.input_attr.size}B, fmt={self.input_attr.fmt})")
        print(f"[rknn] 输出 {out_dims} ({self.output_attr.size}B)")

    def inference(self, input_data):
        """
        input_data: numpy uint8 array (与模型输入一致)
        返回: 单输出时 numpy float32 数组；多输出时 list[numpy float32]
        """
        if input_data.dtype != np.uint8:
            input_data = input_data.astype(np.uint8)
        in_buf = input_data.ctypes.data_as(ctypes.c_void_p)

        inp = RknnInput()
        inp.index = 0
        inp.buf = in_buf
        inp.size = input_data.nbytes
        inp.pass_through = 0
        inp.type = RKNN_TENSOR_UINT8   # 传 uint8，rknn 自动转换
        inp.fmt = RKNN_TENSOR_NHWC
        if self.lib.rknn_inputs_set(self.ctx, 1, ctypes.byref(inp)) != 0:
            raise RuntimeError("rknn_inputs_set 失败")

        if self.lib.rknn_run(self.ctx, None) != 0:
            raise RuntimeError("rknn_run 失败")

        # 查询输出数量
        num = RknnInputOutputNum()
        self.lib.rknn_query(self.ctx, RKNN_QUERY_IN_OUT_NUM,
                            ctypes.byref(num), ctypes.sizeof(num))
        n_out = num.n_output

        # 查询各输出属性（shape/size）
        attrs = []
        for i in range(n_out):
            attr = RknnTensorAttr()
            attr.index = i
            self.lib.rknn_query(self.ctx, RKNN_QUERY_OUTPUT_ATTR,
                                ctypes.byref(attr), ctypes.sizeof(attr))
            attrs.append(attr)

        # 一次获取所有输出（数组方式）
        outs = (RknnOutput * n_out)()
        for i in range(n_out):
            outs[i].want_float = 1
            outs[i].is_prealloc = 0
            outs[i].index = i
            outs[i].size = attrs[i].size
        if self.lib.rknn_outputs_get(self.ctx, n_out, outs, None) != 0:
            raise RuntimeError("rknn_outputs_get 失败")

        results = []
        for i in range(n_out):
            addr = outs[i].buf if isinstance(outs[i].buf, int) else 0
            if not addr:
                raise RuntimeError(f"输出 {i} 缓冲区无效")
            arr = np.ctypeslib.as_array(
                (ctypes.c_float * attrs[i].n_elems).from_address(addr)).copy()
            dims = [attrs[i].dims[j] for j in range(attrs[i].n_dims)]
            results.append(arr.reshape(dims))

        self.lib.rknn_outputs_release(self.ctx, n_out, outs)
        return results if n_out > 1 else results[0]

    def __del__(self):
        try:
            if self.ctx.ptr:
                self.lib.rknn_destroy(self.ctx)
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    rknn = RKNN()
    rknn.load_rknn(sys.argv[1] if len(sys.argv) > 1 else "/tmp/yolo/yolov8s_airborne.rknn")
    # 测试：640x640 灰色图
    img = np.zeros((640, 640, 3), np.uint8)
    out = rknn.inference(img)
    print(f"输出 shape 推断: {rknn.output_attr.n_elems} 元素")
    print(f"输出前 20 值: {out[:20]}")
