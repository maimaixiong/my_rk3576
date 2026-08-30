#!/usr/bin/env python3
"""
v4l2_cap.py — RK3576 ISP mainpath 高效取帧（ctypes V4L2 MPLANE）
rknn_cam.c 已验证直读 V4L2 达 30fps（gst v4l2src 只有 ~10fps）
"""
import ctypes
import ctypes.util
import mmap
import os

# ---------- V4L2 ioctl 常量（板上内核实测） ----------
VIDIOC_S_FMT      = 0xC0D05605   # struct v4l2_format (208)
VIDIOC_REQBUFS    = 0xC0145608   # struct v4l2_requestbuffers (20)
VIDIOC_QUERYBUF   = 0xC0585609   # struct v4l2_buffer (88)
VIDIOC_QBUF       = 0xC058560F
VIDIOC_DQBUF      = 0xC0585611
VIDIOC_STREAMON   = 0x40045612
VIDIOC_STREAMOFF  = 0x40045613

V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE = 9
V4L2_MEMORY_MMAP   = 1
V4L2_PIX_FMT_NV12  = 0x3231564E  # 'NV12'
V4L2_FIELD_NONE    = 0
VIDEO_MAX_PLANES   = 8


# ---------- ctypes 结构体（ctypes 自然对齐 = 内核 64 位布局） ----------
class V4L2PlanePixFormat(ctypes.Structure):
    _fields_ = [
        ("sizeimage", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32),
        ("reserved", ctypes.c_uint16 * 6),
    ]   # 20 bytes

class V4L2PixFormatMplane(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32),
        ("plane_fmt", V4L2PlanePixFormat * VIDEO_MAX_PLANES),
        ("num_planes", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 11),
    ]   # 192 bytes（内核实测）

class V4L2Format(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("pix_mp", V4L2PixFormatMplane),
        ("_pad", ctypes.c_uint8 * 12),
    ]   # 208 bytes（内核实测）

class V4L2RequestBuffers(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 2),
    ]   # 20 bytes

class V4L2PlaneM(ctypes.Union):
    _fields_ = [
        ("mem_offset", ctypes.c_uint32),
        ("userptr", ctypes.c_uint64),
        ("fd", ctypes.c_int32),
    ]

class V4L2Plane(ctypes.Structure):
    _fields_ = [
        ("bytesused", ctypes.c_uint32),
        ("length", ctypes.c_uint32),
        ("m", V4L2PlaneM),
        ("data_offset", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 11),
    ]   # 64 bytes

class V4L2BufferM(ctypes.Union):
    _fields_ = [
        ("offset", ctypes.c_uint32),
        ("userptr", ctypes.c_uint64),
        ("planes", ctypes.c_void_p),
    ]

class V4L2Buffer(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("timestamp", ctypes.c_uint64 * 2),
        ("timecode", ctypes.c_uint32 * 4),
        ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("m", V4L2BufferM),
        ("length", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]   # 88 bytes（内核实测，ctypes 自动对齐）


class V4L2Capture:
    """RK ISP mainpath 取帧器（MPLANE mmap）"""

    def __init__(self, dev="/dev/video11", width=2688, height=1520, nbufs=4):
        self.dev = dev
        self.width = width
        self.height = height
        self.nbufs = nbufs
        self.frame_size = width * height * 3 // 2  # NV12
        self._libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        self._fd = None
        self._maps = []

    def _ioctl(self, req, arg):
        r = self._libc.ioctl(self._fd, req, ctypes.byref(arg))
        if r < 0:
            raise OSError(ctypes.get_errno(), f"ioctl {hex(req)} 失败")

    def open(self):
        assert ctypes.sizeof(V4L2Format) == 208, f"V4L2Format {ctypes.sizeof(V4L2Format)}"
        assert ctypes.sizeof(V4L2Buffer) == 88, f"V4L2Buffer {ctypes.sizeof(V4L2Buffer)}"
        self._fd = os.open(self.dev, os.O_RDWR)
        if self._fd < 0:
            raise OSError(f"无法打开 {self.dev}")

        # S_FMT
        fmt = V4L2Format()
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
        fmt.pix_mp.width = self.width
        fmt.pix_mp.height = self.height
        fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12
        fmt.pix_mp.field = V4L2_FIELD_NONE
        fmt.pix_mp.num_planes = 1
        fmt.pix_mp.plane_fmt[0].sizeimage = self.frame_size
        r = self._libc.ioctl(self._fd, VIDIOC_S_FMT, ctypes.byref(fmt))
        if r == 0:
            self.width = fmt.pix_mp.width
            self.height = fmt.pix_mp.height
            self.frame_size = fmt.pix_mp.plane_fmt[0].sizeimage
        else:
            # RK ISP 可跳过 S_FMT（用默认格式 2688x1520 NV12）
            print(f"[v4l2] S_FMT 跳过（默认格式），errno={ctypes.get_errno()}")
        print(f"[v4l2] {self.dev} {self.width}x{self.height} NV12")

        # REQBUFS
        req = V4L2RequestBuffers()
        req.count = self.nbufs
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
        req.memory = V4L2_MEMORY_MMAP
        self._ioctl(VIDIOC_REQBUFS, req)

        # QUERYBUF + mmap
        for i in range(self.nbufs):
            buf = V4L2Buffer()
            buf.index = i
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
            buf.memory = V4L2_MEMORY_MMAP
            buf.length = 1
            plane = V4L2Plane()
            buf.m.planes = ctypes.cast(ctypes.pointer(plane), ctypes.c_void_p)
            self._ioctl(VIDIOC_QUERYBUF, buf)
            mm = mmap.mmap(self._fd, plane.length, mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE,
                           offset=plane.m.mem_offset)
            self._maps.append((mm, plane))
            self._ioctl(VIDIOC_QBUF, buf)

        # STREAMON
        on = ctypes.c_uint32(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)
        if self._libc.ioctl(self._fd, VIDIOC_STREAMON, ctypes.byref(on)) < 0:
            raise OSError(ctypes.get_errno(), "STREAMON 失败")
        print("[v4l2] 流已启动")

    def grab(self):
        """阻塞取一帧，返回 bytes(NV12)"""
        buf = V4L2Buffer()
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
        buf.memory = V4L2_MEMORY_MMAP
        buf.length = 1
        plane = V4L2Plane()
        buf.m.planes = ctypes.cast(ctypes.pointer(plane), ctypes.c_void_p)
        if self._libc.ioctl(self._fd, VIDIOC_DQBUF, ctypes.byref(buf)) < 0:
            raise OSError(ctypes.get_errno(), "DQBUF 失败")
        mm, _ = self._maps[buf.index]
        data = mm[:plane.bytesused]
        if self._libc.ioctl(self._fd, VIDIOC_QBUF, ctypes.byref(buf)) < 0:
            raise OSError(ctypes.get_errno(), "QBUF 失败")
        return bytes(data)

    def close(self):
        if self._fd is not None:
            off = ctypes.c_uint32(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)
            self._libc.ioctl(self._fd, VIDIOC_STREAMOFF, ctypes.byref(off))
            for mm, _ in self._maps:
                mm.close()
            os.close(self._fd)
            self._fd = None
