/*
 * rknn_cam.c — RK3576 NPU 摄像头实时推理
 * V4L2(ISP /dev/video11 NV12) → 缩放RGB224 → RKNN(mobilenet) → Top5 + FPS
 *
 * 编译: gcc -O2 -o rknn_cam rknn_cam.c -lrknnrt -I/usr/include -pthread
 * 运行: sudo ./rknn_cam [模型.rknn] [标签.txt] [运行秒数]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <linux/videodev2.h>
#include "rknn_api.h"

/* ---------- 软件 AE（曝光闭环） ---------- */
#define SNS_SUBDEV  "/dev/v4l-subdev3"
#define EXP_MIN     2
#define EXP_MAX     1566
#define GAIN_MIN    128
#define GAIN_MAX    1984
#define AE_TARGET   50.0f       /* 目标平均亮度 */
#define AE_EVERY    5           /* 每 N 帧调整一次 */
#define AE_EXP_STEP 1.30f
#define AE_GAIN_STEP 1.20f
#define AE_DEADBAND 3.0f

#define VIDEO_DEV  "/dev/video11"
#define SRC_W 2688
#define SRC_H 1520
#define DST_W 224
#define DST_H 224
#define NBUFS 4

static int ae_set_ctrl(int fd, unsigned id, int val) {
    struct v4l2_control c; c.id = id; c.value = val;
    return ioctl(fd, VIDIOC_S_CTRL, &c);
}
static int ae_get_ctrl(int fd, unsigned id) {
    struct v4l2_control c; c.id = id;
    if (ioctl(fd, VIDIOC_G_CTRL, &c) < 0) return -1;
    return c.value;
}
static float ae_brightness(const unsigned char *y) {
    long long sum = 0; int cnt = 0;
    for (int j = 0; j < SRC_H; j += 20) {
        const unsigned char *row = y + j * SRC_W;
        for (int i = 0; i < SRC_W; i += 20) { sum += row[i]; cnt++; }
    }
    return (float)sum / cnt;
}
static void ae_adjust(int sns_fd, int *exp, int *gain, float bright) {
    float err = AE_TARGET - bright;
    if (err > AE_DEADBAND) {
        if (*exp < EXP_MAX) {
            int ne = (int)(*exp * AE_EXP_STEP); if (ne > EXP_MAX) ne = EXP_MAX;
            ae_set_ctrl(sns_fd, V4L2_CID_EXPOSURE, ne); *exp = ne;
        } else if (*gain < GAIN_MAX) {
            int ng = (int)(*gain * AE_GAIN_STEP); if (ng > GAIN_MAX) ng = GAIN_MAX;
            ae_set_ctrl(sns_fd, V4L2_CID_ANALOGUE_GAIN, ng); *gain = ng;
        }
    } else if (err < -AE_DEADBAND) {
        if (*gain > GAIN_MIN) {
            int ng = (int)(*gain / AE_GAIN_STEP); if (ng < GAIN_MIN) ng = GAIN_MIN;
            ae_set_ctrl(sns_fd, V4L2_CID_ANALOGUE_GAIN, ng); *gain = ng;
        } else if (*exp > EXP_MIN) {
            int ne = (int)(*exp / AE_EXP_STEP); if (ne < EXP_MIN) ne = EXP_MIN;
            ae_set_ctrl(sns_fd, V4L2_CID_EXPOSURE, ne); *exp = ne;
        }
    }
}

static char **g_labels = NULL;
static int   g_nlabels = 0;

static long long now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000LL + ts.tv_nsec / 1000000;
}

/* ---------- ImageNet 标签 ---------- */
static int load_labels(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    int cap = 16, n = 0;
    g_labels = malloc(sizeof(char*) * cap);
    while (fgets(line, sizeof(line), f)) {
        size_t len = strlen(line);
        while (len && (line[len-1]=='\n' || line[len-1]=='\r')) line[--len] = 0;
        if (n >= cap) { cap *= 2; g_labels = realloc(g_labels, sizeof(char*)*cap); }
        g_labels[n] = strdup(line);
        n++;
    }
    fclose(f);
    g_nlabels = n;
    printf("[labels] loaded %d classes from %s\n", n, path);
    return 0;
}

/* ---------- NV12 → RGB224 最近邻缩放 ---------- */
static void nv12_to_rgb224(const unsigned char *y, const unsigned char *uv,
                           unsigned char *rgb) {
    const int sx_step = SRC_W * 256 / DST_W;   /* 10.6 缩放 */
    const int sy_step = SRC_H * 256 / DST_H;
    for (int j = 0; j < DST_H; j++) {
        int sy = (j * sy_step) >> 8;
        if (sy >= SRC_H) sy = SRC_H - 1;
        const unsigned char *yrow  = y  + sy * SRC_W;
        const unsigned char *uvrow = uv + (sy/2) * SRC_W;
        unsigned char *out = rgb + j * DST_W * 3;
        for (int i = 0; i < DST_W; i++) {
            int sx = (i * sx_step) >> 8;
            if (sx >= SRC_W) sx = SRC_W - 1;
            int Y = yrow[sx];
            int U = uvrow[sx & ~1];
            int V = uvrow[(sx & ~1) + 1];
            int R = Y + ((1436 * (V - 128)) >> 10);
            int G = Y - ((354 * (U - 128) + 732 * (V - 128)) >> 10);
            int B = Y + ((1814 * (U - 128)) >> 10);
            if (R < 0) R = 0; else if (R > 255) R = 255;
            if (G < 0) G = 0; else if (G > 255) G = 255;
            if (B < 0) B = 0; else if (B > 255) B = 255;
            *out++ = R; *out++ = G; *out++ = B;
        }
    }
}

/* ---------- V4L2 MPLANE 取流 ---------- */
static int cap_init(int *fd_out, void **bufs, size_t *lens) {
    int fd = open(VIDEO_DEV, O_RDWR);
    if (fd < 0) { perror("open video"); return -1; }

    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) { perror("QUERYCAP"); return -1; }
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE)) {
        fprintf(stderr, "not MPLANE capture\n"); return -1;
    }

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width = SRC_W;
    fmt.fmt.pix_mp.height = SRC_H;
    fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
    fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
    fmt.fmt.pix_mp.num_planes = 1;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) { perror("S_FMT"); return -1; }
    printf("[v4l2] %s %dx%d %c%c%c%c planes=%d size=%u\n", VIDEO_DEV,
           fmt.fmt.pix_mp.width, fmt.fmt.pix_mp.height,
           fmt.fmt.pix_mp.pixelformat & 0xff, (fmt.fmt.pix_mp.pixelformat>>8)&0xff,
           (fmt.fmt.pix_mp.pixelformat>>16)&0xff, (fmt.fmt.pix_mp.pixelformat>>24)&0xff,
           fmt.fmt.pix_mp.num_planes, fmt.fmt.pix_mp.plane_fmt[0].sizeimage);

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = NBUFS; req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) { perror("REQBUFS"); return -1; }

    for (int i = 0; i < NBUFS; i++) {
        struct v4l2_buffer buf;
        struct v4l2_plane planes[1];
        memset(&buf, 0, sizeof(buf));
        memset(planes, 0, sizeof(planes));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        buf.length = 1;
        buf.m.planes = planes;
        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) { perror("QUERYBUF"); return -1; }
        bufs[i] = mmap(NULL, planes[0].length, PROT_READ|PROT_WRITE,
                       MAP_SHARED, fd, planes[0].m.mem_offset);
        if (bufs[i] == MAP_FAILED) { perror("mmap"); return -1; }
        lens[i] = planes[0].length;
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) { perror("QBUF"); return -1; }
    }
    if (ioctl(fd, VIDIOC_STREAMON, &req.type) < 0) { perror("STREAMON"); return -1; }
    *fd_out = fd;
    return 0;
}

/* ---------- RKNN ---------- */
static int rknn_init_ctx(rknn_context *ctx, const char *model) {
    int ret = rknn_init(ctx, (void*)model, 0, 0, NULL);
    if (ret < 0) { fprintf(stderr, "rknn_init fail %d\n", ret); return -1; }
    rknn_input_output_num io_num;
    ret = rknn_query(*ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) return -1;
    printf("[rknn] %s: in=%d out=%d\n", model, io_num.n_input, io_num.n_output);
    rknn_tensor_attr in_attr;
    memset(&in_attr, 0, sizeof(in_attr));
    in_attr.index = 0;
    rknn_query(*ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
    printf("[rknn] input dims=[%d,%d,%d,%d] type=%d fmt=%d\n",
           in_attr.dims[0], in_attr.dims[1], in_attr.dims[2], in_attr.dims[3],
           in_attr.type, in_attr.fmt);
    if (in_attr.dims[1] != DST_W || in_attr.dims[2] != DST_H) {
        fprintf(stderr, "model input %dx%d != %dx%d\n", in_attr.dims[1], in_attr.dims[2], DST_W, DST_H);
        return -1;
    }
    return 0;
}

static void softmax_top5(const float *data, int n, int topk,
                         float *probs, int *ids) {
    /* 找 topk（基于 softmax，简单起见遍历） */
    float maxv = data[0];
    for (int i = 1; i < n; i++) if (data[i] > maxv) maxv = data[i];
    float sum = 0;
    float *exp = malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) { exp[i] = expf(data[i] - maxv); sum += exp[i]; }
    for (int k = 0; k < topk; k++) {
        int bi = -1; float bv = -1;
        for (int i = 0; i < n; i++) {
            if (exp[i] > bv) { bv = exp[i]; bi = i; }
        }
        probs[k] = exp[bi] / sum;
        ids[k] = bi;
        exp[bi] = -1;
    }
    free(exp);
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : "/usr/share/model/RK3576/mobilenet_v1.rknn";
    const char *labels = argc > 2 ? argv[2] : "/usr/share/model/imagenet_classes.txt";
    int run_secs = argc > 3 ? atoi(argv[3]) : 20;

    if (getuid() != 0) { fprintf(stderr, "请用 sudo 运行（需要访问 /dev/video11）\n"); return 1; }

    load_labels(labels);

    rknn_context ctx;
    if (rknn_init_ctx(&ctx, model) < 0) return 1;

    int fd; void *bufs[NBUFS]; size_t lens[NBUFS];
    if (cap_init(&fd, bufs, lens) < 0) return 1;

    unsigned char *rgb = malloc(DST_W * DST_H * 3);
    unsigned char *yuv = malloc(lens[0]);

    /* 软件 AE：打开 sensor 控制 */
    int sns_fd = open(SNS_SUBDEV, O_RDWR);
    if (sns_fd < 0) { perror("open sensor subdev"); return 1; }
    int ae_exp = ae_get_ctrl(sns_fd, V4L2_CID_EXPOSURE);
    int ae_gain = ae_get_ctrl(sns_fd, V4L2_CID_ANALOGUE_GAIN);
    if (ae_exp <= 0) ae_exp = EXP_MAX / 2;
    if (ae_gain <= 0) ae_gain = GAIN_MIN;
    printf("[ae] 软件自动曝光: 初始 exp=%d gain=%d 目标亮度=%.0f\n", ae_exp, ae_gain, AE_TARGET);

    rknn_input in;
    memset(&in, 0, sizeof(in));
    in.index = 0;
    in.type = RKNN_TENSOR_UINT8;
    in.fmt = RKNN_TENSOR_NHWC;
    in.size = DST_W * DST_H * 3;
    in.buf = rgb;

    rknn_output out;
    memset(&out, 0, sizeof(out));
    out.index = 0;
    out.want_float = 1;

    long long t_start = now_ms();
    long long t_last = t_start;
    int frames = 0, shown = 0;
    printf("[run] 推理开始，%d 秒（Ctrl-C 退出）\n", run_secs);

    while (run_secs == 0 || now_ms() - t_start < run_secs * 1000LL) {
        struct v4l2_buffer buf;
        struct v4l2_plane planes[1];
        memset(&buf, 0, sizeof(buf));
        memset(planes, 0, sizeof(planes));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.length = 1;
        buf.m.planes = planes;
        if (ioctl(fd, VIDIOC_DQBUF, &buf) < 0) { perror("DQBUF"); break; }

        memcpy(yuv, bufs[buf.index], lens[0]);
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) { perror("QBUF2"); break; }

        long long t0 = now_ms();
        nv12_to_rgb224(yuv, yuv + SRC_W*SRC_H, rgb);       /* 缩放+色彩转换 */
        rknn_inputs_set(ctx, 1, &in);
        rknn_run(ctx, NULL);
        rknn_outputs_get(ctx, 1, &out, NULL);
        long long t1 = now_ms();

        /* 软件 AE：每 AE_EVERY 帧调曝光 */
        if (frames % AE_EVERY == 0) {
            ae_adjust(sns_fd, &ae_exp, &ae_gain, ae_brightness(yuv));
        }
        if (frames % 90 == 0) {
            printf("[ae] 亮度=%.1f exp=%d gain=%d\n",
                   ae_brightness(yuv), ae_exp, ae_gain);
        }

        frames++;

        if (frames % 10 == 1 || shown < 3) {
            float probs[5]; int ids[5];
            softmax_top5((float*)out.buf, 1001, 5, probs, ids);
            printf("\n[帧 %d] 推理耗时 %lldms  (总 %lldms)\n", frames, t1 - t0, now_ms() - t_last);
            for (int k = 0; k < 5; k++) {
                const char *name = (ids[k] < g_nlabels) ? g_labels[ids[k]] : "?";
                printf("  Top%d [%d] %5.1f%%  %s\n", k+1, ids[k], probs[k]*100, name);
            }
            t_last = now_ms();
            shown++;
        }
        rknn_outputs_release(ctx, 1, &out);
    }

    double secs = (now_ms() - t_start) / 1000.0;
    printf("\n==== 统计: %d 帧 / %.1fs = %.1f FPS (含取流+缩放+推理) ====\n",
           frames, secs, frames / secs);

    rknn_destroy(ctx);
    free(rgb); free(yuv);
    for (int i = 0; i < NBUFS; i++) munmap(bufs[i], lens[i]);
    close(fd);
    return 0;
}
