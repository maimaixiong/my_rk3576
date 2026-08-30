/*
 * soft_ae.c — 软件自动曝光守护进程（RK3576 OS04C10）
 *
 * 背景：板载 rkaiq 因设备树拓扑（sensor 挂 rkcif 链，rkaiq 遍历不到）
 *      无法驱动 AE（自动曝光）。本程序用 V4L2 抓帧 + 亮度统计 + 闭环调节
 *      实现等效的自动曝光功能。
 *
 * 逻辑：每 ~170ms 抓一帧，计算平均亮度（Y 平面采样），
 *       与目标亮度比较，按步进调整 sensor 的 exposure / analogue_gain。
 *       策略：优先调 exposure，到上限后调 gain（标准 AE 策略）。
 *
 * 编译: gcc -O2 -o soft_ae soft_ae.c -pthread
 * 运行: sudo ./soft_ae [目标亮度 默认50]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

#define VIDEO_DEV   "/dev/video11"
#define SNS_SUBDEV  "/dev/v4l-subdev3"
#define SRC_W   2688
#define SRC_H   1520
#define NBUFS   4

#define EXP_MIN     2
#define EXP_MAX     1566
#define GAIN_MIN    128
#define GAIN_MAX    1984

#define TARGET_DEF  50.0f        /* 目标平均亮度 0-255 */
#define ADJUST_EVERY 5           /* 每 N 帧调整一次曝光 */
#define EXP_STEP    1.30f        /* exposure 步进倍率 */
#define GAIN_STEP   1.20f        /* gain 步进倍率 */
#define DEADBAND    3.0f         /* 误差死区，避免振荡 */

static long long now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000LL + ts.tv_nsec / 1000000;
}

/* ---------- V4L2 控件读写 ---------- */
static int set_ctrl(int fd, unsigned id, int val) {
    struct v4l2_control c;
    c.id = id;
    c.value = val;
    return ioctl(fd, VIDIOC_S_CTRL, &c);
}
static int get_ctrl(int fd, unsigned id) {
    struct v4l2_control c;
    c.id = id;
    if (ioctl(fd, VIDIOC_G_CTRL, &c) < 0) return -1;
    return c.value;
}

/* ---------- 取流初始化 ---------- */
static int cap_init(int *fd_out, void **bufs, size_t *lens) {
    int fd = open(VIDEO_DEV, O_RDWR);
    if (fd < 0) { perror("open video"); return -1; }

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width = SRC_W;
    fmt.fmt.pix_mp.height = SRC_H;
    fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
    fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
    fmt.fmt.pix_mp.num_planes = 1;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) { perror("S_FMT"); return -1; }

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
        bufs[i] = mmap(NULL, planes[0].length, PROT_READ|PROT_WRITE, MAP_SHARED, fd, planes[0].m.mem_offset);
        if (bufs[i] == MAP_FAILED) { perror("mmap"); return -1; }
        lens[i] = planes[0].length;
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) { perror("QBUF"); return -1; }
    }
    int t = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (ioctl(fd, VIDIOC_STREAMON, &t) < 0) { perror("STREAMON"); return -1; }
    *fd_out = fd;
    return 0;
}

/* 采样亮度（Y 平面，隔行隔列采样 ~1/400 像素） */
static float calc_brightness(const unsigned char *y) {
    long long sum = 0;
    int cnt = 0;
    for (int j = 0; j < SRC_H; j += 20) {
        const unsigned char *row = y + j * SRC_W;
        for (int i = 0; i < SRC_W; i += 20) {
            sum += row[i];
            cnt++;
        }
    }
    return (float)sum / cnt;
}

/* 闭环：err>0 变暗需要增大曝光；err<0 变亮需要减小 */
static void adjust_exposure(int sns_fd, int *exp, int *gain, float target, float bright) {
    float err = target - bright;

    if (err > DEADBAND) {            /* 太暗 → 增大 */
        if (*exp < EXP_MAX) {
            int ne = (int)(*exp * EXP_STEP);
            if (ne > EXP_MAX) ne = EXP_MAX;
            set_ctrl(sns_fd, V4L2_CID_EXPOSURE, ne);
            *exp = ne;
        } else if (*gain < GAIN_MAX) {
            int ng = (int)(*gain * GAIN_STEP);
            if (ng > GAIN_MAX) ng = GAIN_MAX;
            set_ctrl(sns_fd, V4L2_CID_ANALOGUE_GAIN, ng);
            *gain = ng;
        }
    } else if (err < -DEADBAND) {    /* 太亮 → 减小 */
        if (*gain > GAIN_MIN) {
            int ng = (int)(*gain / GAIN_STEP);
            if (ng < GAIN_MIN) ng = GAIN_MIN;
            set_ctrl(sns_fd, V4L2_CID_ANALOGUE_GAIN, ng);
            *gain = ng;
        } else if (*exp > EXP_MIN) {
            int ne = (int)(*exp / EXP_STEP);
            if (ne < EXP_MIN) ne = EXP_MIN;
            set_ctrl(sns_fd, V4L2_CID_EXPOSURE, ne);
            *exp = ne;
        }
    }
}

int main(int argc, char **argv) {
    float target = argc > 1 ? atof(argv[1]) : TARGET_DEF;

    if (getuid() != 0) { fprintf(stderr, "请用 sudo 运行\n"); return 1; }

    setvbuf(stdout, NULL, _IONBF, 0);

    int sns_fd = open(SNS_SUBDEV, O_RDWR);
    if (sns_fd < 0) { perror("open sensor subdev"); return 1; }
    int exp = get_ctrl(sns_fd, V4L2_CID_EXPOSURE);
    int gain = get_ctrl(sns_fd, V4L2_CID_ANALOGUE_GAIN);
    if (exp <= 0) exp = EXP_MAX / 2;
    if (gain <= 0) gain = GAIN_MIN;
    printf("[soft_ae] 初始 exposure=%d gain=%d 目标亮度=%.0f\n", exp, gain, target);

    int fd; void *bufs[NBUFS]; size_t lens[NBUFS];
    if (cap_init(&fd, bufs, lens) < 0) return 1;
    unsigned char *yuv = malloc(lens[0]);

    long long t0 = now_ms();
    long long last_log = 0;
    int frame = 0;

    printf("[soft_ae] 运行中 (Ctrl-C 退出)...\n");
    while (1) {
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

        frame++;
        if (frame % ADJUST_EVERY == 0) {
            float bright = calc_brightness(yuv);
            adjust_exposure(sns_fd, &exp, &gain, target, bright);

            long long now = now_ms();
            if (now - last_log > 2000) {
                int e2 = get_ctrl(sns_fd, V4L2_CID_EXPOSURE);
                int g2 = get_ctrl(sns_fd, V4L2_CID_ANALOGUE_GAIN);
                printf("[soft_ae] 亮度=%.1f 目标=%.0f exposure=%d gain=%d 运行%.0fs\n",
                       bright, target, e2, g2, (now - t0) / 1000.0);
                last_log = now;
            }
        }
    }

    free(yuv);
    return 0;
}
