// Minimal multi-threaded STREAM benchmark (Copy / Scale / Add / Triad) in C.
// Used to measure sustained DRAM bandwidth on the benchmark device (Apple M4 / Raspberry Pi).
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef STREAM_ARRAY_SIZE
// 64M doubles = 512 MiB per array -> 1.5 GiB total, well above M4's 16 MiB L2 / SLC.
#define STREAM_ARRAY_SIZE (64 * 1024 * 1024)
#endif

#ifndef NTIMES
#define NTIMES 10
#endif

static double *a, *b, *c;

typedef struct {
    size_t start;
    size_t end;
    double scalar;
    int op; // 0=copy, 1=scale, 2=add, 3=triad, 4=read_only
    double sink;
} task_t;

static void *worker(void *arg) {
    task_t *t = (task_t *)arg;
    size_t s = t->start, e = t->end;
    double k = t->scalar;
    if (t->op == 0) {
        for (size_t i = s; i < e; i++) c[i] = a[i];
    } else if (t->op == 1) {
        for (size_t i = s; i < e; i++) b[i] = k * c[i];
    } else if (t->op == 2) {
        for (size_t i = s; i < e; i++) c[i] = a[i] + b[i];
    } else if (t->op == 3) {
        for (size_t i = s; i < e; i++) a[i] = b[i] + k * c[i];
    } else if (t->op == 4) {
        double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
        size_t i = s;
        for (; i + 4 <= e; i += 4) {
            acc0 += a[i];
            acc1 += a[i + 1];
            acc2 += a[i + 2];
            acc3 += a[i + 3];
        }
        for (; i < e; i++) acc0 += a[i];
        t->sink = (acc0 + acc1) + (acc2 + acc3);
    }
    return NULL;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void run_op(int nthreads, int op, double scalar, double *sink_out) {
    pthread_t th[64];
    task_t tasks[64];
    size_t chunk = STREAM_ARRAY_SIZE / (size_t)nthreads;
    for (int t = 0; t < nthreads; t++) {
        tasks[t].start = (size_t)t * chunk;
        tasks[t].end = (t == nthreads - 1) ? STREAM_ARRAY_SIZE : (size_t)(t + 1) * chunk;
        tasks[t].scalar = scalar;
        tasks[t].op = op;
        tasks[t].sink = 0.0;
        pthread_create(&th[t], NULL, worker, &tasks[t]);
    }
    double tot = 0.0;
    for (int t = 0; t < nthreads; t++) {
        pthread_join(th[t], NULL);
        tot += tasks[t].sink;
    }
    if (sink_out) *sink_out = tot;
}

int main(int argc, char **argv) {
    int nthreads = (argc > 1) ? atoi(argv[1]) : 4;
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 64) nthreads = 64;

    size_t bytes = (size_t)STREAM_ARRAY_SIZE * sizeof(double);
    posix_memalign((void **)&a, 64, bytes);
    posix_memalign((void **)&b, 64, bytes);
    posix_memalign((void **)&c, 64, bytes);
    for (size_t i = 0; i < STREAM_ARRAY_SIZE; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }

    const char *names[5] = {"Copy", "Scale", "Add", "Triad", "Read"};
    const double mult[5] = {2.0, 2.0, 3.0, 3.0, 1.0};
    double best[5] = {1e9, 1e9, 1e9, 1e9, 1e9};
    volatile double sink = 0.0;

    // Warmup
    for (int op = 0; op < 5; op++) {
        double s = 0.0;
        run_op(nthreads, op, 3.0, &s);
        sink += s;
    }

    for (int k = 0; k < NTIMES; k++) {
        for (int op = 0; op < 5; op++) {
            double s = 0.0;
            double t0 = now_sec();
            run_op(nthreads, op, 3.0, &s);
            double dt = now_sec() - t0;
            sink += s;
            if (dt < best[op]) best[op] = dt;
        }
    }

    printf("{\"threads\": %d, \"array_mib\": %.1f", nthreads, (double)bytes / (1024.0 * 1024.0));
    for (int op = 0; op < 5; op++) {
        double gbs = (mult[op] * (double)bytes / best[op]) / 1e9;
        printf(", \"%s_gbs\": %.2f", names[op], gbs);
    }
    printf("}\n");

    free(a);
    free(b);
    free(c);
    return (sink == -123456789.0) ? 1 : 0;
}
