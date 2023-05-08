#include<stdio.h>
#include<omp.h>
#include "openmp.h"
static long long num_steps = 100000;//越大值越精确
double step;
#define NUM_THREADS 2

//串行算法
void pi() {
    int i;
    double x, pi, sum = 0.0;
    step = 1.0 / (double)num_steps;
    for (i = 1; i <= num_steps; i++) {
        x = (i - 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    printf("%lf\n", pi);
}

//使用并行域并行化
void pi_1() {
    int i, id;
    double x, pi, sum[NUM_THREADS];
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);//设置2线程
#pragma omp parallel private(i,id,x)//并行域开始，每个线程（0和1）都会执行该代码
    {
        id = omp_get_thread_num();
        for (i = id, sum[id] = 0.0; i < num_steps; i = i + NUM_THREADS) {
            x = (i + 0.5) * step;
            sum[id] += 4.0 / (1.0 + x * x);
        }
    }
    for (i = 0, pi = 0.0; i < NUM_THREADS; i++) {
        pi += sum[i] * step;
    }
    printf("%lf\n", pi);
}
//共2个线程参加计算，其中线程0进行迭代步0，2，4，...线程1进行迭代步1，3，5，...

//使用共享任务结构并行化
void pi_2() {
    int i;
    double pi, sum[NUM_THREADS];
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);//设置2线程
#pragma omp parallel//并行域开始，每个线程（0和1）都会执行该代码
    {
        double x;
        int id;
        id = omp_get_thread_num();
        sum[id] = 0;
#pragma omp for//未指定chunk，迭代平均分配给各线程（0和1），连续划分
        for (i = 0; i < num_steps; i++) {
            x = (i + 0.5) * step;
            sum[id] += 4.0 / (1.0 + x * x);
        }
    }
    for (i = 0, pi = 0.0; i < NUM_THREADS; i++) {
        pi += sum[i] * step;
    }
    printf("%lf\n", pi);
}
//共2个线程参加计算，其中线程0进行迭代步0~49999，线程1进行迭代步50000~99999

//使用private子句和critical部分并行化
void pi_3() {
    int i;
    double pi = 0.0, sum = 0.0, x = 0.0;
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);//设置2线程
#pragma omp parallel private(i, x, sum)//该子句表示i，x，sum变量对于每个线程是私有的
    {
        int id = omp_get_thread_num();
        for (i = id, sum = 0.0; i < num_steps; i += NUM_THREADS) {
            x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
#pragma omp critical//指定代码段在同一时刻只能由一个线程进行执行
        pi += sum * step;
    }
    printf("%lf\n", pi);
}
//共2个线程参加计算，其中线程0进行迭代步0，2，4，...线程1进行迭代步1，3，5，...当被指定为critical的代码段正在被0线程执行时，1线程的执行也到达该代码段，则它将被阻塞知道0线程退出临界区

//使用并行规约
void pi_4() {
    int i;
    double pi = 0.0, sum = 0.0, x = 0.0;
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);//设置2线程
#pragma omp parallel for reduction(+:sum) private(x)//每个线程保留一份私有拷贝sum，x为线程私有，最后对线程中所有sum进行+规约，并更新sum的全局值
    for (i = 1; i <= num_steps; i++) {
        x = (i - 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    pi = sum * step;
    printf("%lf\n", pi);
}
//共2个线程参加计算，其中线程0进行迭代步0~49999，线程1进行迭代步50000~99999
