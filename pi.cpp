#include<stdio.h>
#include<omp.h>
#include "openmp.h"
static long long num_steps = 100000;//Խ��ֵԽ��ȷ
double step;
#define NUM_THREADS 2

//�����㷨
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

//ʹ�ò������л�
void pi_1() {
    int i, id;
    double x, pi, sum[NUM_THREADS];
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);//����2�߳�
#pragma omp parallel private(i,id,x)//������ʼ��ÿ���̣߳�0��1������ִ�иô���
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
//��2���̲߳μӼ��㣬�����߳�0���е�����0��2��4��...�߳�1���е�����1��3��5��...

//ʹ�ù�������ṹ���л�
void pi_2() {
    int i;
    double pi, sum[NUM_THREADS];
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);//����2�߳�
#pragma omp parallel//������ʼ��ÿ���̣߳�0��1������ִ�иô���
    {
        double x;
        int id;
        id = omp_get_thread_num();
        sum[id] = 0;
#pragma omp for//δָ��chunk������ƽ����������̣߳�0��1������������
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
//��2���̲߳μӼ��㣬�����߳�0���е�����0~49999���߳�1���е�����50000~99999

//ʹ��private�Ӿ��critical���ֲ��л�
void pi_3() {
    int i;
    double pi = 0.0, sum = 0.0, x = 0.0;
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);//����2�߳�
#pragma omp parallel private(i, x, sum)//���Ӿ��ʾi��x��sum��������ÿ���߳���˽�е�
    {
        int id = omp_get_thread_num();
        for (i = id, sum = 0.0; i < num_steps; i += NUM_THREADS) {
            x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
#pragma omp critical//ָ���������ͬһʱ��ֻ����һ���߳̽���ִ��
        pi += sum * step;
    }
    printf("%lf\n", pi);
}
//��2���̲߳μӼ��㣬�����߳�0���е�����0��2��4��...�߳�1���е�����1��3��5��...����ָ��Ϊcritical�Ĵ�������ڱ�0�߳�ִ��ʱ��1�̵߳�ִ��Ҳ����ô���Σ�������������֪��0�߳��˳��ٽ���

//ʹ�ò��й�Լ
void pi_4() {
    int i;
    double pi = 0.0, sum = 0.0, x = 0.0;
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);//����2�߳�
#pragma omp parallel for reduction(+:sum) private(x)//ÿ���̱߳���һ��˽�п���sum��xΪ�߳�˽�У������߳�������sum����+��Լ��������sum��ȫ��ֵ
    for (i = 1; i <= num_steps; i++) {
        x = (i - 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    pi = sum * step;
    printf("%lf\n", pi);
}
//��2���̲߳μӼ��㣬�����߳�0���е�����0~49999���߳�1���е�����50000~99999