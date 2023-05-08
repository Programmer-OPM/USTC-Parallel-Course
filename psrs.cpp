#include<stdlib.h>
#include<omp.h>
#include<algorithm>
#include<fstream>
#include<iostream>
#include "openmp.h"

using namespace std;

const int NUM_THREADS = 4;

int** temp;
int** segment;//��������������Ԫ�ĸ��Ե�����κ�
int* sizes;
int* sample;
int* pivot_number;

void show(){

	// ���������С
	int num;
	cout << "Please input the num:\n";
	cin >> num;

	// �����������
	int beginpoint = 0;
	beginpoint = NUM_THREADS - num % NUM_THREADS;

	//�����������
	srand(time(0));
	int* data = new int[num + beginpoint];
	for (int i = 0; i < num; i++){
		data[i] = rand() % 200;
	}
	for (int i = num; i < num + beginpoint; i++){
		data[i] = 0;
	}

	// ������������
	cout << "������飺\n";
	for (int i = 0; i < num; i++){
		cout << data[i] << " ";
	}
	cout << endl;
	num += beginpoint;

	// ��������
	PSRS(data, num, NUM_THREADS);

	// �������
	cout << "PSRS�����\n";
	for (int i = beginpoint; i < num; i++){
		cout << data[i] << " ";
	}
	cout << endl;
	cout << "���������С:" << num - beginpoint << ",�߳���:" << NUM_THREADS << endl;

	// �ͷŶ�̬����
	free(data);
	for (int i = 0; i < NUM_THREADS; i++){
		free(temp[i]);
		free(segment[i]);
	}
}

void PSRS(int* data, int size, int NUM_THREADS){
	int localN = size / NUM_THREADS;
	sample = (int*)malloc(sizeof(int) * (NUM_THREADS * NUM_THREADS));
	pivot_number = (int*)malloc(sizeof(int) * (NUM_THREADS - 1));

	// ���Ȼ��� + �ֲ����� + �������
#pragma omp parallel num_threads(NUM_THREADS)
	{
		int id = omp_get_thread_num();
		int localLeft = id * localN;
		int localRight = (id + 1) * localN;
		int step = localN / NUM_THREADS;
		sort(data + localLeft, data + localRight);
		for (int i = 0; i < NUM_THREADS; i++){
			sample[id * NUM_THREADS + i] = *(data + (id * localN + i * step));
		}
	}
	//��������
	sort(sample, sample + NUM_THREADS * NUM_THREADS);

	//ѡ����Ԫ
	for (int i = 1; i < NUM_THREADS; i++){
		pivot_number[i - 1] = sample[i * NUM_THREADS];
	}
	segment = (int**)malloc(sizeof(int*) * NUM_THREADS);
	for (int i = 0; i < NUM_THREADS; i++){
		segment[i] = (int*)malloc(sizeof(int) * (NUM_THREADS + 1));
	}

	//��Ԫ����
#pragma omp parallel num_threads(NUM_THREADS)
	{
		int id = omp_get_thread_num();
		int localLeft = id * localN;
		int localRight = (id + 1) * localN;
		int count = 0;
		int mleft = localLeft;
		segment[id][count] = 0;
		segment[id][NUM_THREADS] = localN;
		for (; mleft < localRight && count < NUM_THREADS - 1;){
			if (*(data + mleft) <= pivot_number[count]){
				mleft += 1;
			}
			else{
				count += 1;
				segment[id][count] = mleft - localLeft;
			}
		}
		for (; count < NUM_THREADS - 1; count++){
			segment[id][count + 1] = mleft - localLeft;
		}
	}
	
	// �ͷŶ�̬����
	free(sample);
	free(pivot_number);
	sizes = (int*)malloc(sizeof(int) * NUM_THREADS);
	temp = (int**)malloc(sizeof(int*) * NUM_THREADS);
	
	//ȫ�ֽ���
	// ����ÿһ�εĴ�С����̬��ʼ��
	for (int i = 0; i < NUM_THREADS; i++){
		sizes[i] = 0;
		for (int j = 0; j < NUM_THREADS; j++){
			sizes[i] += (segment[j][i + 1] - segment[j][i]);
			//cout << sizes[i] << endl;
		}
		temp[i] = (int*)malloc(sizeof(int) * sizes[i]);
		int index = 0;
		for (int j = 0; j < NUM_THREADS; j++){
			for (int k = segment[j][i]; k < segment[j][i + 1]; k++){
				data[localN * j + k];
				temp[i][index] = data[localN * j + k];
				index += 1;
			}
		}
	}

	//�鲢����
#pragma omp parallel num_threads(NUM_THREADS)
	{
		int id = omp_get_thread_num();
		sort(temp[id], temp[id] + sizes[id]);
	}
	int i = 0;
	for (int j = 0; j < NUM_THREADS; j++){
		for (int k = 0; k < sizes[j]; k++){
			*(data + i) = *(temp[j] + k);
			i++;
		}
	}
	free(sizes);
}