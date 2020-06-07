#include <stdio.h>
#include <pthread.h>

int count = 1;
pthread_mutex_t cou_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t fun_cond = PTHREAD_COND_INITIALIZER;

void *function1();
void *function2();

int main(){
        pthread_t tid1, tid2;

        pthread_create(&tid1, NULL, function1, NULL);
        pthread_create(&tid2, NULL, function2, NULL);

        pthread_join(tid1, NULL);
        pthread_join(tid2, NULL);

        pthread_mutex_destroy(&cou_mutex);

        return 0;
}

void *function1(){ // num = 1 : 1~3, 8~10
        pthread_mutex_lock(&cou_mutex);
        while(1){
                if(count == 4){
                        pthread_cond_signal(&fun_cond);
                        pthread_cond_wait(&fun_cond, &cou_mutex);
                }
                else if(count == 11){
                        pthread_cond_signal(&fun_cond);
                        pthread_mutex_unlock(&cou_mutex);
                        break;
                }
                printf("By function1, count value : %d\n", count);
                count++;
        }
        pthread_exit((void *) 0);
}

void *function2(){ // num = 2 : 4~7, 11, END
        pthread_mutex_lock(&cou_mutex);
        while(1){
                if(count < 4) pthread_cond_wait(&fun_cond, &cou_mutex);
                else if(count == 8){
                        pthread_cond_signal(&fun_cond);
                        pthread_cond_wait(&fun_cond, &cou_mutex);
                }
                else if( count == 12) {
                        pthread_mutex_unlock(&cou_mutex);
                        break;
                }
                printf("By function2, count value : %d\n", count);
                count++;
        }
        pthread_exit((void *) 0);
}