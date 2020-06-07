#include <stdio.h>
#include <pthread.h>

void *function1(void *arg);
void *function2(void *arg);

typedef struct {
    int num[5];
    pthread_barrier_t *barr;
} barr_thr;

int main(){
    pthread_t tid1, tid2;
    barr_thr threa[2];
    int i;
    pthread_barrier_t bar;

    pthread_barrier_init(&bar, NULL, 2);

    for(i = 0; i < 5; i++){
        threa[0].num[i] = i;
        threa[1].num[i] = i+5;
    }
    threa[0].barr = &bar;
    threa[1].barr = &bar;

    pthread_create(&tid1, NULL, function1, (void *)&threa[0]);
    pthread_create(&tid2, NULL, function2, (void *)&threa[1]);

    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);

    for(i = 0; i < 10; i++){
        if(i < 5) printf("%d\n",threa[0].num[i]);
        else printf("%d\n",threa[1].num[i-5]);
    }

    return 0;
}

void *function1(void *arg){ // num = 1 : 1~3, 8~10
    barr_thr *threa = (barr_thr *)arg;
    int i;

    for(i = 0; i < 5; i++) threa->num[i] = (i+1)*(i+1);

    pthread_barrier_wait(threa->barr);
    pthread_exit((void *) 0);
}

void *function2(void *arg){ // num = 2 : 4~7, 11, END
    barr_thr *threa = (barr_thr *)arg;
    int i;
    
    for(i = 0; i < 5; i++) threa->num[i] = (i+6)*(i+6);

    pthread_barrier_wait(threa->barr);
    pthread_exit((void *) 0);
}