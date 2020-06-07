#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>

#define buf_size 100

int result[10];

int get_w_lock(int fd);
int get_r_lock(int fd);
int unlock(int fd);
void *function(void *i);

int main(){

    pthread_t tid1, tid2;

    pthread_create(&tid1, NULL, function, (void *)1);
    pthread_create(&tid2, NULL, function, (void *)2);

    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);

    return 0;
}

void *function(void *i){
    int fd, *num, tmp;
    char *name;
    int fir, sec, thir;

    num = (int *)i;

    if((fd = open("data.txt", O_CREAT|O_RDWR)) == -1){
        printf("file open error\n");
        exit(1);
    }

    tmp = 0;
    if(i == 1){
        if(get_r_lock(fd) == -1){
            printf("file read_lock error\n");
            exit(1);
        }

        while(fscanf(fd, "%d %d %d", &fir, &sec, &thir) != EOF){
            if((fir + sec + thir) > 150) result[tmp] = 1;
            else result[tmp] = 0;
        }
    }
    else if(i == 2){
        sleep(1);
        if(get_w_lock(fd) == -1){
            printf("file write_lock error\n");
            exit(1);
        }

        while(tmp < 10){
            fprintf(fd, "%d %d %d %d\n", fir, sec, thir, result[tmp]);
        }
    }

    if(unlock(fd) == -1){
        printf("file unlock error\n");
        exit(1);
    }

    close(fd);

}

int get_w_lock(int fd){
    struct flock lock;
    lock.l_type = F_WRLCK;
    lock.l_start = 0;
    lock.l_whence = SEEK_SET;
    lock.l_len = 0;

    return fcntl(fd, F_SETLKW, &lock); // SETLKW는 lock을 얻을 때까지 blocking한다.
}

int get_r_lock(int fd){
    struct flock lock;
    lock.l_type = F_RDLCK;
    lock.l_start = 0;
    lock.l_whence = SEEK_SET;
    lock.l_len = 0;

    return fcntl(fd, F_SETLKW, &lock); // SETLKW는 lock을 얻을 때까지 blocking한다.
}

int unlock(int fd){
    struct flock lock;
    lock.l_type = F_UNLCK;
    lock.l_start = 0;
    lock.l_whence = SEEK_SET;
    lock.l_len = 0;

    return fcntl(fd, F_SETLK, &lock); // 
}