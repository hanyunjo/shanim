#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

#define buf_size 9

int real = 0;
char result[10][11];

int get_w_lock(int fd);
int get_r_lock(int fd);
int unlock(int fd);
void *function1();
void *function2();

int main(){

    pthread_t tid1, tid2;

    pthread_create(&tid1, NULL, function1, NULL);
    pthread_create(&tid2, NULL, function2, NULL);

    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);

    return 0;
}

void *function1(){
    int fd, tmp2;
    char rbuf[buf_size];
    int fir, sec, thir;

    if((fd = open("data.txt", O_CREAT|O_RDWR)) == -1){
        printf("file open error\n");
        exit(1);
    }

    tmp2 = 0;

    memset(rbuf, 0x00, buf_size);

    if(get_r_lock(fd) == -1){
        printf("file read_lock error\n");
        exit(1);
    }
    
    while(read(fd, rbuf, buf_size) > 8){
        if((int)strlen(rbuf) == 9) rbuf[(int)strlen(rbuf)-1] = '\0';
        
        char *num = strtok(rbuf, " ");
        fir = atoi(num);
        while(num != NULL){
            num = strtok(NULL, " ");
            if(tmp2 == 0){
                sec = atoi(num);
                tmp2++;
            }
            else if(tmp2 == 1) {
                thir = atoi(num);
                break;
            }
        }
        tmp2 = 0;

        if((fir + sec + thir) > 150) sprintf(result[real], "%d %d %d 1\n", fir, sec, thir);
        else sprintf(result[real], "%d %d %d 0\n", fir, sec, thir);

        real++;
    }

    if(unlock(fd) == -1){
        printf("file unlock error\n");
        exit(1);
    }

    close(fd);
}
void *function2(){
    int fd, tmp;

    if((fd = open("data.txt", O_CREAT|O_RDWR)) == -1){
        printf("file open error\n");
        exit(1);
    }

    tmp = 0;
    sleep(1);

    if(get_w_lock(fd) == -1){
        printf("file write_lock error\n");
        exit(1);
    }

    write(fd, result[tmp], strlen(result[tmp]));

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