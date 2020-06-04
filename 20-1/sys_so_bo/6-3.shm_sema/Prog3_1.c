#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>



int main(){
    int shmid, count, semid;
    void *memory = (void *)0;
    char *text;

    struct sembuf semb;

    if((shmid = shmget((key_t)1735, sizeof(char)*10, IPC_CREAT|0666)) == -1){
        printf("failed shmget func\n");
        exit(1);
    }
    
    if((memory = shmat(shmid, (void *)0, 0)) == (void *)-1){
        printf("failed shmat func\n");
        exit(1);
    }

    text = (char *)memory;

    semb.sem_flg = SEM_UNDO;
    semb.sem_num = 0;

    if((semid = semget((key_t)9432, 1, IPC_CREAT|0666)) == -1){
        printf("failed semget func\n");
        exit(1);
    }

    if(semctl(semid, 0, SETVAL, 1) == -1){
        printf("failed semctl func\n");
        exit(1);
    }

    for(count = 0; count < 10; count++){
        semb.sem_op = -1;
        if(semop(semid, &semb, 1) == -1){
            printf("failed set -1\n");
            exit(1);
        }
        strcpy(text, "Sema");
        sleep(1);
        printf("A : %s\n", text);
        semb.sem_op = 1;
        if(semop(semid, &semb, 1) == -1){
            printf("failed set 1\n");
            exit(1);
        }
    }

    return 0;
}
