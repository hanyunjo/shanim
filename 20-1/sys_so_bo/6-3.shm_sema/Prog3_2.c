#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
/*
void getsem(int semid){
    struct sembuf getbuf;
    getbuf.sem_num = 0;
    getbuf.sem_op = -1;
    getbuf.sem_flg = SEM_UNDO;
    if((semop(semid, &getbuf, 1)) == -1){
        printf("failed get sem\n");
            exit(1);
    }
}

void returnsem(int semid){
    struct sembuf returnbuf;
    returnbuf.sem_num = 0;
    returnbuf.sem_op = 1;
    returnbuf.sem_flg = SEM_UNDO;
    if((semop(semid, &returnbuf, 1)) == -1){
        printf("failed return sem\n");
            exit(1);
    }
}*/

int main(){
    int shmid, count, semid;
    void *memory = (void *)0;
    char *text;
    union semun{
        int val;
        struct semid_ds *buf;
        unsigned short int *array;
    } arg;

    struct sembuf semb;

    // shared_memory
    if((shmid = shmget((key_t)1735, sizeof(char)*10, IPC_CREAT|0666)) == -1){
        printf("failed shmget func\n");
        exit(1);
    }
    
    if((memory = shmat(shmid, (void *)0, 0)) == (void *)-1){
        printf("failed shmat func\n");
        exit(1);
    }

    text = (char *)memory;

    // semaphore
    semb.sem_flg = SEM_UNDO;
    semb.sem_num = 0;

    if((semid = semget((key_t)9432, 1, IPC_CREAT|0666)) == -1){
        printf("failed semget func\n");
        exit(1);
    }

    arg.val = 1;
    if(semctl(semid, 0, SETVAL, arg) == -1){
        printf("failed semctl1 func\n");
        exit(1);
    }

    // function
    for(count = 0; count < 10; count++){
        //getsem(semid);

        semb.sem_op = -1;
        if((semop(semid, &semb, 1)) == -1){
            printf("failed get sem\n");
            exit(1);
        }
        
        strcpy(text, "Prog");
        sleep(1);
        printf("B : %s\n", text);
        
        semb.sem_op = 1;
        if((semop(semid, &semb, 1)) == -1){
            printf("failed get sem\n");
            exit(1);
        }
        //returnsem(semid);
    }

    if(shmdt(memory) == -1){
        printf("failed shmdt func\n");
        exit(1);
    }

    return 0;
}
