#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>

typedef struct {
    char text[7];
    int semid;
    struct sembuf getbuf;
    struct sembuf rebuf;
} shm_sem;

int main(){
    int shmid, count;
    void *memory = (void *)0;
    //char *text;
    shm_sem *sha;
    //struct sembuf semb;

    // shared memory
    if((shmid = shmget((key_t)1735, sizeof(shm_sem), IPC_CREAT|0666)) == -1){
        printf("failed shmget func\n");
        exit(1);
    }
    
    if((memory = shmat(shmid, (void *)0, 0)) == (void *)-1){
        printf("failed shmat func\n");
        exit(1);
    }

    //text = (char *)memory;
    sha = (shm_sem *)memory;



    // semaphore
    if((sha->semid = semget((key_t)9432, 1, IPC_CREAT|0666)) == -1){
        printf("failed semget func\n");
        exit(1);
    }

    if(semctl(sha->semid, 0, SETVAL, 1) == -1){
        printf("failed semctl1 func\n");
        exit(1);
    }




    // function
    for(count = 0; count < 10; count++){
        sha->getbuf.sem_num = 0;
        sha->getbuf.sem_op = -1;
        sha->getbuf.sem_flg = SEM_UNDO;
        if((semop(sha->semid, &sha->getbuf, 1)) == -1){
            printf("failed get sem\n");
            exit(1);
        }
        
        strcpy(sha->text, "Sema");
        sleep(1);
        printf("A : %s\n", sha->text);

        sha->rebuf.sem_num = 0;
        sha->rebuf.sem_op = 1;
        sha->rebuf.sem_flg = SEM_UNDO;
        if((semop(sha->semid, &sha->rebuf, 1)) == -1){
            printf("failed return sem\n");
            exit(1);
        }
    }

    if(shmdt(memory) == -1){
        printf("failed shmdt func\n");
        exit(1);
    }

    return 0;
}
