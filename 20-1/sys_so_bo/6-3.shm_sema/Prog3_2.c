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
    struct sembuf semb;
} shm_sem;

int main(){
    int shmid, count, semid;
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
    sha->semb.sem_flg = SEM_UNDO;
    sha->semb.sem_num = 0;

    if((semid = semget((key_t)9432, 1, IPC_CREAT|0666)) == -1){
        printf("failed semget func\n");
        exit(1);
    }

    if(semctl(semid, 0, SETVAL, 1) == -1){
        printf("failed semctl1 func\n");
        exit(1);
    }

    // function
    for(count = 0; count < 10; count++){
        sha->semb.sem_op = -1;
        if((semop(semid, &sha->semb, 1)) == -1){
            printf("failed get sem\n");
            exit(1);
        }
        
        strcpy(sha->text, "Prog");
        sleep(1);
        printf("B : %s\n", sha->text);

        sha->semb.sem_op = 1;
        if((semop(semid, &sha->semb, 1)) == -1){
            printf("failed get sem\n");
            exit(1);
        }
    }

    if(shmdt(memory) == -1){
        printf("failed shmdt func\n");
        exit(1);
    }

    if(shmctl(smId, IPC_RMID, 0) == -1)
    {
        printf("failed remove shm\n");
        return 0;
    }

    if(semctl(semId, 0, IPC_RMID, 0) == -1)
    {
        fprintf(stderr, "failed remove sem\n");
        exit(0);
    }

    return 0;
}
