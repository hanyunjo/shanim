#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

typedef struct {
    char text[7];
    int a;
    int b;
} shm_sem;

int main(){
    int shmid, count;
    void *memory = (void *)0;
    shm_sem *sha;

    if((shmid = shmget((key_t)1735, sizeof(shm_sem), IPC_CREAT|0666)) == -1){
        printf("failed shmget func\n");
        exit(1);
    }
    
    if((memory = shmat(shmid, (void *)0, 0)) == (void *)-1){
        printf("failed shmat func\n");
        exit(1);
    }

    sha = (shm_sem *)memory;

    sha->a = 0;
    sha->b = 0;

    for(count = 0; count < 10; count++){
        strcpy(sha->text, "System");
        sleep(1);
        printf("A : %s\n", sha->text);

        if(count == 9) sha->a = 2;
    }

    if((sha->a == 2 && sha->b == 0) || (sha->a == 2 && sha->b == 2)){
        if(shmdt(memory) == -1){
            printf("shmdt failed\n");
        }

        if(shmctl(shmid, IPC_RMID, NULL) == -1){
            printf("shmctl failed\n");
        }
    }

    return 0;
}
