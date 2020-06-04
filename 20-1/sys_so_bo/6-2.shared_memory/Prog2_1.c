#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

int main(){
    int shmid, count;
    void *memory = (void *)0;
    char *text;

    if((shmid = shmget((key_t)1735, sizeof(char)*10, IPC_CREAT|0666)) == -1){
        printf("failed shmget func\n");
        exit(1);
    }
    
    if((memory = shmat(shmid, (void *)0, 0)) == (void *)-1){
        printf("failed shmat func\n");
        exit(1);
    }

    text = (char *)memory;
    strcpy(text, "System");

    for(count = 0; count < 10; count++){
        printf("A : %s\n", text);
        strcpy(text, "System");
        sleep(1);
    }

    return 0;
}
