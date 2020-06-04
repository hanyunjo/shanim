#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

int main(){
    int shmid, count;
    key_t key = 1735;
    size_t size = 10;
    void *memory = (void *)0;
    char *text;

    if((shmid = shmget(key, size, IPC_CREAT|0666)) == -1){
        printf("failed shmget func\n");
        exit(1);
    }
    
    if((memory = shmat(shmid, (void *)0, 0)) == (void *)-1){
        printf("failed shmat func\n");
        exit(1);
    }

    text = (char *)memory;

    for(count = 0; count < 10; count++){
        strcpy(text, "System");
        printf("A : %s\n", text);
        sleep(1);
    }

    return 0;
}
