#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#define buf_size 512

int main(){
        int fd[2], status;
        pid_t pid;
        char buf[buf_size];

        if(pipe(fd) < 0){
                printf("pipe error\n");
                exit(1);
        }
        if((pid=fork()) < 0){
                printf("fork error\n");
                exit(1);
        }

        if(pid > 0){ //parent = server
                close(fd[0]);
                strcpy(buf,"do something");
                write(fd[1],buf,strlen(buf));
                waitpid(pid, &status, 0);
                close(fd[1]);
        }
        else if(pid == 0){  //child = client
                close(fd[1]);
                read(fd[0],buf,buf_size);
                printf("result : %s\n",buf);
                close(fd[0]);
        }

        return 0;
}