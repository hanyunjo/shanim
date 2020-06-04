#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#define Buf_len 128

int main(){

    int sock, i, l;
    struct sockaddr_in server_addr;
    char buf[Buf_len+1], next[5] = "next";

    if((sock = socket(PF_INET, SOCK_STREAM, 0)) < 0){
            printf("failed socket func\n");
            exit(0);
    }

    memset(&server_addr, 0, sizeof(server_addr));

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(50000);
    server_addr.sin_addr.s_addr = inet_addr("10.178.0.2");
    // = inet_pton( AF_INET, "34.64.182.41", &server_addr.sin_addr.s_addr );

    if(connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) != 0){
        printf("failed connect func\n");
        exit(0);
    }

    l = 0;
    while((i = read(sock, buf, Buf_len)) > 0){
        buf[i] = '\0';
        if(l == 0){
            printf("Get From Server - My Host Name : %s\n", buf);
            write(sock, next, strlen(next));
        }
        else if(l == 1){
            printf("Get From Server - My IP Address : %s\n", buf);
            write(sock, next, strlen(next));
        }
        else if(l == 2){
            printf("Get From Server - My Port Number : %s\n", buf);
            write(sock, next, strlen(next));
        }
        l++;
        if(l == 3) break;
    }
    close(sock);

    return 0;
}