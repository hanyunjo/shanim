#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#define Buf_len 128
#define Listen_que 3

int main(){
    char buf[Buf_len];
    struct sockaddr_in server_addr, client_addr, peer_addr;
    int server_fd, client_fd;
    int len, i, mess_len;

    // socket()
    if((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == -1){
        printf("server can't open socket\n");
        exit(0);
    }

    // init server_addr, buffer to 0
    memset(&server_addr, 0, sizeof(server_addr));
    memset(&buf, 0, sizeof(buf));

    // set sever_addr
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    server_addr.sin_port = htons(50000);

    // bind()
    if(bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) != 0){ 
        printf("failed bind func\n");
        exit(0);
    }

    //listen()
    if(listen(server_fd, Listen_que) != 0){
        printf("failed listen func\n");
        exit(0);
    }

    len = sizeof(client_addr);
    
    client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &len);
    if(client_fd < 0){
        printf("failed accept func\n");
        exit(0);
    }
    
    // host name
    if(getnameinfo((struct sockaddr*)&client_addr, sizeof(client_addr), buf, sizeof(buf), NULL, 0, NI_NAMEREQD)){
        printf("couldn't get hostname\n");
        exit(0);
    }
    printf("Client Info - Host Name : %s\n", buf);
    write(client_fd, buf, strlen(buf));
    read(client_fd, buf, Buf_len);

    // ip
    inet_ntop(AF_INET, &client_addr.sin_addr.s_addr, buf, sizeof(buf));
    printf("Client Info - IP Address : %s\n", buf);
    write(client_fd, buf, strlen(buf));
    read(client_fd, buf, Buf_len);

    // port
    sprintf(buf, "%d", ntohs(client_addr.sin_port));
    printf("Client Info - Port Number : %s\n", buf);
    write(client_fd, buf, strlen(buf));
    read(client_fd, buf, Buf_len);

    close(client_fd);
    close(server_fd);

    return 0;
}