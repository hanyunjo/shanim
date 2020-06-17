#include <stdio.h>
#include <netdb.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define Buf_len 128
#define Listen_que 3

void check_cipher(int client_fd);
int get_privacy_and_check_database(int client_fd);
void send_question(int client_fd);
void send_append_question(int client_fd);
int check_database(char privacy[]);

int main(){
    char buf[Buf_len];
    struct sockaddr_in server_addr, client_addr, peer_addr;
    int server_fd, client_fd;
    int len, i, mess_len, check_num;
    int in_database;

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
        printf("failed bine func\n");
        exit(0);
    }

    //listen()
    if(listen(server_fd, Listen_que) != 0){
        printf("failed listen func\n");
        exit(0);
    }

    len = sizeof(client_addr);
    
    // accept
    client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &len);
    if(client_fd < 0){
        printf("failed accept func\n");
        exit(0);
    }

    // 2
    check_cipher(client_fd);

    // 3 ~ 5 /  b ~ c
    in_database = get_privacy_and_check_database(client_fd);
    if(in_database == 1){
        int appen;
        // b 
        sprintf(buf, "If you want new one, type 1 / If you want to append, type 2 : ");
        write(client_fd, buf, strlen(buf));
        read(client_fd, buf, Buf_len); 
        appen = atoi(buf);
        // c
        if(appen == 1) send_question(client_fd);
        else if(appen == 2) send_append_question(client_fd); 
        else perror("Error about : ");
    }
    else send_question(client_fd); // 5

    // 6
    write(client_fd, buf, strlen(buf));

    close(client_fd);
    close(server_fd);

    return 0;
}

void check_cipher(int client_fd){
    char buf[Buf_len];
    char *version;

    sprintf(buf, "Check cipher version & Response about updating cipher version(yes or no) : ");
    write(client_fd, buf, strlen(buf));
    read(client_fd, buf, Buf_len); 
    if(); // cipher version 확인 작성하기++
    // file 전송하기++
}

int get_privacy_and_check_database(int client_fd){
    int count = 0, exist;
    char buf[Buf_len], privacy[512];

    while(count != 2){
        // 3
        sprintf(buf, "Input your front part of Resident registration number : ");
        write(client_fd, buf, strlen(buf));
        read(client_fd, buf, Buf_len); 
        if(strcmp(buf, "fail")) close(client_fd);

        // 4
        // 암호화되어 전송된 data 복호화 추가하기++ // data 형식은 client에서 확인

        // b
        exist = check_database(privacy);
        if(exist == 0) return 0; 
        else return 1; // client answer in past.
    }
}

void send_question(int client_fd){
    int ques_fd, respon_fd;
    char buf[Buf_len];

    ques_fd = fopen("question.txt", "r");
    respon_fd = fopen("respond.txt", "w"); // RESPOND FILE VERSION 관리 CODE++
    while(ques_fd != NULL){
        fgets(buf, Buf_len, ques_fd);  
        write(client_fd, buf, strlen(buf));
        read(client_fd, buf, Buf_len); // Question에 대한 답 얻기
        fputs(buf, respon_fd);
    }

    close(ques_fd);
    close(respon_fd);
}

void send_append_question(int client_fd){
    int app_fd, respon_fd;
    char buf[Buf_len];

    app_fd = fopen("append_question.txt", "r");
    respon_fd = fopen("app_respond.txt", "w"); // APPEND FILE VERSTION 관리 CODE++
    while(app_fd != NULL){
        fgets(buf, Buf_len, app_fd);  
        write(client_fd, buf, strlen(buf));
        read(client_fd, buf, Buf_len); // Question에 대한 답 얻기
        fputs(buf, respon_fd);
    }

    close(app_fd);
    close(respon_fd);
}

int check_database(char privacy[]){
    int database_r_fd, database_w_fd;
    char buf[Buf_len];

    database_r_fd = fopen("database.txt", "r");
    while(database_r_fd != NULL){
        fgets(buf, Buf_len, database_r_fd);
        if(strcmp(buf, privacy) == 0){
            close(database_r_fd);
            return 1;
        }
        else{
            close(database_r_fd);
            database_w_fd = fopen("database.txt", "a");
            fputs(privacy, database_w_fd);
            close(database_w_fd);
            return 0;
        }
    }
}