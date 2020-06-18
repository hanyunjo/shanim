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
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <openssl/bio.h>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/sha.h>

#define Buf_len 128
#define Listen_que 3

void check_cipher(int client_fd);
int get_privacy_and_check_database(int client_fd);
void send_question(int client_fd);
void send_append_question(int client_fd);
int check_database(char privacy[]);
void getsem(int semid);
void returnsem(int semid);

typedef struct {
    int remain_num;
} shm_mem;

int main(){
    char buf[Buf_len];
    int len, i, mess_len, check_num;
    int in_database;
    //socket
    struct sockaddr_in server_addr, client_addr, peer_addr;
    //fork
    pid_t client_pid[100] = { 0, };
    int pid_num;
    //IO
    int shm_id, sema_id;
    void *memory = (void *)0;
    shm_mem *shared;
    union semun{
        int                  val;
        struct   semid_ds   *buf;
        unsigned short int  *arrary;
    }  arg;
    //file
    int server_fd, client_fd;

    //Openssl
    SSL_CTX *ctx;
    SSL *ssl;
    X509 *cli_cert;
    char *str;
    char ssl_buf[4096];
    int err;
    SSL_METHOD *method;

    // Init ssl++++++++++++++++++++
    SSL_load_error_strings();
    SSLeay_add_ssl_algorithms();
    method = SSLv23_server_method();
    ctx = SSL_CTX_new(method);

    if(!ctx) {
        printf("failed to creat SSL_CTX\n");
        exit(2);
    }

    // set certiciate file
    if(SSL_CTX_use_certificate_file(ctx, CERTF, SSL_FILETYPE_PEM) <= 0) {      // 인증서를 파일로 부터 로딩할때 사용함.
        ERR_print_errors_fp(stderr);
        exit(1);
    }
   
    // set private key
    if(SSL_CTX_use_PrivateKey_file(ctx, KEYF, SSL_FILETYPE_PEM) <= 0) {
        ERR_print_errors_fp(stderr);
        exit(1);
    }
   
    // check private key
    if(!SSL_CTX_check_private_key(ctx)) {
        printf("Private key does not correct\n");
        exit(1);
    }

    // socket()++++++++++++++++
    if((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == -1){
        printf("server can't open socket\n");
        exit(1);
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

    // listen()
    if(listen(server_fd, Listen_que) != 0){
        printf("failed listen func\n");
        exit(0);
    }

    // shared_memory++++++++++++++++
    if((shm_id = shmget((key_t)1735, sizeof(shm_mem), IPC_CREAT|0666)) == -1){
        printf("failed shmget func\n");
        exit(1);
    }
    
    if((memory = shmat(shm_id, (void *)0, 0)) == (void *)-1){
        printf("failed shmat func\n");
        exit(1);
    }

    shared = (shm_mem *)memory;

    // semaphore++++++++++++++++
    if((sema_id = semget((key_t)9432, 1, IPC_CREAT|IPC_EXCL|0666)) != -1){
        arg.val = 1;
        if(semctl(sema_id, 0, SETVAL, arg) == -1){
            printf("failed semctl1 func\n");
            exit(1);
        }
    }
    else sema_id = semget((key_t)9432, 1, IPC_CREAT|0666);

    // operating++++++++++++++++
    pid_num = 0;
    while(1){
        len = sizeof(client_addr);
    
        // accept
        client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &len);
        if(client_fd < 0){
            printf("failed accept func\n");
            exit(0);
        }
        
        getsem(sema_id);
        if(shared->remain_num == 0){
            // sigwait
            // signal 받아서
            shared->remain_num++;
        }
        returnsem(sema_id);

        if(client_pid[pid_num] == 0){
            if((client_pid[pid_num] = fork()) == 0){ // child
                close(server_fd);

                // SSL setting
                if((ssl = SSL_new(ctx) == NULL){
                    printf("ssl_new error\n");
                    exit(1);
                }
                SSL_set_fd(ssl, client_fd);
                if((err = SSL_accept(ssl) == -1){
                    printf("SSL_accept error\n");
                    exit(1);
                } 

                cli_cert = SSL_get_peer_certificate(ssl);
                if(cli_cert == NULL){
                    printf("client doesn't have certificate\n");
                    exit(1);
                }

                // semaphore 적용
                getsem(sema_id);
                shared->remain_num--;
                returnsem(sema_id);
                
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
                SSL_free(ssl);
                SSL_CTX_free(ctx);
                exit(0);
            }
            pid_num++;
        }
    }

    close(server_fd);

    // destory shared_memory & semaphore++++++++++++++++
    if(shmdt(memory) == -1){
        printf("shmdt failed\n");
    }

    if(shmctl(shm_id, IPC_RMID, NULL) == -1){
        printf("shmctl failed\n");
    }

    if(semctl(sema_id, 0, IPC_RMID, 0) == -1){
        printf("semctl failed\n");
        exit(1);
    }

    return 0;
}

void check_cipher(int client_fd){
    char buf[Buf_len];
    char *version;
    int len;

    sprintf(buf, "Check cipher version & Response about updating cipher version(yes or no) : ");
    write(client_fd, buf, strlen(buf));
    read(client_fd, buf, Buf_len); 
    if(); // cipher version 확인 작성하기++
    // file 전송하기++
}

int get_privacy_and_check_database(int client_fd){
    int count = 0, exist, len;
    char buf[Buf_len], privacy[512];

    while(count != 2){
        // 3
        sprintf(buf, "Input your front part of Resident registration number : ");
        write(client_fd, buf, strlen(buf));
        len = SSL_read(client_fd, buf, Buf_len); 
        buf[len] = '\0';
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

// about semaphore
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
    struct sembuf rebuf;

    rebuf.sem_num = 0;
    rebuf.sem_op = 1;
    rebuf.sem_flg = SEM_UNDO;
    if((semop(semid, &rebuf, 1)) == -1){
        printf("failed return sem\n");
        exit(1);
    }
}