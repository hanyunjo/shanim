#include <stdio.h>
#include <netdb.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <signal.h>
#include <sys/signal.h>

#define Buf_len 128
#define Listen_que 3

typedef struct {
    int remain_num;
} shm_mem;

int check_cipher(int client_fd);
char *get_privacy(int client_fd);
int check_database(char *hash);
void send_question(int client_fd, char *hash);
void send_append_question(int client_fd, char *hash);
void some_function(int client_fd, char *hash);
void close_child(int sema_id, shm_mem *shared);

void handler(int signo);

void getsem(int semid);
void returnsem(int semid);

int get_w_lock(int fd);
int get_r_lock(int fd);
int unlock(int fd);

int main(){
    char buf[Buf_len];
    int len, i, mess_len, check_num, chk_ci, count;
    int in_database;
    char *hash_value, *result;
    //socket
    int server_fd, client_fd;
    struct sockaddr_in server_addr, client_addr;
    //fork
    pid_t client_pid[100];
    for(i = 0; i < 100; i++) client_pid[i] = -1;
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
    //siganl
    sigset_t sigwait;
    struct sigaction act;

    // socket()++++++++++++++++
    if((server_fd = socket(PF_INET, SOCK_STREAM, 0)) == -1){
        printf("server can't open socket\n");
        exit(1);
    }

    // init server_addr, buffer to 0
    memset(&server_addr, 0, sizeof(server_addr));
    memset(&buf, 0, sizeof(buf));

    // set sever_addr
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    server_addr.sin_port = htons(15240);

    // bind()
    if(bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1){ 
        printf("failed bind func\n");
        exit(0);
    }

    // listen()
    if(listen(server_fd, Listen_que) == -1){
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
    shared->remain_num = 100;
    pid_num = 0;

    // semaphore++++++++++++++++
    if((sema_id = semget((key_t)9432, 1, IPC_CREAT|IPC_EXCL|0666)) != -1){
        arg.val = 1;
        if(semctl(sema_id, 0, SETVAL, arg) == -1){
            printf("failed semctl1 func\n");
            exit(1);
        }
    }
    else sema_id = semget((key_t)9432, 1, IPC_CREAT|0666);

    // signal setting
    sigemptyset(&act.sa_mask);
    //act.sa_flags = SA_RESETHAND;
    act.sa_flags = 0;
    act.sa_handler = handler;
    if(sigaction(SIGUSR1, &act, (struct sigaction *)NULL) < 0){
        printf("failed sigaction\n");
        exit(1);
    }
    
    count = 0;
    while(1){
        getsem(sema_id);
        if(shared->remain_num == 0){
            // signal wait
            returnsem(sema_id);
            while(1){
                sigsuspend(&sigwait);
                count++;
                if(count == 100)
                    break;
            }
            count = 0;
            printf("all client is close, so initiate\n");
            sleep(1);
            pid_num = 0;
            for(i = 0; i < 100; i++) 
                client_pid[i] = -1;
        }
        else{
            returnsem(sema_id);
            // accept
            len = sizeof(client_addr);
            while(1){
                client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &len);
                if(client_fd != -1)
                    break;
            }

            // semaphore 적용
            getsem(sema_id);
            shared->remain_num--;
            returnsem(sema_id);
        }

        // real_operating++++++++++++++++
        if(client_pid[pid_num] == -1){
            if((client_pid[pid_num] = fork()) == 0){ // child
                close(server_fd);
                printf("%d-client connect\n", pid_num);

                // 2
                chk_ci = check_cipher(client_fd);
                if(chk_ci == 0){
                    close(client_fd);
                    close_child(sema_id, shared);

                    printf("%d-client disconnect\n", pid_num);
                    exit(0);
                }

                // 3
                hash_value = get_privacy(client_fd);
                if(strncmp(hash_value, "fail", 4) == 0){
                    close(client_fd);
                    close_child(sema_id, shared);

                    printf("%d-client disconnect\n", pid_num);
                    exit(0);
                }

                // 4, b 
                in_database = check_database(hash_value);

                if(in_database == 1){
                    int appen;
                    sprintf(buf, "If you want new one, type 1 / If you want to append, type 2 : ");
                    write(client_fd, buf, strlen(buf));
                    i = read(client_fd, buf, Buf_len); 
                    buf[i] = '\0';
                    appen = atoi(buf);

                    // 5, c
                    if(appen == 1) 
                        send_question(client_fd, hash_value);
                    else if(appen == 2) 
                        send_append_question(client_fd, hash_value); 
                    else 
                        perror("Error about : ");
                }
                else 
                    send_question(client_fd, hash_value); // 5

                // 6
                some_function(client_fd, hash_value);

                close(client_fd);   
                close_child(sema_id, shared);

                printf("%d-client disconnect\n", pid_num);
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

int check_cipher(int client_fd){
    char buf[Buf_len], cmp[20];
    char *version = "wget https://www.openssl.org/source/openssl-1.1.1g.tar.gz";
    int len;

    sprintf(buf, "Check cipher version & Response about updating cipher version(yes or no) : ");
    write(client_fd, buf, strlen(buf));
    len = read(client_fd, buf, Buf_len); // take cipher version
    buf[len] = '\0';

    if(strncmp(buf, "fail", 4) == 0)
        return 0;

    if(strncmp(buf, "OpenSSL 1.1.1", 13) != 0){ // cipher version 확인
        write(client_fd, version, strlen(version)); //file 전송하기
    }
    else{
        sprintf(buf, "safe");
        write(client_fd, buf, strlen(buf));
    }
    return 1;
}

char *get_privacy(int client_fd){
    int len;
    char buf[Buf_len], *privacy;

    // 3
    sprintf(buf, "Input your front part of Resident registration number : ");
    write(client_fd, buf, strlen(buf));
    len = read(client_fd, buf, Buf_len);  // take privacy 
    buf[len] = '\0';

    if(strncmp(buf, "fail", 4) == 0) return "fail";
    
    privacy = malloc(sizeof(char)*65);
    strcpy(privacy, buf);

    return privacy;
}

int check_database(char *privacy){
    int database_r_fd, database_w_fd;
    char buf[Buf_len], tmp[100];

    database_r_fd = open("database.txt", O_CREAT|O_RDONLY, 0644);

    if(get_r_lock(database_r_fd) == -1){
        printf("file read_lock error\n");
        exit(1);
    }

    while(read(database_r_fd, buf, 65) > 60){ 
        buf[64] = '\0';
        if(strcmp(buf, privacy) == 0){ // exist
            close(database_r_fd);
            return 1;
        }
    }

    if(unlock(database_r_fd) == -1){
        printf("file unlock error\n");
        exit(1);
    }
    close(database_r_fd);

    // not exist
    database_w_fd = open("database.txt", O_WRONLY);

    if(get_w_lock(database_w_fd) == -1){
        printf("file write_lock error\n");
        exit(1);
    }

    lseek(database_w_fd, 0, SEEK_END);
    sprintf(tmp, "%s\n", privacy);
    write(database_w_fd, tmp, strlen(tmp));

    if(unlock(database_w_fd) == -1){
        printf("file unlock error\n");
        exit(1);
    }
    close(database_w_fd);

    return 0;
}

void send_question(int client_fd, char *hash){
    FILE* ques_fd;
    FILE* respon_fd;
    char buf[Buf_len], filename[50];
    int i, l, ques_num;

    ques_fd = fopen("question.txt", "r");
    sprintf(filename, "%s_respond.txt", hash);
    const char *tmp = filename;
    respon_fd = fopen(tmp, "w"); // RESPOND FILE VERSION 관리 CODE++
    l = 0;
    ques_num = 5;
    while(l < ques_num){
        fgets(buf, Buf_len, ques_fd);
        buf[strlen(buf) - 1] = '\0';
        write(client_fd, buf, strlen(buf));

        i = read(client_fd, buf, Buf_len); // Question에 대한 답 얻기
        buf[i] = '\0';
        fputs(buf, respon_fd);
        l++;
    }

    fclose(ques_fd);
    fclose(respon_fd);
}

void send_append_question(int client_fd, char *hash){
    FILE* app_fd;
    FILE* respon_fd;
    char buf[Buf_len], filename[50];
    int i, l, ques_num;

    app_fd = fopen("append_question.txt", "r");
    sprintf(filename, "%s_app_res.txt", hash);
    const char *tmp = filename;
    respon_fd = fopen(tmp, "a"); // APPEND FILE VERSTION 관리 CODE++
    l = 0;
    ques_num = 5;
    while(l < ques_num){
        fgets(buf, Buf_len, app_fd);
        buf[strlen(buf) - 1] = '\0';
        write(client_fd, buf, strlen(buf));
        
        i = read(client_fd, buf, Buf_len); // Question에 대한 답 얻기
        buf[i] = '\0';
        fputs(buf, respon_fd);
        l++;
    }

    fclose(app_fd);
    fclose(respon_fd);
}

void some_function(int client_fd, char *hash){
    FILE* app_fd;
    FILE* respon_fd;
    char result[50], filename[80];

    // hash's respond data
    sprintf(filename, "%s_respond.txt", hash);
    const char *tmp1 = filename;
    respon_fd = fopen(tmp1, "r");

    // hash's append_respond data
    sprintf(filename, "%s_app_res.txt", hash);
    const char *tmp2 = filename;
    app_fd = fopen(tmp2, "r");

    // result function do something by using data
    strcpy(result, "result : ~~~~~~~~");
    write(client_fd, result, strlen(result));

    fclose(respon_fd);
    if(app_fd != NULL) fclose(app_fd);
}

void close_child(int sema_id, shm_mem *shared){
    // semaphore 적용
    getsem(sema_id);
    shared->remain_num++;
    // signal send
    kill(getppid(), SIGUSR1);
    returnsem(sema_id);
}

//signal
void handler(int signo){
    printf("one client is close\n");
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

int get_w_lock(int fd){
    struct flock lock;
    lock.l_type = F_WRLCK;
    lock.l_start = 0;
    lock.l_whence = SEEK_SET;
    lock.l_len = 0;

    return fcntl(fd, F_SETLKW, &lock); // SETLKW는 lock을 얻을 때까지 blocking한다.
}

int get_r_lock(int fd){
    struct flock lock;
    lock.l_type = F_RDLCK;
    lock.l_start = 0;
    lock.l_whence = SEEK_SET;
    lock.l_len = 0;

    return fcntl(fd, F_SETLKW, &lock); // SETLKW는 lock을 얻을 때까지 blocking한다.
}

int unlock(int fd){
    struct flock lock;
    lock.l_type = F_UNLCK;
    lock.l_start = 0;
    lock.l_whence = SEEK_SET;
    lock.l_len = 0;

    return fcntl(fd, F_SETLK, &lock);
}