#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <termio.h>

#define Buf_len 128

int check_cipher(int sock, char buf[]);
int input_privacy();
int getch();

int main(){

    int sock, i, succ, err = 0;
    struct sockaddr_in server_addr;
    char buf[Buf_len+1], type_num[2];

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

    while((i = read(sock, buf, Buf_len)) > 0){
        buf[i] = '\0';
        if(strcmp(buf, "Check cipher version & Response about updating cipher version(yes or no) : ")){ // 2
            err = check_cipher(sock, buf);
            if(err == 0){
                sprintf(buf, "fail", NULL);
                write(sock, buf, strlen(buf));
                break;
            }
        }
        else if(strcmp(buf, "Input your front part of Resident registration number : ")){ // 3
            printf("%s", buf);
            err = input_privacy();
            if(err == 0){
                sprintf(buf, "fail", NULL);
                write(sock, buf, strlen(buf));
                break;
            }
        }
        else if(strcmp(buf, "If you want new one, type 1 / If you want to append, type 2 : ") == 0){ // b
            scanf("%d", type_num);
            write(sock, type_num, strlen(type_num));
        }
        else if(  ){ // 5 / c
            printf("%s\n", buf);
            scanf("%s", buf);
            write(sock, buf, strlen(buf));
        }
        else if(strcmp(buf, "result") == 0){ // 6
            read(sock, buf, Buf_len);
            printf("%s\n", buf);
            break;
        }
    }

    close(sock);

    return 0;
}

int check_cipher(int sock, char buf[]){
    char tmpbuf[Buf_len], ver[50];
    int err = 1;

    strcpy(tmpbuf, buf);
    
    printf("%s\n", tmpbuf);
    while(1){
        scanf("%s", tmpbuf);
        if(strcmp((int)tmpbuf[0], 'y') || strcmp((int)tmpbuf[0], 'Y')){
            if(strcmp((int)tmpbuf[1], 'e') || strcmp((int)tmpbuf[1], 'E')){
                if(strcmp((int)tmpbuf[2], 's') || strcmp((int)tmpbuf[2], 'S')) err = 1;
                else err = 0;
            }
            else err = 0;
        }
        else err = 0;

        if(strcmp((int)tmpbuf[0], 'n') || strcmp((int)tmpbuf[0], 'N')){
            if(strcmp((int)tmpbuf[1], 'o') || strcmp((int)tmpbuf[1], 'O')){
                printf("If you don't want to upadte cipher version\n");
                printf("Vulnerahble we can do nothing!!\n");
                printf("Please attempt to when you think updating cipher version.\n");
                return 0;
            }
        }

        if(err == 1){
            // cipher version 확인+++


            sprintf(buf, "%s yes", ver);
            write(sock, buf, strlen(buf));
            break;
        }
        printf("Please input only 'yes' or 'no'\n");
    }
    
    return 1;
}

int input_privacy(){
    char buf[Buf_len], privacy[14];
    int i, err = 2;

    for(i = 0; i < 13; i++){
        privacy[i] = getch();

        if((int)privacy[i] == 8) i--;
        else if((int)privacy[i] == 13){
            if(err == 2){
                printf("Please input 13 character.(remain 1 chance)\n");
                printf("Input : ");
                i = -1;
            }
            err--;
        }
        else if(strcmp(privacy[6], '1') || strcmp(privacy[6], '2') || strcmp(privacy[6], '3') || strcmp(privacy[6], '4')){
            if(err == 2){
                printf("First number of back part is incorrect.\n");
                printf("Input : ");
                i = -1;
            }
            err--;
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