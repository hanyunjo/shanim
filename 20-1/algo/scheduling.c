#include <stdio.h>
#include <stdlib.h>

typedef struct sche{
        int job;
        int pro;
        int dead;
        struct sche *prev;
        struct sche *next;
}Sche;

int job[100][3];
  
void initlist(Sche *schelist);
void AddNode(Sche *schelist, int num);
Sche *findposi(Sche *schelist, Sche *insertnode);
int checkfeasible(Sche *schelist);
void deletnode(Sche *schelist, int i);
void printprofit(Sche *schelist);
void printjob(Sche *schelist);

int main(){
        int i;
        int realnum;

        // take input
        scanf("%d", &realnum);
        for(i = 0; i < realnum; i++) scanf("%d %d %d ", &job[i][0], &job[i][1], &job[i][2]);

        // init mainlist
        Sche *schelist = (Sche*)malloc(sizeof(Sche));
        initlist(schelist);

        // scheduling
        for(i = 0; i < realnum; i++){
                AddNode(schelist, i);
                if(checkfeasible(schelist) == 0) deletnode(schelist, i);
                //debug
                //printprofit(schelist);
                //printjob(schelist);
        }


        //print result
        printprofit(schelist);
        printjob(schelist);

        free(schelist);

        return 0;
}

void initlist(Sche *schelist){
        schelist->job = 0;
        schelist->pro = 0;
        schelist->dead = 0;
        schelist->prev = schelist;
        schelist->next = schelist;
}

void AddNode(Sche *schelist, int num){
        Sche* newnode = (Sche*)malloc(sizeof(Sche));
        newnode->job = job[num][0];
        newnode->dead = job[num][1];
        newnode->pro = job[num][2];

        // insert new node before curr node
        Sche* curr = findposi(schelist, newnode);
        curr->prev->next = newnode;
        newnode->prev = curr->prev;
        newnode->next = curr;
	curr->prev = newnode;
}

Sche *findposi(Sche *schelist, Sche *insertnode){
        int i;
        Sche *curr = schelist->next; // first node

        while(1){
                if(curr->job == 0) i = 0;
                else if(curr->dead < insertnode->dead) i = 1;
                else if(curr->dead == insertnode->dead){
                        if(curr->job < insertnode->job) i = 1;
                        else if(curr->job > insertnode->job) i = 0;
                }
                else if(curr->dead > insertnode->dead) i = 0;

                if(i == 0) break;
                else if(i == 1) curr = curr->next;
        }
        return curr;
}

int checkfeasible(Sche *schelist){
        int correct_dead = 1;
        Sche *curr = schelist->next;

        while(1){
                if(curr->job == 0) return 1;

                if(curr->dead < correct_dead) return 0;
                else curr = curr->next;

                correct_dead++;
        }
        return 1;
}


void deletnode(Sche *schelist, int deletnum){
        Sche *curr = schelist->next;
        //Sche *minnode = curr;
        //int min = curr->pro;
        int i = deletnum + 1;

        while(1){
                if((curr->job == 0) || (curr->next->job == 0)) break;

               /* if(minnode->pro >= curr->next->pro){
                        minnode = curr->next;
                        curr = curr->next;
                }
                else curr = curr->next;*/
                if(curr->job == i) break;
                else curr = curr->next;
        }
        /*
        minnode->prev->next = minnode->next;
        minnode->next->prev = minnode->prev;*/
        curr->prev->next = curr->next;
        curr->next->prev = curr->prev;
        free(curr);
}

void printprofit(Sche *schelist){
        int profit = 0;
        Sche *curr = schelist->next;

        while(1){
                if(curr->job != 0) profit += curr->pro;
                else break;

                curr = curr->next;
        }
        printf("%d", profit);
}

void printjob(Sche *schelist){
        Sche *curr = schelist->next;

        while(1){
                if(curr->job != 0) printf(" %d", curr->job);
                else break;

                curr = curr->next;
        }
        printf("\n");
}