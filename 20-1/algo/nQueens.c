#include <stdio.h>

void queens(int i);
void print_fir_sol();
int promising(int i);
int abs(int i);

int num, col[13] = { 0, }, fir_sol[13] = { 0, }; 
int tree_node = 1, pro_node = 1, sol_node = 0;

int main(){
    int i;

    scanf("%d", &num);

    for(i = 1; i <= num ; i++){
        col[1] = i;
        queens(1);
    }

    if(sol_node == 0) printf("No solution\n");
    else{
        print_fir_sol();
        printf("%d %d %d\n",tree_node, pro_node, sol_node);    
    }
    
    return 0;
}

void queens(int i){
    int j = 0, tmp;

    if(promising(i)){
        pro_node++;
        if(i == num){
            sol_node++;
            if(sol_node == 1){
                for(tmp = 1; tmp <= num; tmp++) fir_sol[tmp] = col[tmp];
            }
        }
        else{
            for(j = 1; j <= num; j++){
                col[i + 1] = j;
                queens(i + 1);
            }
        }
    }
    tree_node++;
}

void print_fir_sol(){
    int i;
    for(i = 1; i <= num; i++) printf("%d ", fir_sol[i]);
}

int promising(int i){
    int j, pro;

    j = 1;
    pro = 1;
    while(j < i && pro){
        if( (col[i] == col[j]) || (abs(col[i] - col[j]) == (i - j)) ) pro = 0;
        j++;
    }
    
    return pro;
}

int abs(int i){
    if(i >= 0 ) return i;
    else{
        i = -1*i;
        return i;
    } 

}