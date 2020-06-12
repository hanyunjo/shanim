#include <stdio.h>

#define min(a, b) ((a < b) ? a : b)
#define que_size 1024

int num_vertex, result[17];
int edge[16][16] ={ 0, };

typedef struct {
    int path[17];
    int level;
    int bound;
} node;

typedef struct{
    int size;
    node arr[que_size];
} pq;

void input_func();
int travel(pq *que);
void print_func();
int length_func(node no);
int bound_func(node no);

void add(pq *que, node *no); 
node delete_root(pq *que); 
int empty(pq *que); 

int main(){
    int i, minlen;
    pq pri_que;
    pri_que.size = 0;

    input_func();

    minlen = travel(&pri_que);

    printf("%d\n", minlen);
    print_func();

    return 0;
}

void input_func(){
    int i, j;

    scanf("%d", &num_vertex);

    for(i = 1; i <= num_vertex; i++){
        for(j = 1; j <= num_vertex; j++){
            scanf("%d", &edge[i][j]);
        }
    }
}

int travel(pq *que){
    int i, j, minlen, last, inpath[16] = { 0, }, tmp_num = 1;
    node tmp, u, root;

    for(i = 0; i < 16; i++) tmp.path[i] = 0;
    tmp.path[1] = 1;
    tmp.level = 1;
    tmp.bound = bound_func(tmp);
    minlen = 1000;
    add(que, &tmp);

    while(!empty(que)){
        root = delete_root(que);

        if(root.bound < minlen){
            u.level = root.level + 1;
            for(i = 1; i <= root.level; i++){
                if(root.path[i] != 0) inpath[root.path[i]] = 1;
            }
            //printf("%d %d %d %d %d\n",root.path[1],root.path[2],root.path[3],root.path[4], root.path[5]);
            //printf("%d %d %d %d %d\n", inpath[1], inpath[2], inpath[3], inpath[4], inpath[5]);
                
            for(i = 2; i <= num_vertex; i++){
                if(inpath[i] == 1) continue;

                for(j = 0; j < 17; j++) u.path[j] = root.path[j];
                u.path[u.level] = i;
                
                if(u.level == num_vertex - 1){
                    for(j = 1; j <= num_vertex; j++){
                       if(inpath[j] == 0) last = j;
                    }
                    u.path[num_vertex] = last;
                    u.path[num_vertex + 1] = 1;
                    if(length_func(u) < minlen){
                        minlen = length_func(u);
                        for(j = 0; j <= num_vertex + 1; j++) result[j] = u.path[j];
                    }
                }
                else{
                    u.bound = bound_func(u);
                    //printf("%d %d %d\n", u.path[1], u.path[2], u.bound);
                    if(u.bound < minlen) add(que, &u);
                }
            }
        }/*
        printf("%d %d %d / ", tmp_num++, root.level, root.bound);
        for(i = 0; i < que->size; i++) printf("%d ", que->arr[i].bound);
        printf("\n");*/
        for(i = 2; i < 16; i++) inpath[i] = 0;
    }
    return minlen;
}

void print_func(){
    int i;
    for(i = 1; i <= num_vertex + 1; i++) printf("%d ", result[i]);
    printf("\n");
}

int length_func(node v){
    int j, len = 0;

    for(j = 1; j <= num_vertex; j++)
        len += edge[v.path[j]][v.path[j+1]];

    return len;
}

int bound_func(node v){
    int i, j, result = 0, real_min;
    int nonpath[16]; // path에 존재하지 않는다.
    for(j = 1; j < 16; j++) nonpath[j] = 1;

    // path에 존재하는 결정된 edge값 result에 더하기
    for(j = 1; j < v.level; j++){ 
        result += edge[v.path[j]][v.path[j+1]];
    }

    // path에 존재하는 node(마지막 node는 제외), nonpath에서 제거
    for(j = 1; j < v.level; j++) nonpath[v.path[j]] = 0;

    // j : 각 minimum 구해야하는 시작 node
    for(j = 1; j <= num_vertex; j++){
        real_min = 1000;

        if(nonpath[j] == 0) continue;
        else if(j == v.path[v.level]){ // path의 마지막 노드
            for(i = 1; i <= num_vertex; i++){
                if((nonpath[i] != 0) && (j != i))
                    real_min = min(real_min, edge[j][i]);
            }
        }
        else{ // path에 존재하지 않는 노드
            real_min = edge[j][1];
            for(i = 1; i <= num_vertex; i++){
                if((nonpath[i] != 0) && (j != i) && (v.path[v.level] != i))
                    real_min = min(real_min, edge[j][i]);
            }
        }
        result += real_min;
    }

    return result;
}

// priority queue
void swap(node *a, node *b){
    node tmp = *a;
    *a = *b;
    *b = tmp;
}

void add(pq *que, node *no){
    int cur, pare, i;

    que->arr[que->size].level = no->level;
    que->arr[que->size].bound = no->bound;
    for(i = 0; i < 17; i++) que->arr[que->size].path[i] = no->path[i];

    cur = que->size;
    pare = (que->size - 1) / 2;

    while(cur > 0 && que->arr[cur].bound < que->arr[pare].bound){
        swap(&que->arr[cur], &que->arr[pare]);
        cur = pare;
        pare = (pare - 1) / 2;
    }
    que->size++;
}

node delete_root(pq *que){
    int i, curr, left, right, minnode;
    node ret;

    // return node set;
    ret.level = que->arr[0].level;
    ret.bound = que->arr[0].bound;
    for(i = 0; i < 17; i++) ret.path[i] = que->arr[0].path[i];
    que->size--;

    // min_node set at root
    que->arr[0].level = que->arr[que->size].level;
    que->arr[0].bound= que->arr[que->size].bound;
    for(i = 0; i < 17; i++) que->arr[0].path[i] = que->arr[que->size].path[i];

    curr = 0;
    left = curr * 2 + 1;
    right = curr * 2 + 2;
    minnode = curr;

    while(left < que->size){
        if(que->arr[minnode].bound > que->arr[left].bound) 
            minnode = left;
        
        if(right < que->size && que->arr[minnode].bound > que->arr[right].bound) 
            minnode = right;
        
        if(minnode == curr) break;
        else{
            swap(&que->arr[curr], &que->arr[minnode]);
            curr = minnode;
            left = curr * 2 + 1;
            right = curr * 2 + 2;
        }
    }

    return ret;
}

int empty(pq *que){
    if(que->size == 0) return 1;
    else return 0;
}