bool verify(weighted_digraph G, number d, claimed_tour S){
    if(S is a tour && the total weight of the edges in S is <= d)
        return ture;
    else
        return false;
}