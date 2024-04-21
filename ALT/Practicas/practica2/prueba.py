def langford(N):
    N2 = 2*N
    seq = [0]*N2
    def backtracking(num):
        if num<=0:
            yield "-".join(map(str, seq))
        else:
        # buscamos una posicion para situar una pareja num
            for i in range(0,len(seq)):
                if seq[i]==0 and i+1+num <= 2*N-1:
                    if seq[i+1+num] == 0:
                        seq[i]= num
                        seq[i+1+num] = num
                        yield from backtracking(num-1)
                        seq[i]= 0
                        seq[i+1+num] = 0

    if N%4 in (0,3):
        yield from backtracking(N)
N= 11
for solution in langford(N):
    print(solution)
