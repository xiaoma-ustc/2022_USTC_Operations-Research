import math
import numpy as np
import time


class Solution:
    def minDistance(self, houses: list[int], k: int) -> int:
        n = len(houses)
        houses.sort()

        medsum = [[0] * n for _ in range(n)]
        
        for i in range(n - 2, -1, -1):
            for j in range(i + 1, n):
                medsum[i][j] = medsum[i + 1][j - 1] + houses[j] - houses[i]
        
        BIG = 10**9
        
        f = [[BIG] * (k + 1) for _ in range(n)]
        
        for i in range(n):
            f[i][1] = medsum[0][i]
            for j in range(2, min(k, i + 1) + 1):
                for i0 in range(i):
                    if f[i0][j - 1] != BIG:
                        f[i][j] = min(f[i][j], f[i0][j - 1] + medsum[i0 + 1][i])
        
        return f[n - 1][k]

if __name__ == "__main__":
    
    k = input("Please input the number of Mailbox : ")
    k = int(k)
    
    List = input("Please input the site of users : ").split()
    List = list(map(int, List))
    
    solution = Solution()
    
    tic = time.time()
    print("The smallest dis is : ", solution.minDistance(List, k))
    toc = time.time()
    
    print('That tooks %fs' % (toc - tic))

