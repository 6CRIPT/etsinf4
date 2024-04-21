class Solution(object):
    def totalNQueens(self, n):
        """
        :type n: int
        :rtype: int
        """
        self.sol= 0
        def isComplete(s, n):
            return len(s) == n
    
        def is_promising(s, row):
            return all(row != s[i] and len(s)-i != abs(row-s[i]) for i in range(len(s)))
        
        def backtracking(n, s):
            if isComplete(s, n):
                self.sol +=1
            else:
                for row in range(n):
                    if is_promising(s, row):
                        s.append(row)
                        yield from backtracking(n, s)
                        s.pop()
        s= []                
        backtracking(n,s)
        return self.sol