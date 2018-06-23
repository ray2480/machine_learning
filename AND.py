def AND(x1, x2): 
    x = np.array([x1,x2]) 
    w = np.array([0.5,0.5]) 
    b = -0.7 
    theta = 0 
    tmp = np.sum(w*x) + b 
    if tmp <= theta: 
        return 0 
    else: 
        return 1 
        
def __name__ == '__main__':        
  binary = np.array([[0,0],[1,0],[0,1],[1,1]])
  for ipt in binary:
      y = AND(ipt[0],ipt[1])
      print(str(xs) + " -> " + str(y))
