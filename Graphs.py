import random
#number of vertices in the graph
n=5
# creates random digraph in an adjacency matrix with a size of n
def randomDigraphMatrix():
    m=[[random.randint(0,1) for i in range(n)] for j in range (n)]
    return m
# creates non-directed graph in an adjacency matrix with a size of n 
def randomGraphMatrix():#there are possible loops
    m=[[random.randint(0,1) for i in range(n)] for j in range (n)]#треба придумати щось краще
    for i in range (n):
        for j in range (n):
            if m[i][j]!=m[j][i]:
                m[i][j]=m[j][i]
            if (i==j) & (m[i][j]==1):# to avoid loops     
                m[i][j]=0
    return m
# makes an adjacency list out of adjacency matrix
def adjlistFromMatrix(m):
    al=[[] for i in range (n)]
    for i in range (n):
        for j in range (n):
            if m[i][j]==1:
                al[i].append(j)
    return al
# makes an adjacency matrix out of adjacency list
def adjmatrixFromList(l):
    am=[[0 for i in range(n)] for j in range (n)]
    for i in range(n):
        for j in range(n):
            if l[i].count(j)==0:
                am[i][j]=0
            else:
                am[i][j]=1
    return am
# prints an adjacency matrix
def showAdjm(m):
    for i in range(n):
        print(m[i])
    print()
# prints an adjacency list
def showAdjl(l):
    for i in range (n):    
        print(l[i])
    print()
# adds an inputted edge to adjacency matrix
def addEdgeToMatrix():
    edge=[-1,-1]
    for i in range (2):
        edge[i]=int(input())
        if edge[i]>=n or edge[i]<0:
            print('Error')
            break
    print()
    adjm[edge[0]][edge[1]]=1
    adjm[edge[1]][edge[0]]=1
#----------------------------------------------------------------------------------------------------------------------------------------------
# Depth for search algo
visited=[False for i in range (n)]
def dfs(at):
    global count
    if visited[at]:
        return
    visited[at]=True
    neighbours=adjl[at]
    components[at]=count#for a count components line
    for next in neighbours:
        dfs(next)
#count components algo
count=0
components=[0 for i in range(n)]
def countComponents():
    global count
    global components
    components=[0 for i in range(n)]
    count=0
    for i in range(n):
        if not visited[i]:
            count+=1
            dfs(i)
    return(components)
#finding bridges
def findBridges():#неоптимізований  
    global visited
    global components
    global count
    bridges=[]
    visited=[False for i in range (n)]
    checkBridge=countComponents()
    visited=[False for i in range (n)]
    for i in range (n):
        for j in range (len(adjl[i])):
            prBridge=adjl[i].pop(j)
            if (checkBridge!=countComponents()) and (not [i,prBridge] in bridges) and (not [prBridge,i] in bridges) and ([i,prBridge]!=[i,i]):
                bridges.append([i,prBridge])
            adjl[i].insert(j,prBridge)
            visited=[False for i in range (n)]
    return(bridges)
#--------------------------------------------------------------------------------------------------------------------------------------------    
#Breath for search algo
def bfs(s,e):
    prev=solve(s)
    return reconstructPath(s,e,prev)
#checking if list has something despite none
def nonEmpty(q):
    count=0
    k=len(q)
    for i in range(k):
        if q[i]!=None:
            count+=1
    if count==0:
        return False
    else:
        return True
#additional function to bfs() which gives us a list in which we have roots for each vertex
def solve(s):
    q=[]
    q.append(s)
    attended=[False for i in range (n)]
    attended[s]=True
    prev=[None for i in range(n)]   
    while nonEmpty(q)==True:
        node=q.pop(0)
        neighbours=adjl[node]
        for next in neighbours:
            if attended[next]==False:
                q.append(next)
                attended[next]=True
                prev[next]=node
    return prev
#additional function to bfs() which gives us a path from s vertex to e 
def reconstructPath(s,e,prev):
    path=[]
    at=e
    while at!=None:
        path.append(at)
        at=prev[at]
    path.reverse()
    if path[0]==s:
        return path
    return []
#direction vectors algos
dr=[-1,1,0,0]
dc=[0,0,1,-1]
r=1
c=2
R=5
C=7
curcell=[r,c]
for i in range (4):
    rr=r+dr[i]
    cc=c+dc[i]
    if rr<0 or cc<0:
        continue
    if rr>=R or cc>=C:
        continue



startNode=0

adjm=randomGraphMatrix()
showAdjm(adjm)
adjl=adjlistFromMatrix(adjm)
showAdjl(adjl)
print(countComponents())
print(findBridges())
print(bfs(0,2))