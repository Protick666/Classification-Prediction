


import numpy
import math

#a=numpy.zeros((2,2,2))

#print(a[0][0][2])

#num_of_layer neurons_in_layer temp
#weight [layer] [ith] [jth of prev]

def f_func(val):
    e=math.exp(-val)
    e=e+1
    e=1/e;
    return e

def forward():
    global temp
    global y_val
    for l in range (2,num_of_layer+1):
        for i in range (1,neurons_in_layer[l]+1):
            for j in range (1,neurons_in_layer[l-1]+2):
                temp[l][i]+=y_val[l-1][j]*weight[l][i][j]
            y_val[l][i]=f_func(temp[l][i])
    return

def back_propagate():
    global weight
    error=float(0.0)
    for i in range(1,neurons_in_layer[num_of_layer]+1):
        error = error + .5*(actual_ans[i]-y_val[num_of_layer][i])*(actual_ans[i]-y_val[num_of_layer][i])

    differntial_val = numpy.zeros((num_of_layer + 3, a + 5))
    for i in range(1,neurons_in_layer[num_of_layer]+1):
        differntial_val[num_of_layer][i]=-(actual_ans[i]-y_val[num_of_layer][i])

    #x->y edge
    for l in range (num_of_layer,1,-1):
        for y in range(1,neurons_in_layer[l]+1):
            for x in  range(1,neurons_in_layer[l-1]+1):
                first_element=differntial_val[l][y]
                second_element=y_val[l][y]*(1-y_val[l][y])
                third_element=y_val[l-1][x]
                weight[l][y][x] = weight[l][y][x]-learning_rate*first_element*second_element*third_element
            ff = differntial_val[l][y]
            ss = y_val[l][y]*(1 - y_val[l][y])
            tt = 1
            weight[l][y][neurons_in_layer[l-1]+1] = weight[l][y][neurons_in_layer[l-1]+1] - learning_rate*ff*ss*tt
        for i in range(1,neurons_in_layer[l-1]+1):
            for j in range(1,neurons_in_layer[l]+1):
                differntial_val[l-1][i]+= weight[l][j][i]*y_val[l][j]*(1-y_val[l][j])*differntial_val[l][j]

    return


b = open('troy.txt').read()
b = [item.split() for item in b.split('\n')[:-1]]

test=numpy.array(b,dtype=float)
numpy.random.shuffle(test)



a = open('hello.txt').read()
a = [item.split() for item in a.split('\n')[:-1]]
train=numpy.array(a,dtype=float)
numpy.random.shuffle(train)








num_of_layer=int(input("no of layers"))
learning_rate=float(input("learning rate"))

neurons_in_layer=numpy.zeros((num_of_layer+3), dtype=int)

for i in range(1,num_of_layer+1):
    neurons_in_layer[i] = float(input("put no of neurons in "+str(i)+"th layer"))

a=int(0)

for i in neurons_in_layer:
    a=max(a,(i))

weight=numpy.ones((num_of_layer+3,a+a,a+a))

tot_example=int(len(train))

input_feat=numpy.zeros((tot_example+2 , neurons_in_layer[1]+2))
print(input_feat.shape)
input_ans=numpy.zeros((tot_example+2),dtype=int)

for i in range(1,tot_example+1):
    for j in range(1,neurons_in_layer[1]+1):
        input_feat[i][j]=float(train[i-1][j-1])
    input_ans[i]=int(train[i-1][neurons_in_layer[1]])


temp=numpy.zeros((num_of_layer+3,a+5))
y_val=numpy.zeros((num_of_layer+3,a+5))
actual_ans=numpy.zeros(a+a)
tm=0
for i in range(1,tot_example+1):
    for j in range(1,neurons_in_layer[1]+1):
        temp[1][j]=input_feat[i][j]
        y_val[1][j] = input_feat[i][j]
    for j in range(1, num_of_layer+1):
        temp[j][neurons_in_layer[j]+1] = 1
        y_val[j][neurons_in_layer[j] + 1] = 1
    forward()
    tm=tm+1
    #for j in range(1, neurons_in_layer[num_of_layer] + 1):
        #print(temp[num_of_layer][j])

    actual_ans[input_ans[i]] = 1
    # print(input_ans[i])


    back_propagate()
    for j in range(1, num_of_layer + 1):
        for k in range(1, neurons_in_layer[j] + 1):
            temp[j][k] = 0
    actual_ans[input_ans[i]] = 0
    #print(y_val[3][1],y_val[3][2],y_val[3][3])








tot_example=int(len(test))
for i in range(1,tot_example+1):
    for j in range(1,neurons_in_layer[1]+1):
        input_feat[i][j]=float(test[i-1][j-1])
    input_ans[i]=int(test[i-1][neurons_in_layer[1]])


correct=int(0)
actual_ans=numpy.zeros(a+a)
tota=0
for i in range(1,tot_example+1):
    tota=tota+1;
    for j in range(1,neurons_in_layer[1]+1):
        temp[1][j]=input_feat[i][j]
        y_val[1][j] = input_feat[i][j]
    for j in range(1, num_of_layer+1):
        temp[j][neurons_in_layer[j]+1] = 1
        y_val[j][neurons_in_layer[j] + 1] = 1
    forward()

    for j in range(1, num_of_layer + 1):
        for k in range(1, neurons_in_layer[j] + 2):
            temp[j][k] = 0
    actual_ans[input_ans[i]] = 1
    predict=1
    val=y_val[num_of_layer][1]
    for j in range(1,neurons_in_layer[num_of_layer]+1):
        if(val<y_val[num_of_layer][j]):
            #print(val,y_val[num_of_layer][j])
            val=y_val[num_of_layer][j]
            predict=j
    if predict == input_ans[i]:
        correct=correct+1
    #print( predict,input_ans[i])

    actual_ans[input_ans[i]] = 0

print("correct predictions",correct,"out of ",tota," samples")
print("precision ",correct/tota)


















