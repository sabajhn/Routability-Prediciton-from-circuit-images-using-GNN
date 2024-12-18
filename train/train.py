import time
import torch
def train1(train1_indc,dataloader1,gf1,GAT,criterion,optimizer, scheduler):
    GAT.train()

    for i in range(len(train1_indc)):  # Iterate in batches over the train1ing dataset.
        optimizer.zero_grad()  # Clear gradients.
        id = train1_indc[i]
        data1 = dataloader1[id]
         
        #  gf1 = gf1[id]
        data1.y = torch.Tensor(data1.y)
        out = GAT(data1.x.float().T, data1.edge_index,data1.edge_attr, data1.batch, gf1[id])  

        # print("criterion(out.float(), data1.y.float())", out.float(), data1.y.float())
        # print(out.shape,data1.y.shape)
        print(out.float(), data1.y.float())
        loss = criterion(out.float(), data1.y.float()) # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        scheduler.step(loss.item())

def test1(indc1,dataloader1, gf1,GAT, batch, graph_features):
    print("test")
    t1=time.time()
    GAT.eval()

    correct = 0.0001
    times=[]
    TN = 0.0001
    FP = 0.0001
    FN = 0.0001
    TP = 0.0001
    tresh = 0.5
    for i in range(len(indc1)):  # Iterate in batches over the training dataset.
        id = indc1[i]
        data1 = batch[id]
        out = GAT(data1.x.float().T, data1.edge_index,data1.edge_attr, data1.batch, gf1[id])  
        #  print(gf1[id])
        print(out, data1.y, graph_features[id][0],graph_features[id][-10], id)
        if(out[0] >= tresh and data1.y[0] == 1):
            TP += 1
        if(out[0] < tresh and data1.y[0] == 1):
            FN += 1
        if(out[0] >= tresh and data1.y[0] == 0):
            FP += 1
        if(out[0] < tresh and data1.y[0] == 0):
            TN += 1
            
        correct += int(((out>0.5) == data1.y).sum())  # Check against ground-truth labels.
        t2=time.time()
    #  print(t2-t1)
    #  print(len(data1.x))
    #  times.append[t2-t1]
    accuracy = int( (TP + TN) / (TP + TN + FP + FN) * 1000)
    Prec = int( (TP) / (TP + FP) * 1000)
    Rec = int( (TP) / (TP + FN) * 1000)
    Spec = int( (TN) / (TN + FP) * 1000)
    print("zeros = ",TN + FP)
    print("ones = ",TP + FN)
    return correct / len(indc1), accuracy / 10. , Prec / 10. , Rec / 10. , Spec / 10. ,t2-t1  # Derive ratio of correct predictions.

