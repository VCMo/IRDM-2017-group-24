
# coding: utf-8

# ## Learning to Rank Evaluation Metrics
# 

# ## Confusion matrix¶

# In[26]:

def plot_confusion_matrix(cm, classes,dataset,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(dataset + ' : ' + title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(dataset + " : Normalized confusion matrix" )
    else:
        print(dataset + ' : Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
def CM_metric(target,predict,name):   
    cnf_matrix = confusion_matrix(target,predict)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes='01234',dataset = name, normalize=False,
                          title='without normalization')
    

CM_metric(y_test,y_test_predict,'Testset') 
precision, recall, fscore, support = precision_recall_fscore_support(y_test,y_test_predict)
print('fscore: {}'.format(fscore))


#   ##  Normalized Discounted Cumulative Gain （NDCG）

# In[27]:

# Normalized Discounted Cumulative Gain

def NDCG_metric(dataset,predict,target,position):
    Result_predict = collections.defaultdict(int)
    Result_target = collections.defaultdict(int)
    Result_normalization = collections.defaultdict(int)

    index = 0
    
    qid = qid_old = dataset['qid'][0]
    for i in range(len(dataset)): 
        qid = dataset['qid'][i]
        
        if qid != qid_old:
            index = 0
        
        if index < position:
            Result_predict[qid] += (np.power(2, predict[i])-1)/(np.log2(index+2))
            Result_target[qid] += (np.power(2,  target[i])-1)/(np.log2(index+2))
            index += 1
            qid_old = qid
       
    for key in Result_predict.keys():
        Result_normalization[key] = Result_predict[key]/Result_target[key] if Result_target[key] > 0 else 0
     
    data = [Result_predict.values(),Result_target.values(),Result_normalization.values()]
    df = pd.DataFrame(data,index=['predict','target','NDCG'],columns=Result_predict.keys())
    return df.T

order = 1
get_ipython().magic('time NDCG_test = NDCG_metric(testset,y_test_predict,y_test,order)')
score = np.mean(NDCG_test['NDCG'])
print ('The NDCG socre on test set : '  + str(score))
order = 3
get_ipython().magic('time NDCG_test = NDCG_metric(testset,y_test_predict,y_test,order)')
score = np.mean(NDCG_test['NDCG'])
print ('The NDCG socre on test set : '  + str(score))
order = 10
get_ipython().magic('time NDCG_test = NDCG_metric(testset,y_test_predict,y_test,order)')
score = np.mean(NDCG_test['NDCG'])
print ('The NDCG socre on test set : '  + str(score))


# ## Mean Average Precision (MAP)

# In[28]:

def AP_metric(dataset,predict,target):

    Result_K = collections.defaultdict(int)
    Result_APK = collections.defaultdict(int)
    Result_APN = collections.defaultdict(int)
    
    index = 0
    
    qid = qid_old = dataset['qid'][0]
    for i in range(len(dataset)): 
        qid = dataset['qid'][i]
        
        if qid != qid_old:
            index = 0
         
        Result_K[qid] += 1 if predict[i] == target[i] else 0
        Result_APK[qid] += Result_K[qid]/(index+1) if predict[i] == target[i] else 0
        index += 1
        qid_old = qid 
      
    for key in Result_APK.keys():
        Result_APN[key] = Result_APK[key]/Result_K[key] if Result_K[key] > 0 else 0

    data = [Result_APN.values(),Result_APK.values(),Result_K.values()]
    df = pd.DataFrame(data,index=['APN','APK','K'],columns=Result_APN.keys())
    return df.T

get_ipython().magic('time MAP_valid = AP_metric(testset,y_test_predict,y_test)')
score = np.mean(MAP_valid['APN'])
print ('The MAP socre on Test set : ' + str(score))


# In[ ]:



