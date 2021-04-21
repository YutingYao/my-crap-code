import pandas as pd
import numpy as np
import os

def bigDataProcessing(func, infile, outfile, check=True, chunkSitu=False, chunkSize=20000, **args):
    """
    [大数据处理框架]

    :param func: [传入的处理函数]
    :type func: [type]
    :param infile: [数据文件路径]
    :type infile: [type]
    :param outfile: [输出文件路径]
    :type outfile: [type]
    :param check: [是否为测试模式，测试模式仅处理前两个chunk], defaults to True
    :type check: bool, optional
    :param chunkSitu: [是否打印chunk情况], defaults to False
    :type chunkSitu: bool, optional
    :param chunkSize: [块大小], defaults to 20000
    :type chunkSize: int, optional
    """    
    funcName = func.__name__  # 函数名
    fileNameTmp = infile.split('/')[-1].split('.')
    fileName = outfile+fileNameTmp[0]+'_'+funcName+'.'+fileNameTmp[1]
    loop = True
    reader = pd.read_csv(infile, iterator=True,
                         nrows=None if check == False else 2*chunkSize)  # delete nrows to read all data
    i = 1
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            # 添加对于chunk的操作
            if chunk.empty:
                pass
            else:
                pass
                # 添加对于chunk的操作
                # 编辑输出
                chunk = func(chunk, **args)
                if not os.path.exists(fileName):
                    chunk.to_csv(fileName, index=False, mode='a')
                else:
                    chunk.to_csv(fileName, header=0, index=False, mode='a')
        except StopIteration:
            loop = False
        if chunkSitu == True:
            print(str(i)+' chunk done!')
        i += 1
        
def trans_data(data=None,fname='特征测试_newspeed_遮挡.csv',path='D:/DiplomaProject/output/paper/特征测试/',minN=2,vlist=[20,30,40,50,60,70,80],onehot_y=True,minmax_x=True,onehot_x=False,concols=['viaduct'],out2File=False,outpath='D:/DiplomaProject/output/paper/特征测试/',postfix='转换',fcols=[]):
    if data is None: 
        data=pd.read_csv(path+fname)
        data.maxspeed=data.maxspeed.astype(int)
    data.drop(['linkID','pNum'],axis=1,inplace=True)
    data=data[data.maxspeed.isin(vlist)]
    data=data.reset_index(drop=True)
    tmp=data['maxspeed']
    if onehot_y==True:
        tmp=pd.get_dummies(tmp)
    else:
        tmp=pd.DataFrame(tmp)
    df=data.drop('maxspeed',axis=1)
    if onehot_x==True:
        df=pd.get_dummies(df,columns=concols)
    if minmax_x==True:
        df=df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    title=list(tmp)
    ans=tmp.join(df)
    vdrop=['VTop'+str(i) for i in range(minN,24)]
    ans.drop(vdrop,axis=1,inplace=True)
    ans.dropna(inplace=True)
    ans.reset_index(drop=True,inplace=True)
    if fcols!=[]:
        title.extend(fcols)    
        ans=ans[title]
    if out2File==True and outpath!='':
        new_name=fname.replace('.csv','_%s.csv'%postfix)
        ans.to_csv(outpath+new_name,index=0)
    else:
        return ans
