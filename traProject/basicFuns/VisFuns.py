import gc

import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import traProject.utils as tu
from shapely.wkt import loads
from scipy.stats import norm
from tqdm import  tqdm
import warnings
warnings.filterwarnings("ignore")

def creatGif(image_list, gif_name, duration=2):
    """
    [生成gif文件，原始图像仅仅支持png格式]

    :param image_list: [图像路径列表]
    :type image_list: [type]
    :param gif_name: [生成的gif路径及文件名]
    :type gif_name: [type]
    :param duration: [gif图像时间间隔 单位s], defaults to 2
    :type duration: int, optional
    """
    frames = []

    # 利用方法append把图片挨个存进列表

    for image_name in image_list:
        frames.append(imageio.imread(image_name))

    # 保存为gif格式的图
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    
def visV(filespath,outFile,speedName='speed',fitpara=None,beginIndex=0):
    files=tu.getFilesBySize(filespath)
    # fig, axs = plt.subplots(4, 6, figsize=[16, 12], dpi=300)
    for idx,f in enumerate(tqdm(files)):
        if idx<beginIndex:
            continue
        fig, axs = plt.subplots(4, 6, figsize=[16, 12], dpi=300)
        titleName=f.split('/')[-1].split('.')[0]
        fig.suptitle(titleName)
        filename = titleName
        A=pd.read_csv(f)
        for i in range(4):
            for j in range(6):
                t = i*6+j
                pdata = A[(A.hour == t)]
                if pdata.empty:
                    continue
                else:
                    pdata = pdata[speedName]
                pillar = None
                if not pdata.empty:
                    pillar = int(abs(pdata.max()-pdata.min()))//2
                    if pillar < 1:
                        pillar = 1
                axs[i, j].set_title('Time='+str(t))
                try:
                    sns.distplot(pdata, fit=fitpara,
                            norm_hist=True, ax=axs[i, j], color='g', bins=pillar)
                except:
                    pass        
        fig.tight_layout(pad=3)
        plt.savefig(tu.pathCheck(outFile)+filename+'_路段速度直方图.png')
        plt.clf()
        plt.close()
        
# filePath = 'D:/DiplomaProject/output/tmp/轨迹_按link_id/'
# outFile=tu.pathCheck('D:/DiplomaProject/output/pic/WUHAN/速度分布直方图/')
# visV(filePath,outFile,beginIndex=887)

# def visV_bug(A, titleName, outFile, speedName='speed', fitpara=None):
#     filename = titleName
#     # weekList = ['Monday', 'Tuesday', 'Wednesday',
#     #             'Thursday', 'Friday', 'Saturday', 'Sunday']
#     # if 'weekday' in A.columns:
#     #     titleName = titleName+'_'+weekList[A.weekday.unique()[0]]
#     #     filename = 'weekday='+str(A.weekday.unique()[0])+'_'+titleName
#     # if all(item in A.columns for item in ['day', 'month']):
#     #     titleName = titleName+'_' + \
#     #         str(A.month.unique()[0])+'_'+str(A.day.unique()[0]).rjust(2, '0')
#     #     filename = 'date=' + \
#     #         str(A.month.unique()[0])+'_'+str(A.day.unique()
#     #                                          [0]).rjust(2, '0')+'_'+titleName
#     fig, axs = plt.subplots(4, 6, figsize=[16, 12], dpi=300)
#     fig.suptitle(titleName)
#     for i in range(4):
#         for j in range(6):
#             t = i*6+j
#             pdata = A[(A.hour == t)]
#     #         pdata=pdata[(A.hour == t)&(A[speedName] >= vlimt[0])
#     #                   & (A[speedName]<= vlimt[1])]
#             if pdata.empty:
#                 continue
#             else:
#                 pdata = pdata[speedName]
#             pillar = None
#             if not pdata.empty:
#                 pillar = int(abs(pdata.max()-pdata.min()))//2
#                 if pillar < 1:
#                     pillar = 1
#             sns.distplot(pdata, fit=fitpara,
#                          norm_hist=True, ax=axs[i, j], color='g', bins=pillar)
#             axs[i, j].set_title('Time='+str(t))

#     fig.tight_layout(pad=3)
#     plt.savefig(tu.pathCheck(outFile)+filename+'_路段速度直方图.png')
#     plt.cla()
#     plt.clf()
#     plt.close(fig)
#     del fig, axs
#     gc.collect()

def visV_24h(A,speedName='newspeed',fitpara=None,outFile='',fileName=''):
    fig, axs = plt.subplots(4, 6, figsize=[16, 12], dpi=300)
    for i in range(4):
        for j in range(6):
            t = i*6+j
            pdata = A[(A.hour == t)]
            if pdata.empty:
                continue
            else:
                pdata = pdata[speedName]
            pillar = None
            if not pdata.empty:
                pillar = int(abs(pdata.max()-pdata.min()))//2
                if pillar < 1:
                    pillar = 1
            axs[i, j].set_title('Hour='+str(t))
            try:
                sns.distplot(pdata, fit=fitpara,
                        norm_hist=False, ax=axs[i, j], color='g', bins=pillar)
            except:
                pass        
    fig.tight_layout(pad=3)
    if outFile!='':
        plt.savefig(tu.pathCheck(outFile)+'%s_路段速度直方图'%fileName)
        
def visTra1(p,roadpath,outpath,style='-o',speedName='speed',drawV=[],drawR=[]):
    p=p.replace('\\','/')
    road_all=pd.read_csv(roadpath)
    psplit=p.split('/')
    rid=psplit[-1].split('.csv')[0]
    maxspeedstr=psplit[-2]
    maxspeed=maxspeedstr.split('=')[-1]
    if drawV!=[]:
        if int(maxspeed) in drawV:
            pass
        else:
            return 0
    if drawR!=[]:
        if rid in drawR:
            pass
        else:
            return 0
    shp=loads(road_all[road_all.link_id==rid]['geometry'].values[0])
    lon0,lat0,lon1,lat1=shp.bounds
    lonlimit=[lon0,lon1]
    latlimit=[lat0,lat1]
    A=pd.read_csv(p)
    gs=A.groupby(['fid','tid'])
    plt.figure()
    for name,g in gs:
        lon=g.lon.to_list()
        lat=g.lat.to_list()
        speed=g[speedName].to_list()
        plt.plot(g.lon,g.lat,style)
        plt.grid()
        plt.xlim(lonlimit)
        plt.ylim(latlimit)
        plt.xlabel('经度')
        plt.ylabel('纬度')
        plt.title('MAXSPEED=%s,ID=%s,HOUR=%s'%(maxspeed,name,g.hour.unique()[0]))
        for a,b,t in zip(lon,lat,speed):
            plt.annotate(t,xy=(a,b))
        plt.tight_layout()
        fname='%s_ID=%s_HOUR=%s'%(rid,name,g.hour.unique()[0])
        outpath1=tu.pathCheck(outpath+'轨迹可视化/%s_speedName=%s/%s/%s/'%(psplit[-3],speedName,psplit[-2],rid))
        plt.savefig(outpath1+fname)
        plt.clf()
    plt.close()