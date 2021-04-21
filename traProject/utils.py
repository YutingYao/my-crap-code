# utils

import heapq
import os

import numpy as np
from numpy.lib import interp
import pandas as pd
from numpy.lib.shape_base import tile
from scipy.stats import norm
from tqdm import tqdm
from shapely.geometry import LineString, Point, Polygon
from shapely.wkt import dumps, loads


def timespan2unix(timespan, isBeijing=True):
    """
    [文本时间戳转unix时间戳]

    :param timespan: [文本类型时间/时间范围]
    :type timespan: [type]
    :param isBeijing: [是否+8h], defaults to True
    :type isBeijing: bool, optional
    :return: [unix时间戳]
    :rtype: [type]
    """
    delta = 0
    if isBeijing:
        delta = 28800
    if len(timespan) == 1:
        unix = pd.to_datetime(timespan).value/1000000000 - delta
    else:
        unix = [pd.to_datetime(timespan[0]).value/1000000000 - delta,
                pd.to_datetime(timespan[1]).value/1000000000-delta]
    return unix


def csvMerge(targetFile, outFile='', outName='filesMerge', title=None, sepstr=',', postfix='.csv', sortf=False, sortr=None, fileNameAsIdx=False):
    """
    [合并csv文件]

    :param targetFile: [目标文件夹]
    :type targetFile: [type]
    :param outFile: [输出文件夹], defaults to ''
    :type outFile: str, optional
    :param outName: [输出文件名], defaults to 'filesMerge'
    :type outName: str, optional
    :param title: [表头], defaults to None
    :type title: [type], optional
    :param sepstr: [分隔符], defaults to ','
    :type sepstr: str, optional
    :param postfix: [文件名后缀], defaults to '.csv'
    :type postfix: str, optional
    :param sortf: [是否排序文件], defaults to False
    :type sortf: bool, optional
    :param sortr: [排序规则], defaults to None
    :type sortr: [例：key=lambda x: int(x.split('/')[-1][:-4])], optional
    :param fileNameAsIdx: [文件名作为索引 路网项目组数据使用], defaults to False
    :type fileNameAsIdx: bool, optional
    """
    if outFile == '':
        outFile = targetFile
    filenameList = getFileNames(targetFile, postfix=postfix)
    fnameList = getFileNames(targetFile, postfix=postfix, fullName=False)
    if fileNameAsIdx == True:
        idList = list(range(len(fnameList)))
        f = np.array(fnameList).reshape(-1, 1)
        l = np.array(idList).reshape(-1, 1)
        ANS = np.hstack([l, f])
        ANS = pd.DataFrame(ANS, columns=['ID', 'fname']).to_csv(
            outFile+'/ID-fname.csv', index=0)
    if sortf == True:
        filenameList.sort(key=lambda x: int(x.split('/')[-1][:-4]))
    for idx, f in enumerate(filenameList):
        tmp = pd.read_csv(f, sep=sepstr, names=title, header=None)
        if fileNameAsIdx == True:
            tmp['fileNames'] = idList[idx]
            if not title is None:
                title.append('fileNames')
        if idx == 0:
            tmp.to_csv(outFile+outName+'.csv', mode='a',
                       header=title, index=False)
        else:
            tmp.to_csv(outFile+outName+'.csv', mode='a', header=0, index=False)
               
def csvMerge2(targetFile, outFile='', outName='filesMerge', title=None, sepstr=',', postfix='.csv', sortf=False, sortr=None, fileNameAsIdx=False):
    """
    [合并csv文件]

    :param targetFile: [目标文件夹]
    :type targetFile: [type]
    :param outFile: [输出文件夹], defaults to ''
    :type outFile: str, optional
    :param outName: [输出文件名], defaults to 'filesMerge'
    :type outName: str, optional
    :param title: [表头], defaults to None
    :type title: [type], optional
    :param sepstr: [分隔符], defaults to ','
    :type sepstr: str, optional
    :param postfix: [文件名后缀], defaults to '.csv'
    :type postfix: str, optional
    :param sortf: [是否排序文件], defaults to False
    :type sortf: bool, optional
    :param sortr: [排序规则], defaults to None
    :type sortr: [例：key=lambda x: int(x.split('/')[-1][:-4])], optional
    :param fileNameAsIdx: [文件名作为索引 路网项目组数据使用], defaults to False
    :type fileNameAsIdx: bool, optional
    exmaple：csvMerge2('./test0/','./','lzlmerge',[l for l in range(11)],postfix='.txt',fileNameAsIdx=True)
    """
    if outFile == '':
        outFile = targetFile
    filenameList = getFileNames(targetFile, postfix=postfix)
    fnameList = getFileNames(targetFile, postfix=postfix, fullName=False)
    if fileNameAsIdx == True:
        idList = list(range(len(fnameList)))
        fn = np.array(fnameList).reshape(-1, 1)
        fid = np.array(idList).reshape(-1, 1)
        ANS = np.hstack([fid, fn])
        ANS = pd.DataFrame(ANS, columns=['ID', 'fname']).to_csv(
            outFile+'/ID-fname_lzl.csv', index=0)
    if sortf == True:
        filenameList.sort(key=lambda x: int(x.split('/')[-1][:-4]))
    for idx, f in enumerate(filenameList):
        title1=title.copy()
        tmp = pd.read_csv(f, sep=sepstr, names=title1, header=None)
        if fileNameAsIdx == True:
            tmp['fileNames'] = idList[idx]
            if not title1 is None:
                title1.append('fileNames')
        if idx == 0:
            tmp.to_csv(outFile+outName+'.csv', mode='a',
                       header=title1, index=False)
        else:
            tmp.to_csv(outFile+outName+'.csv', mode='a', header=None, index=False)
        title1.clear()

def pathCheck(A):
    """
    [辅助函数，检查路径是否存在,不存在则生成]

    :param A: [路径]
    :type A: [type]
    :return: [路径]
    :rtype: [type]
    """
    if not os.path.exists(A):
        os.makedirs(A)
    return A


def read_txt(filePath, nrows=100):
    """
    [用于读取大型txt文件]

    :param filePath: [路径]
    :type filePath: [type]
    :param nrows: [前n行], defaults to 100
    :type nrows: int, optional
    """
    fp = open(filePath, 'r')
    while nrows > 0:
        print(fp.readline(), end=' ')
        nrows -= 1
    fp.close()


def getFileNames(fileName, prefix=None, postfix=None, fullName=True):
    """
    [返回一个文件夹内所有文件的完整路径名]

    :param fileName: [文件夹名]
    :type fileName: [type]
    :param prefix: [前缀], defaults to None
    :type prefix: [type], optional
    :param postfix: [后缀], defaults to None
    :type postfix: [type], optional
    :param fullName: [完整路径], defaults to True
    :type fullName: bool, optional
    :return: [路径名]
    :rtype: [type]
    """
    nlist = os.listdir(fileName)
    if (prefix == None) & (postfix == None):
        x = nlist
    else:
        x = []
        if (prefix != None) & (postfix != None):
            for file in nlist:
                if file.startswith(prefix) & file.endswith(postfix):
                    x.append(file)
        elif prefix != None:
            for file in nlist:
                if file.startswith(prefix):
                    x.append(file)
        elif postfix != None:
            x = []
            for file in nlist:
                if file.endswith(postfix):
                    x.append(file)
    y = x
    if fullName == True:
        y = [fileName+file for file in x]
    return y


def scan_files(directory, prefix=None, postfix=None):
    """
    [扫描指定路径下符合要求的文件（包含子目录）]

    :param directory: [目标路径]
    :type directory: [type]
    :param prefix: [前缀], defaults to None
    :type prefix: [type], optional
    :param postfix: [后缀], defaults to None
    :type postfix: [type], optional
    :return: [files_list]
    :rtype: [type]
    """
    files_list = []

    for root, sub_dirs, files in os.walk(directory):
        for special_file in files:
            if postfix:
                if special_file.endswith(postfix):
                    files_list.append(os.path.join(root, special_file))
            elif prefix:
                if special_file.startswith(prefix):
                    files_list.append(os.path.join(root, special_file))
            else:
                files_list.append(os.path.join(root, special_file))

    return files_list


def csv2txt(csvpath, txtpath):
    """
    [csv转txt]

    :param csvpath: [csv路径]
    :type csvpath: [type]
    :param txtpath: [txt路径]
    :type txtpath: [type]
    """
    data = pd.read_csv(csvpath, encoding='utf-8')
    with open(txtpath, 'a+', encoding='utf-8') as f:
        for line in data.values:
            f.write((str(line)+'\n'))


def calDate(data, timeindex='timestamp', timestr=False, calList=[], isbeijing=True, unit='s'):
    """
    [生成时间列]

    :param data: [数据或数据路径]
    :type data: [type]
    :param timestr: [计算文本形式时间], defaults to False
    :type timestr: bool, optional
    :param calList: [计算列表], defaults to []
    :type calList: list, optional
    :param isbeijing: [是否UTC+8], defaults to True
    :type isbeijing: bool, optional
    :return: [处理后的数据]
    :rtype: [type]
    """
    delta = 0
    if isbeijing:
        delta = 28800
    if isinstance(data, str):
        data = pd.read_csv(data)
    tmp = pd.to_datetime(data[timeindex]+delta, unit=unit)
    if timestr == True:
        data['timeStr'] = tmp
    if 'hour' in calList:
        data['hour'] = tmp.dt.hour
    if 'weekday' in calList:
        data['weekday'] = tmp.dt.weekday
    if 'month' in calList:
        data['month'] = tmp.dt.month
    if 'day' in calList:
        data['day'] = tmp.dt.day
    if 'year' in calList:
        data['year'] = tmp.dt.year
    return data


def calFeature(datapath, outfile='', N=24, minPNum1=10, minPNum2=30, speedName='speed'):
    """
    [计算特征]

    :param datapath: [输入数据/路径]
    :type datapath: [type]
    :param outfile: [输出路径未指定则以函数返回值形式处理], defaults to ''
    :type outfile: str, optional
    :param N: [最大前N个速度], defaults to 24
    :type N: int, optional
    :param minPNum: [最小特征点数], defaults to 10
    :type minPNum: int, optional
    :param speedName: [速度列名], defaults to 'speed'
    :type minPNum: str, optional
    :return: [特征：
    VTopN:最大的前N个速度
    Vmean:总体平均瞬时速度
    Vstd:路段总体标准差
    V85 百分之85车速
    V15 百分之15车速
    V95 百分之95车速
    deltaV V85-V15
    Vmode 路段速度众数]
    :rtype: [type]
    """
    oridata = pd.read_csv(datapath)
    if len(oridata) < minPNum1:
        return 0
    linkID = datapath.split('/')[-1].split('.')[0]
    data = oridata[oridata[speedName] >= 5]
    ANS = None
    Features = [linkID]
    gs = data.groupby('hour')
    Vs = []
    for idx, g in gs:
        if len(g) < minPNum1:
            continue
        else:
            vtmp = g[speedName].mean()  # cal vTOPN
            Vs.append(vtmp)
    VTopN = heapq.nlargest(N, Vs)
    if len(VTopN) < N:
        VTopN.extend(['null']*(N-len(VTopN)))
    Features.extend(VTopN)
    tmp = data[speedName].copy()
    if len(tmp) < minPNum2:
        Features.extend(['null']*7)
    else:
        tmp.sort_values(inplace=True)
        Vmean = tmp.mean()  # cal vmean
        Vstd = tmp.std()  # cal Rstd
        V95 = tmp.quantile(0.95, interpolation='nearest')  # V95
        V85 = tmp.quantile(0.85, interpolation='nearest')  # V85
        V15 = tmp.quantile(0.15, interpolation='nearest')  # V15
        deltaV = V85-V15  # V85-V15
        Vmode = tmp.apply(lambda x: int(x+0.5)).mode()[0]  # Vmode
        Features.extend([Vmean, Vstd, V95, V85, V15, deltaV, Vmode])
    if ANS is None:
        ANS = np.array([[c] for c in Features]).T
    else:
        ansTmp = np.array([[c] for c in Features]).T
        ANS = np.vstack((ANS, ansTmp))
    cols = ['linkID']
    VN = ['V'+str(idx) for idx in range(N)]
    cols.extend(VN)
    cols.extend(['Vmean', 'Vstd', 'V95', 'V85', 'V15', 'deltaV', 'Vmode'])
    ANS = pd.DataFrame(ANS, columns=cols)
    if outfile != '':
        # fname = str(linkID)+'_特征工程.csv'
        # ANS.to_csv(outfile+fname, index=0)
        fname = outfile + \
            'N=%d_minPNum1=%d_minPNum2=%d_speedName=%s.csv' % (
                N, minPNum1, minPNum2, speedName)
        if not os.path.exists(fname):
            ANS.to_csv(fname, index=0, mode='a')
        else:
            ANS.to_csv(fname, index=0, mode='a', header=None)
    else:
        return ANS


def genDataSet(datapath, roadpath, outfile='', dname='测试数据集', speedName='speed', minPNum1=10, yrName='link_id', xrName='linkID', attrNames=['oneway', 'layer', 'bridge', 'tunnel', 'viaduct', 'numofLines', 'maxspeed', 'geometry']):
    """
    [计算特征]

    :param datapath: [输入数据/路径]
    :type datapath: [type]
    :param outfile: [输出路径未指定则以函数返回值形式处理], defaults to ''
    :type outfile: str, optional
    :param N: [最大前N个速度], defaults to 24
    :type N: int, optional
    :param minPNum: [最小特征点数], defaults to 10
    :type minPNum: int, optional
    :param speedName: [速度列名], defaults to 'speed'
    :type minPNum: str, optional
    :return: [特征：
    VTopN:最大的前N个速度
    Vmean:总体平均瞬时速度
    Vstd:路段总体标准差
    V85 百分之85车速
    V15 百分之15车速
    V95 百分之95车速
    deltaV V85-V15
    Vmode 路段速度众数]
    :rtype: [type]
    """
    oridata = pd.read_csv(datapath)
    PNum2 = len(oridata)
    linkID = datapath.split('/')[-1].split('.')[0]
    data = oridata[oridata[speedName] >= 5]
    ANS = None
    Features = [linkID]
    gs = data.groupby('hour')
    Vs = []
    for idx, g in gs:
        if len(g) < minPNum1:
            continue
        else:
            vtmp = g[speedName].mean()  # cal vTOPN
            Vs.append(vtmp)
    VTopN = heapq.nlargest(24, Vs)
    N = len(VTopN)
    if N < 24:
        VTopN.extend(['null']*(24-len(VTopN)))
    Features.extend(VTopN)
    tmp = data[speedName].copy()
    if len(tmp) < 1:
        Features.extend(['null']*7)
    else:
        tmp.sort_values(inplace=True)
        Vmean = tmp.mean()  # cal vmean
        Vstd = tmp.std()  # cal Rstd
        V95 = tmp.quantile(0.95, interpolation='nearest')  # V95
        V85 = tmp.quantile(0.85, interpolation='nearest')  # V85
        V15 = tmp.quantile(0.15, interpolation='nearest')  # V15
        deltaV = V85-V15  # V85-V15
        Vmode = tmp.apply(lambda x: int(x+0.5)).mode()[0]  # Vmode
        Features.extend([Vmean, Vstd, V95, V85, V15, deltaV, Vmode])
    Features.extend([N, PNum2])
    if ANS is None:
        ANS = np.array([[c] for c in Features]).T
    else:
        ansTmp = np.array([[c] for c in Features]).T
        ANS = np.vstack((ANS, ansTmp))
    cols = ['linkID']
    VN = ['VTOP'+str(idx) for idx in range(24)]
    cols.extend(VN)
    cols.extend(['Vmean', 'Vstd', 'V95', 'V85', 'V15', 'deltaV', 'Vmode'])
    cols.extend(['N', 'PNum2'])

    ANS = pd.DataFrame(ANS, columns=cols)
    ylist = pd.read_csv(roadpath)
    Check = ylist[yrName].tolist()
    ANS = ANS[ANS[xrName].isin(Check)]
    if len(attrNames) == 0:
        attrNames = list(ylist)
    for attr in attrNames:
        ANS[attr] = ANS[xrName].apply(
            lambda x: ylist[ylist[yrName] == x][attr].values[0])
    if outfile != '':
        # fname = str(linkID)+'_特征工程.csv'
        # ANS.to_csv(outfile+fname, index=0)
        fname = outfile + '%s.csv' % (dname)
        # 'N=%d_minPNum1=%d_minPNum2=%d_speedName=%s.csv' % (N, minPNum1,minPNum2,speedName)
        if not os.path.exists(fname):
            ANS.to_csv(fname, index=0, mode='a', encoding="utf-8-sig")
        else:
            ANS.to_csv(fname, index=0, mode='a',
                       header=None, encoding="utf-8-sig")
    else:
        return ANS


def appendRoadAttr(xpath, ypath, outpath, yrName='link_id', xrName='linkID', attrNames=['oneway', 'layer', 'bridge', 'tunnel', 'viaduct', 'numofLines', 'maxspeed', 'geometry']):
    # '''
    # 生成数据集合文件
    # xpath 特征文件路径
    # ypath 标签文件路径
    # outpath 输出路径
    # '''
    """
    [拼接路段属性]

    :param xpath: [特征文件路径]
    :type xpath: [type]
    :param ypath: [路段文件路径]
    :type ypath: [type]
    :param outpath: [description]
    :type outpath: [输出文件路径]
    :param yrName: [路段文件id], defaults to 'link_id'
    :type yrName: str, optional
    :param xrName: [特征文件id], defaults to 'linkID'
    :type xrName: str, optional
    :param attrNames: [待拼接属性], defaults to ['oneway','layer', 'bridge', 'tunnel', 'viaduct', 'numofLines','maxspeed','geometry']
    :type attrNames: list, optional
    """
    flist = getFileNames(xpath)
    ylist = pd.read_csv(ypath)
    Check = ylist[yrName].tolist()
    for idx, f in enumerate(tqdm(flist)):
        fname = f.split('/')[-1].split('.')[0]
        tmp = pd.read_csv(f)
        tmp = tmp[tmp[xrName].isin(Check)]
        for attr in attrNames:
            tmp[attr] = tmp[xrName].apply(
                lambda x: ylist[ylist[yrName] == x][attr].values[0])
        tmp.to_csv(outpath+fname+'dataSet.csv', index=0, encoding="utf-8-sig")


def getFilesBySize(filePath, descending=True):
    """
    [按文件大小获取列表]

    :param filePath: [文件路径]
    :type filePath: [type]
    :param descending: [降序], defaults to True
    :type descending: bool, optional
    :return: [description]
    :rtype: [type]
    """
    fileMap = {}
    size = 0
    for parent, dirnames, filenames in os.walk(filePath):
        for filename in filenames:

            size = os.path.getsize(os.path.join(parent, filename))
            fileMap.setdefault(os.path.join(parent, filename), size)

    filelist = sorted(fileMap.items(), key=lambda d: d[1], reverse=descending)
    ANS = []
    for filename, size in filelist:
        ANS.append(filename)
    return ANS


def vFilter(A, speedName='speed', vlimt=[5, 120]):
    """
    [速度过滤]

    :param A: [速度]
    :type A: [type]
    :param speedName: [速度列名], defaults to 'speed'
    :type speedName: str, optional
    :return: [description]
    :rtype: [type]
    """
    A = A[(A[speedName] >= vlimt[0]) & (A[speedName] <= vlimt[1])]
    mu, sigma = norm.fit(A[speedName])
    gFilter = [mu-3*sigma, mu+3*sigma]
    A = A[(A[speedName] > gFilter[0]) & (A[speedName] < gFilter[1])]
    return A


def dfReC(df, cname, cpos):
    '''
    dataframe 移动列位置
    '''
    tmp = df.pop(cname)
    df.insert(cpos, cname, tmp)
    return df


def vFilter2(A, speedName='speed', vlimt=[5, 120]):
    A = A[(A[speedName] >= vlimt[0]) & (A[speedName] <= vlimt[1])]
    percentile = np.percentile(
        A[speedName], (25, 50, 75), interpolation='linear')
    Q1 = percentile[0]  # 上四分位数
    Q2 = percentile[1]
    Q3 = percentile[2]  # 下四分位数
    IQR = Q3 - Q1  # 四分位距
    ulim = Q3 + 1.5*IQR  # 上限 非异常范围内的最大值
    llim = Q1 - 1.5*IQR  # 下限 非异常范围内的最小值
    if llim < 0:
        llim = 0
    gFilter = [llim, ulim]
    A = A[(A[speedName] > gFilter[0]) & (A[speedName] < gFilter[1])]  # 箱型图剔除异常
    return A


def genNewspeed(A, vlimt=[5, 120],minAcc=50,velocityName='velocity',ws=0.5,wv=0.5):
    """
    [速度过滤]

    :param A: [速度]
    :type A: [type]
    :param speedName: [速度列名], defaults to 'speed'
    :type speedName: str, optional
    :return: [description]
    :rtype: [type]
    """
    # A=vFilter2(A,'speed')#先筛选一遍原始速度
    # A=vFilter2(A,'velocity')#再筛一遍计算速度
    speedName = 'speed'
    A = A[(A[speedName] >= vlimt[0]) & (A[speedName] <= vlimt[1])]
    speedName = velocityName
    A = A[(A[speedName] >= vlimt[0]) & (A[speedName] <= vlimt[1])]
    A['newspeed'] = A.apply(lambda a: a[velocityName]*wv+a.speed*ws if (a.type ==
                                                                 0 and a[velocityName] > 0 and a.acc <= minAcc) else a.speed, axis=1)  # 融合
    A = vFilter2(A, 'newspeed')  # 筛一遍新速度
    return A


def csvMerge1(targetFile, outFile='', outName='filesMerge', title=None, sepstr=',', postfix='.csv', sortf=False, sortr=None, fileNameAsIdx=False):
    """
    [合并csv文件]

    :param targetFile: [目标文件夹]
    :type targetFile: [type]
    :param outFile: [输出文件夹], defaults to ''
    :type outFile: str, optional
    :param outName: [输出文件名], defaults to 'filesMerge'
    :type outName: str, optional
    :param title: [表头], defaults to None
    :type title: [type], optional
    :param sepstr: [分隔符], defaults to ','
    :type sepstr: str, optional
    :param postfix: [文件名后缀], defaults to '.csv'
    :type postfix: str, optional
    :param sortf: [是否排序文件], defaults to False
    :type sortf: bool, optional
    :param sortr: [排序规则], defaults to None
    :type sortr: [例：key=lambda x: int(x.split('/')[-1][:-4])], optional
    :param fileNameAsIdx: [文件名作为索引 路网项目组数据使用], defaults to False
    :type fileNameAsIdx: bool, optional
    """
    if outFile == '':
        outFile = targetFile
    filenameList = getFileNames(targetFile, postfix=postfix)
    fnameList = getFileNames(targetFile, postfix=postfix, fullName=False)
    if fileNameAsIdx == True:
        idList = ['F'+str(i).rjust(7, '0') for i in range(len(fnameList))]
        f = np.array(fnameList).reshape(-1, 1)
        l = np.array(idList, np.str).reshape(-1, 1)
        ANS = np.hstack([l, f])
        ANS = pd.DataFrame(ANS, columns=['ID', 'fname']).to_csv(
            outFile+'/ID-fname.csv', index=0)
    if sortf == True:
        filenameList.sort(key=lambda x: int(x.split('/')[-1][:-4]))
    for idx, f in enumerate(filenameList):
        tmp = pd.read_csv(f, sep=sepstr, names=title, header=None)
        if fileNameAsIdx == True:
            tmp['fileNames'] = idList[idx]
            tmp[0] = tmp[0].apply(
                lambda x: tmp['fileNames']+'P'+str(x).rjust(4, '0'))
            if not title is None:
                title.append('fileNames')
        if idx == 0:
            tmp.to_csv(outFile+outName+'.csv', mode='a',
                       header=title, index=False)
        else:
            tmp.to_csv(outFile+outName+'.csv', mode='a', header=0, index=False)

# csvMerge1('./test/','./',postfix='.txt',fileNameAsIdx=True)


def joind2data_hw(floder1, floder2, floder3):
    list_dir = os.listdir(floder2)  # bug
    filesnum = len(list_dir)
#     print(filesnum)
    process = 1
    for cur_filename in tqdm(list_dir):
        #         print(cur_filename)
        process += 1
#         print('\r', process, end='')
        curfilepath1 = os.path.join(floder1, cur_filename)
        curfilepath2 = os.path.join(floder2, cur_filename)
        curfilepath3 = os.path.join(floder3, cur_filename)
        with open(curfilepath1, 'rb')as p1_obj, open(curfilepath2, 'rb')as p2_obj:
            p_lines1 = pd.read_csv(p1_obj, header=None)
            p_lines2 = pd.read_csv(p2_obj, header=None)
            # print(p_lines1)
            # print(p_lines2)
            p_lines2[5] = p_lines1[1]
            p_lines2[6] = p_lines1[2]
            p_lines2[7] = p_lines1[3]
            p_lines2[8] = p_lines1[4]
            p_lines2[9] = p_lines1[7]
            p_lines2[10] = p_lines1[8]
            # print(p_lines2)
            p_lines2.to_csv(curfilepath3, float_format='%6f',
                            sep=',', header=None, index=0)
            # with open(curfilepath3, 'rb')as p3_obj:
            #p_lines3 = pd.read_csv(p3_obj, header=None)
            # print(p_lines3)
            # return


def dfReC(df, cname, cpos):
    '''
    dataframe 移动列位置
    '''
    tmp = df.pop(cname)
    df.insert(cpos, cname, tmp)
    return df


def cutBZ(df, clist=[]):
    '''
    删除备注
    '''
    for c in clist:
        df[c] = df[c].apply(lambda x: x[0])
    return df


def calXdis(geo):
    '''计算位移'''
    line = loads(geo)
    plist = list(line[0].coords)
    p0 = np.array(plist[0])
    p1 = np.array(plist[-1])
    ans = np.sqrt(np.sum(np.square(p0-p1)))
    return ans


def calFeature1(datapath, outfile='', N=24, minPNum=50, speedName='speed', linkID='', maxspeed=''):
    """
    [计算特征]

    :param datapath: [输入数据/路径]
    :type datapath: [type]
    :param outfile: [输出路径未指定则以函数返回值形式处理], defaults to ''
    :type outfile: str, optional
    :param N: [最大前N个速度], defaults to 24
    :type N: int, optional
    :param minPNum: [最小特征点数], defaults to 10
    :type minPNum: int, optional
    :param speedName: [速度列名], defaults to 'speed'
    :type minPNum: str, optional
    :return: [特征：
    VTopN:最大的前N个速度
    Vmean:总体平均瞬时速度
    Vstd:路段总体标准差
    V85 百分之85车速
    V15 百分之15车速
    V95 百分之95车速
    deltaV V85-V15
    Vmode 路段速度众数]
    :rtype: [type]
    """
    oridata = pd.read_csv(datapath)
    if len(oridata) < minPNum:
        return 0
    if linkID == '':
        linkID = datapath.split('/')[-1].split('.')[0]  # 分割link_id
    if maxspeed == '':
        maxspeed = datapath.split('/')[-2].split('maxspeed=')[1]  # maxspeed
    data = oridata[oridata[speedName] >= 5]
    ANS = None
    Features = [linkID, int(maxspeed)]
    gs = data.groupby('hour')
    Vs = []
    for idx, g in gs:
        if len(g) < minPNum:
            continue
        else:
            vtmp = g[speedName].mean()  # cal vTOPN
            Vs.append(vtmp)
    VTopN = heapq.nlargest(N, Vs)
    if len(VTopN) < N:
        VTopN.extend(['null']*(N-len(VTopN)))
    Features.extend(VTopN)

    tmp = data[speedName].copy()
    percetIndex = list(np.arange(0, 1.05, 0.05))
    if len(tmp) < minPNum:
        Features.extend(['null']*24)
    else:
        tmp.sort_values(inplace=True)
        Vmean = tmp.mean()  # cal vmean #待修改
        Vstd = tmp.std()  # cal Rstd
        # VMin,V5~V95,VMax
        Vpercent = [tmp.quantile(p, interpolation='nearest')
                    for p in percetIndex]
        Vmode = tmp.apply(lambda x: int(x+0.5)).mode()[0]  # Vmode
        Features.extend([Vmean, Vstd])
        Features.extend(Vpercent)
        Features.append(Vmode)
    if ANS is None:
        ANS = np.array([[c] for c in Features]).T
    else:
        ansTmp = np.array([[c] for c in Features]).T
        ANS = np.vstack((ANS, ansTmp))
    cols = ['linkID', 'maxspeed']
    VTopN = ['VTop'+str(idx) for idx in range(N)]
    cols.extend(VTopN)
    cols.extend(['Vmean', 'Vstd'])
    cols.extend(['V'+str(p*100) for p in percetIndex])
    cols.extend(['Vmode'])
    ANS = pd.DataFrame(ANS, columns=cols)
    if outfile != '':
        # fname = str(linkID)+'_特征工程.csv'
        # ANS.to_csv(outfile+fname, index=0)
        fname = outfile + \
            'N=%d_minPNum=%d_speedName=%s.csv' % (N, minPNum, speedName)
        if not os.path.exists(fname):
            ANS.to_csv(fname, index=0, mode='a')
        else:
            ANS.to_csv(fname, index=0, mode='a', header=None)
    else:
        return ANS


def genDataSet1(datapath,
                roadpath,
                outfile='',
                dname='测试数据集',
                speedName='speed',
                minPNum=50,
                yrName='link_id',
                xrName='linkID',
                linkID='',
                maxspeed='',
                attrNames=['oneway', 'layer', 'bridge', 'tunnel', 'viaduct',
                           'numofLines', 'geometry', 'Sdis', 'Xdis', 'RC'],
                dropWifi=False,
                dropViaduct=False,
                vTopNMode=1,
                percent=0.5):
    """
    [生成数据集]

    :param datapath: [轨迹数据路径]
    :type datapath: [type]
    :param roadpath: [路网数据路径]
    :type roadpath: [type]
    :param outfile: [输出路径], defaults to ''
    :type outfile: str, optional
    :param dname: [输出文件名], defaults to '测试数据集'
    :type dname: str, optional
    :param speedName: [速度名], defaults to 'speed'
    :type speedName: str, optional
    :param minPNum: [参与统计的最小点数], defaults to 50
    :type minPNum: int, optional
    :param yrName: [路网id列名], defaults to 'link_id'
    :type yrName: str, optional
    :param xrName: [轨迹id列名], defaults to 'linkID'
    :type xrName: str, optional
    :param linkID: [轨迹路段linkid], defaults to ''
    :type linkID: str, optional
    :param maxspeed: [轨迹路段限速], defaults to ''
    :type maxspeed: str, optional
    :param attrNames: [拼接属性名], defaults to ['oneway', 'layer', 'bridge', 'tunnel', 'viaduct', 'numofLines', 'geometry','Sdis','Xdis','RC']
    :type attrNames: list, optional
    :param dropWifi: [删除wifi点], defaults to False
    :type dropWifi: bool, optional
    :param dropViaduct: [删除有遮挡路段], defaults to False
    :type dropViaduct: bool, optional
    :return: [description]
    :rtype: [type]
    """
    oridata = pd.read_csv(datapath)
    if dropWifi == True:
        oridata = oridata[oridata.type == 0]
        oridata.reset_index(drop=True, inplace=True)
    pNum = len(oridata)
    if pNum < minPNum:
        return 0
    if linkID == '':
        linkID = datapath.split('/')[-1].split('.')[0]  # 分割link_id
    if maxspeed == '':
        maxspeed = datapath.split('/')[-2].split('maxspeed=')[1]  # maxspeed
    data = oridata[(oridata[speedName] >= 5) & (oridata[speedName] <= 120)]
    ANS = None
    Features = [linkID, int(maxspeed), pNum]
    gs = data.groupby('hour')
    Vs = []
    for idx, g in gs:
        if len(g) < minPNum:
            continue
        else:
            if vTopNMode==1:
                vtmp = g[speedName].mean()  # cal vTOPN #TODO待改(不是取平均值，而是取分位数值)
            else:
                vtmp = g[speedName].quantile(percent,interpolation='nearest') #TAG 不是取平均值，而是取分位数值 50
            Vs.append(vtmp)
    VTopN = heapq.nlargest(24, Vs)
    N = len(VTopN)
    if N < 24:
        VTopN.extend(['null']*(24-len(VTopN)))
    Features.extend(VTopN)
    tmp = data[speedName].copy()
    percetIndex = list(np.arange(0, 1.05, 0.05))
    if len(tmp) < minPNum:
        Features.extend(['null']*24)
    else:
        tmp.sort_values(inplace=True)
        Vmean = tmp.mean()  # cal vmean #FIXME 待修改
        Vstd = tmp.std()  # cal Vstd
        # VMin,V5~V95,VMax
        Vpercent = [tmp.quantile(p, interpolation='nearest')
                    for p in percetIndex]
        Vmode = tmp.apply(lambda x: int(x+0.5)).mode()[0]  # Vmode
        Features.extend([Vmean, Vstd])
        Features.extend(Vpercent)
        Features.append(Vmode)
    if ANS is None:
        ANS = np.array([[c] for c in Features]).T
    else:
        ansTmp = np.array([[c] for c in Features]).T
        ANS = np.vstack((ANS, ansTmp))
    cols = ['linkID', 'maxspeed', 'pNum']
    VTopN = ['VTop'+str(idx) for idx in range(24)]
    cols.extend(VTopN)
    cols.extend(['Vmean', 'Vstd'])
    cols.extend(['V'+str(int(p*100)) for p in percetIndex])
    cols.extend(['Vmode'])
    ANS = pd.DataFrame(ANS, columns=cols)

    ylist = pd.read_csv(roadpath)
    Check = ylist[yrName].tolist()
    ANS = ANS[ANS[xrName].isin(Check)]
    if len(ANS)==0:
        return 0
    if len(attrNames) == 0:
        attrNames = list(ylist)
    for attr in attrNames:
        ANS[attr] = ANS[xrName].apply(
            lambda x: ylist[ylist[yrName] == x][attr].values[0])
    if dropViaduct == True and ANS.loc[0, 'viaduct'] == 'T':
        return 0
    else:
        if outfile != '':
            fname = outfile + '%s.csv' % (dname)
            # 'N=%d_minPNum1=%d_minPNum2=%d_speedName=%s.csv' % (N, minPNum1,minPNum2,speedName)
            if dropViaduct==True:
                ANS.drop('viaduct',axis=1,inplace=True)
            if not os.path.exists(fname):
                ANS.to_csv(fname, index=0, mode='a', encoding="utf-8-sig")
            else:
                ANS.to_csv(fname, index=0, mode='a',
                           header=None, encoding="utf-8-sig")
        else:
            return ANS



def trans_data(data=None, fname='特征测试_newspeed_遮挡.csv', path='D:/DiplomaProject/output/paper/特征测试/', minN=2, vlist=[20, 30, 40, 50, 60, 70, 80], onehot_y=True, minmax_x=True, onehot_x=False, concols=['viaduct'], out2File=False, outpath='D:/DiplomaProject/output/paper/特征测试/', postfix='转换'):
    if data is None:#TODO 该函数待重构
        data = pd.read_csv(path+fname)
        data.maxspeed = data.maxspeed.astype(int)
    data.drop(['linkID', 'pNum'], axis=1, inplace=True)
    data = data[data.maxspeed.isin(vlist)]
    data = data.reset_index(drop=True)
    tmp = data['maxspeed']
    if onehot_y == True:
        tmp = pd.get_dummies(tmp)
    else:
        tmp = pd.DataFrame(tmp)
    df = data.drop('maxspeed', axis=1)
    if onehot_x == True:
        df = pd.get_dummies(df, columns=concols)
    if minmax_x == True:
        df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    ans = tmp.join(df)
    vdrop = ['VTop'+str(i) for i in range(minN, 24)]
    ans.drop(vdrop, axis=1, inplace=True)
    ans.dropna(inplace=True)
    ans.reset_index(drop=True, inplace=True)
    if out2File == True and outpath != '':
        new_name = fname.replace('.csv', '_%s.csv' % postfix)
        ans.to_csv(outpath+new_name, index=0)
    else:
        return ans

def getSample(rpath='../../output/paper/轨迹_按link_id_v2_Pts_minN=50_bufR=1e-05_dropWifi=False_velocity=velocity1/',
              outfile='../../output/paper/样本路段/',
              sampleSize=[0,0.25,0.5,0.75]):
    """
    [获取样本]

    :param rpath: [轨迹路径], defaults to '../../output/paper/轨迹_按link_id_v2_Pts_minN=50_bufR=1e-05_dropWifi=False_velocity=velocity1/'
    :type rpath: str, optional
    :param outfile: [样本输出路径], defaults to '../../output/paper/样本路段/'
    :type outfile: str, optional
    :param sampleSize: [样本分位数列表], defaults to [0,0.25,0.5,0.75]
    :type sampleSize: list, optional
    """    
    
    postfix=rpath.split('/')[-2].split('轨迹_')[1]
    flist0=getFileNames(rpath)
    sampleName=[str(int(100*(1-i)))+'位' for i in sampleSize]
    # sampleName=['最大','75位','50位','25位']

    for f in flist0:
        slist=[]
        flist1=getFilesBySize(f)
        for i in range(len(sampleName)):
            tmp=flist1[int(len(flist1)*sampleSize[i])]#百分位值
            slist.append(tmp.replace('\\','/')+'|%s'%sampleName[i])
        for s in slist:
            tmp0=s.split('|')
            tmp=tmp0[0].split('/')
            fname=tmp[-2]+'_%s_'%tmp0[1]+tmp[-1]
            pd.read_csv(tmp0[0]).to_csv(pathCheck(outfile+'%s/%s/'%(postfix,tmp[-2]))+fname,index=0)
    return outfile+'%s/'%postfix