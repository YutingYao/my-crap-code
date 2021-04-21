# This is a python package for lzl's diploma project
import math

import pandas as pd
import traProject.utils as tu
from shapely.geometry import LineString, Point, Polygon
from shapely.wkt import dumps, loads
from tqdm import tqdm
import numpy as np
tqdm.pandas(desc="my bar!")


def getDistance(lat1, lon1, lat2, lon2):
    """
    [计算距离]

    :param lat1: [纬度1]
    :type lat1: [type]
    :param lon1: [经度1]
    :type lon1: [type]
    :param lat2: [纬度2]
    :type lat2: [type]
    :param lon2: [经度2]
    :type lon2: [type]
    :return: [距离]
    :rtype: [type]
    """
    radLat1 = lat1 * math.pi / 180.0  # 角度转换为弧度
    radLat2 = lat2 * math.pi / 180.0
    a = radLat1 - radLat2  # 两点纬度之差
    b = (lon1-lon2) * math.pi / 180.0  # 两点经度之差
    # 计算两点距离的公式
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) +
                                math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    # 弧长乘地球半径，半径为米
    s = s * 6378137.0
    # 精确距离的数值，可有可无
    s = round(s * 10000) / 10000
    return s


def getSpeed_2p(lat1, lon1, lat2, lon2, t1, t2, unit='km/h'):
    """
    [速度计算方法 1 两点]

    :param lat1: [纬度1]
    :type lat1: [type]
    :param lon1: [经度1]
    :type lon1: [type]
    :param lat2: [纬度2]
    :type lat2: [type]
    :param lon2: [经度2]
    :type lon2: [type]
    :param t1: [时间1]
    :type t1: [type]
    :param t2: [时间2]
    :type t2: [type]
    :param unit: [单位], defaults to 'km/h'
    :type unit: str, optional
    :return: [速度]
    :rtype: [type]
    """
    d = getDistance(lat1, lon1, lat2, lon2)
    if t1 != t2:
        v = d / abs(t1-t2)
    else:
        v = 0
    if unit == 'km/h':
        v *= 3.6
    return v


def getSpeed_3p(lat0, lon0, lat1, lon1, lat2, lon2, t0, t2, unit='km/h'):
    """
    [速度计算方法 2 三点]

    :param lat0: [上一点纬度]
    :type lat0: [type]
    :param lon0: [上一点经度]
    :type lon0: [type]
    :param lat1: [当前点纬度]
    :type lat1: [type]
    :param lon1: [当前点经度]
    :type lon1: [type]
    :param lat2: [下一点纬度]
    :type lat2: [type]
    :param lon2: [下一点经度]
    :type lon2: [type]
    :param t0: [上一点时间]
    :type t0: [type]
    :param t2: [下一点时间]
    :type t2: [type]
    :param unit: [单位], defaults to 'km/h'
    :type unit: str, optional
    :return: [速度]
    :rtype: [type]
    """
    d1 = getDistance(lat2, lon2, lat1, lon1)
    d2 = getDistance(lat1, lon1, lat0, lon0)
    if t2 != t0:
        v = (d1+d2) / abs(t2-t0)
    else:
        v = 0
    if unit == 'km/h':
        v *= 3.6
    return v


def getAngle(lat1, lon1, lat2, lon2):
    """
    [计算方向角，单位为°]

    :param lat1: [纬度1]
    :type lat1: [type]
    :param lon1: [经度1]
    :type lon1: [type]
    :param lat2: [纬度2]
    :type lat2: [type]
    :param lon2: [经度2]
    :type lon2: [type]
    :param unit: [单位], defaults to 'degree'
    :type unit: str, optional
    :return: [方向角]
    :rtype: [type]
    """
    lat1_rad = lat1 * math.pi / 180
    lon1_rad = lon1 * math.pi / 180
    lat2_rad = lat2 * math.pi / 180
    lon2_rad = lon2 * math.pi / 180
    y = math.sin(lon2_rad - lon1_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * \
        math.cos(lon2_rad - lon1_rad)
    bearing = math.atan2(y, x)
    bearing = 180 * bearing / math.pi
    bearing = float((bearing + 360.0) % 360.0)
    return round(bearing, 2)


def getVCR(vl, vc):
    """
    [计算VCR]

    :param vl: [vlast]
    :type vl: [type]
    :param vc: [vcurrent]
    :type vc: [type]
    :return: [description]
    :rtype: [type]
    """
    if vc == 0:
        return -9999
    VCR = (vc-vl)/vc
    return VCR


def calv1(x):
    #     lambda x: (
    #             getSpeed_2p(x.latLast, x.lonLast, x.lat, x.lon, x.timestamp, x.timestampLast))
    if pd.isnull(x.lonLast):
        ANS = -9999
    else:
        ANS = getSpeed_2p(x.latLast, x.lonLast, x.lat, x.lon,
                          x.timestamp, x.timestampLast)
    return ANS


def calv2(x):
    #     lambda x: (
    #             getSpeed_3p(x.latLast, x.lonLast, x.lat, x.lon, x.latNext, x.lonNext, x.timestampLast, x.timestampNext))
    if pd.isnull(x.lonLast) or pd.isnull(x.lonNext):
        ANS = -9999
    else:
        ANS = getSpeed_3p(x.latLast, x.lonLast, x.lat, x.lon,
                          x.latNext, x.lonNext, x.timestampLast, x.timestampNext)
    return ANS


def calvcr(x):
    #   lambda x:(getVCR(x.speedLast,x.speed))
    if pd.isnull(x.speedLast):
        ANS = 0
    else:
        ANS = getVCR(x.speedLast, x.speed)
    return ANS


def calV(A, by=['fid', 'tid'], calvfun=1, nameList=['lon', 'lat', 'timestamp'], vname='speed'):
    """
    [利用数学公式计算速度]

    :param A: [轨迹数据]
    :type A: [type]
    :param by: [groupby参数], defaults to ['fid','tid']
    :type by: list, optional
    :param calvfun: [计算速度的方式 1：与前一点构成线段的平均速度 2：前后两点平均速度], defaults to 1
    :type calvfun: int, optional
    :param nameList: [经度、纬度、时间戳的列名], defaults to ['lon','lat','timestamp']
    :type nameList: list, optional
    :param vname: [速度列名], defaults to 'speed'
    :type vname: str, optional
    :return: [A]
    :rtype: [type]
    """
    A, lnames, _ = getNearAtrr(A, by, nameList)
    if calvfun == 2:
        A, _, nnames = getNearAtrr(A, by, nameList, calL=False, calN=True)
    if calvfun == 1:
        A[vname] = A.progress_apply(calv1, axis=1)
    elif calvfun == 2:
        A[vname] = A.progress_apply(calv2, axis=1)
    A = A.drop(lnames, axis=1)
    if calvfun == 2:
        A = A.drop(nnames, axis=1)
    return A


def getNearAtrr(A, by=['fid', 'tid'], anameList=['speed'], calL=True, calN=False):
    """
    [获取临近点属性 如速度、经纬度等]

    :param A: [轨迹数据]
    :type A: [type]
    :param by: [groupby参数], defaults to ['fid','tid']
    :type by: str, optional
    :param anameList: [属性列名], defaults to 'speed'
    :type anameList: str, optional
    :param calL: [获取前一点], defaults to True
    :type calL: bool, optional
    :param calN: [获取后一点], defaults to False
    :type calN: bool, optional
    :return: [A]
    :rtype: [type]
    """
    lnames = []
    nnames = []
    for aname in anameList:
        if calL == True:
            alastName = aname+'Last'
            lnames.append(alastName)
            A[alastName] = A.groupby(by)[aname].shift(1)
        if calN == True:
            anextName = aname+'Next'
            nnames.append(anextName)
            A[anextName] = A.groupby(by)[aname].shift(-1)
    return A, lnames, nnames


def calVCR(A, by=['fid', 'tid'], vname='speed'):
    """
    [计算VCR]

    :param A: [description]
    :type A: [type]
    :param by: [description], defaults to ['fid','tid']
    :type by: str, optional
    """
    A, vl, _ = getNearAtrr(A, by, anameList=[vname])
    vcrName = vname+'VCR'
    A[vcrName] = A.progress_apply(calvcr, axis=1)
    A = A.drop(vl, axis=1)
    return A


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


def genNewspeed(A, vlimt=[5, 120], minAcc=50, velocityName='velocity', ws=0.5, wv=0.5, newSpeedName='newspeed'):
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
    A[newSpeedName] = A.apply(lambda a: (a[velocityName]*wv+a.speed*ws) if (a.type ==
                                                                            0 and a[velocityName] > 0 and a.acc <= minAcc) else a.speed, axis=1)  # 融合
    A = vFilter2(A, newSpeedName)  # 筛一遍新速度
    return A


def calV_link(roadpath='../../output/paper/武汉路网/WUHAN_new.csv',
              osmpath='../../output/paper/轨迹_按osmid/',
              version=0,
              bufR=0.00003,
              minN=50,
              dropWifi=False,
              filename='tra_bylinkid_v',
              wfilename='数据质量问题路段',
              velocityName='velocity',
              outfile='',
              woutfile='',
              vlimit=[5, 120]):
    """
    [按路段分组并计算速度]

    :param roadpath: [路网数据位置], defaults to '../../output/paper/武汉路网/WUHAN_new.csv'
    :type roadpath: str, optional
    :param osmpath: [轨迹数据位置], defaults to '../../output/paper/轨迹_按osmid/'
    :type osmpath: str, optional
    :param version: [版本], defaults to 0,{0:保留原始speed+wifi重新计算,1:全部重新计算,2:高acc保留+低acc和计算取均值}
    :type version: int, optional
    :param bufR: [缓冲区半径，过大会导致相邻路段错分], defaults to 0.00001
    :type bufR: float, optional
    :param minN: [路段最小点数，数量过少路段不输出并记录至错误日志], defaults to 50
    :type minN: int, optional
    :param dropWifi: [是否删除wifi定位点], defaults to False
    :type dropWifi: bool, optional
    :param filename: [输出文件夹名], defaults to '轨迹_按link_id_v'
    :type filename: str, optional
    :param wfilename: [错误日志文件名], defaults to '数据质量问题路段'
    :type wfilename: str, optional
    :param velocityName: [计算速度名], defaults to 'velocity'，{'velocity':2点计算,'velocity1':3点计算}
    :type velocityName: str, optional
    """
    if outfile == '':
        outfile = '../../output/paper/'
    if woutfile == '':
        woutfile = '../../output/paper/'
    road_all = pd.read_csv(roadpath)
    files = tu.getFileNames(osmpath, postfix='.csv')
    wlist = []
    for i in tqdm(range(len(files))):
        rfile = files[i]
        rname = int(rfile.split('/')[-1].split('.')[0])
        tmpTra = pd.read_csv(rfile)
        if dropWifi == True:
            tmpTra = tmpTra[tmpTra.type == 0]
            tmpTra.reset_index(drop=True, inplace=True)
        try:
            speedversion = 'speed%s' % version
            if version == 0:
                tmpTra[speedversion] = tmpTra.apply(
                    lambda a: a[velocityName] if a.type == 1 else a.speed, axis=1)
            if version == 1:
                tmpTra[speedversion] = tmpTra.apply(
                    lambda a: a[velocityName], axis=1)
            if version == 2:
                tmpTra = genNewspeed(
                    tmpTra, velocityName=velocityName, newSpeedName=speedversion)
            tmpTra = tu.vFilter(tmpTra, vlimt=vlimit, speedName=speedversion)
            tmpRd = road_all[road_all.osm_id == rname]
            tmpRd = tmpRd.reset_index(drop=True)
            for j in range(len(tmpRd.link_id)):
                shp = loads(tmpRd.loc[j, 'geometry'])
                link_id = tmpRd.loc[j, 'link_id']
                label = tmpRd.loc[j, 'maxspeed']
                buf = shp.buffer(bufR, cap_style=2)
                if not label in [20, 30, 40, 50, 60, 70, 80]:  # 暂时不考虑10、15、25
                    continue
                tmp = tmpTra[tmpTra.apply(lambda x:buf.contains(
                    Point(x['lon'], x['lat'])), axis=1)]
                if len(tmp) <= minN:
                    wlist.append([link_id, len(tmp)])
                    continue
                tmp.to_csv(tu.pathCheck(
                    '%s%s%s_Pts_minN=%s_bufR=%s_dropWifi=%s_velocity=%s/maxspeed=%d/' % (outfile, filename, version, minN, bufR, dropWifi, velocityName, label))+'%s.csv' % link_id, index=0)  # TODO version
        except:
            continue
    pd.DataFrame(wlist, columns=['wrong_link_id', 'num_of_data']).to_csv(
        tu.pathCheck('%s%s_%s/' % (woutfile, wfilename, velocityName))+'%s%s.csv' % (filename, version), index=0)


def calV_link_by_tra(pathName):
    """
    [将按点的记录转为按轨迹的记录]

    :param pathName: [按点记录文件路径]
    :type pathName: [type]
    """
    files0 = tu.getFileNames(pathName)
    for f0 in tqdm(files0):
        files1 = tu.getFileNames(f0+'/')
        for f1 in files1:
            newname = f1.replace('Pts', 'Tra')
            tu.pathCheck(newname.rsplit('/', 1)[0])
            tmp = pd.read_csv(f1)
            tmp1 = tmp.groupby(['fid', 'tid']).mean()
            tmp1.hour = tmp1.hour.apply(lambda x: int(x+0.5))
            tmp1.to_csv(newname)


def calV_link_v0(roadpath='../../output/paper/wuhan_road/WUHAN_new.csv',
                 trapath='../../output/paper/velocity/tra_all.csv',
                 version=0,
                 bufR=0.00001,
                 minN=50,
                 dropWifi=False,
                 filename='tra_bylinkid_v',
                 wfilename='数据质量问题路段',
                 velocityName='velocity',
                 outfile='',
                 woutfile='',
                 vlimit=[5, 120],
                 toSingleFile=False):
    """
    [按link分组 新]

    :param roadpath: [路网路径], defaults to '../../output/paper/wuhan_road/WUHAN_new.csv'
    :type roadpath: str, optional
    :param trapath: [轨迹路径], defaults to '../../output/paper/velocity/tra_all.csv'
    :type trapath: str, optional
    :param version: [计算v方法], defaults to 0，{0:保留原始speed+wifi重新计算,1:全部重新计算,2:高acc保留+低acc和计算取均值}
    :type version: int, optional
    :param bufR: [缓冲区半径], defaults to 0.00001
    :type bufR: float, optional
    :param minN: [路段最小点数，数量过少路段不输出并记录至错误日志], defaults to 50
    :type minN: int, optional
    :param dropWifi: [是否剔除wifi], defaults to False
    :type dropWifi: bool, optional
    :param filename: [文件名], defaults to 'tra_bylinkid_v'
    :type filename: str, optional
    :param wfilename: [错误日志文件名], defaults to '数据质量问题路段'
    :type wfilename: str, optional
    :param velocityName: [计算速度名], defaults to 'velocity'，{'velocity':2点计算,'velocity1':3点计算}
    :type velocityName: str, optional
    :param outfile: [输出文件路径], defaults to ''
    :type outfile: str, optional
    :param woutfile: [错误日志路径], defaults to ''
    :type woutfile: str, optional
    :param vlimit: [速度筛选阈值], defaults to [5,120]
    :type vlimit: list, optional
    :param toSingleFile: [是否输出至单个文件], defaults to False
    :type toSingleFile: bool, optional
    """
    if outfile == '':
        outfile = '../../output/paper/'
    if woutfile == '':
        woutfile = '../../output/paper/'
    count = 0
    road_all = pd.read_csv(roadpath)
    data_all = pd.read_csv(trapath)
    data_all.drop(['roadid'], axis=1, inplace=True)  # TODO
    if dropWifi == True:
        data_all = data_all[data_all.type == 0]
        data_all.reset_index(drop=True, inplace=True)
    wlist = []
    for j in tqdm(range(len(road_all))):
        link_id = road_all.loc[j, 'link_id']
        label = road_all.loc[j, 'maxspeed']
        if not label in [20, 30, 40, 50, 60, 70, 80]:  # 暂时不考虑10、15、25
            continue
        shp = loads(road_all.loc[j, 'geometry'])
        lon0, lat0, lon1, lat1 = shp.bounds
        delta = 0.0001
        lon0 -= delta
        lat0 -= delta
        lon1 += delta
        lat1 += delta
        buf = shp.buffer(bufR, cap_style=2)
        tmpTra = data_all[(data_all.lon > lon0) & (data_all.lon <= lon1) & (
            data_all.lat > lat0) & (data_all.lat <= lat1)]
        speedversion = 'speed%s' % version
        if version == 0:
            tmpTra[speedversion] = tmpTra.apply(
                lambda a: a[velocityName] if a.type == 1 else a.speed, axis=1)
        if version == 1:
            tmpTra[speedversion] = tmpTra.apply(
                lambda a: a[velocityName], axis=1)
        if version == 2:
            tmpTra = genNewspeed(
                tmpTra, velocityName=velocityName, newSpeedName=speedversion)
        tmpTra = tu.vFilter(tmpTra, vlimt=vlimit, speedName=speedversion)
        tmp = tmpTra[tmpTra.apply(lambda x:buf.contains(
            Point(x['lon'], x['lat'])), axis=1)]
        if toSingleFile == True:
            tmp['maxspeed'] = label
            tmp['link_id'] = link_id
        if len(tmp) <= minN:
            wlist.append([link_id, len(tmp)])
            continue
        if toSingleFile == True:
            fname = '%s%s_Pts_minN=%s_bufR=%s_dropWifi=%s_velocity=%s' % (
                filename, version, minN, bufR, dropWifi, velocityName)
            if count == 0:
                tmp.to_csv(tu.pathCheck(outfile)+'%s.csv' %
                           fname, mode='a', index=0)
                count += 1
            else:
                tmp.to_csv(tu.pathCheck(outfile)+'%s.csv' %
                           fname, mode='a', index=0, header=None)
        else:
            tmp.to_csv(tu.pathCheck('%s%s%s_Pts_minN=%s_bufR=%s_dropWifi=%s_velocity=%s/maxspeed=%d/' % (outfile,
                                                                                                         filename, version, minN, bufR, dropWifi, velocityName, label))+'%s.csv' % link_id, index=0)  # TODO version

    pd.DataFrame(wlist, columns=['wrong_link_id', 'num_of_data']).to_csv(
        tu.pathCheck('%s%s_%s/' % (woutfile, wfilename, velocityName))+'%s%s.csv' % (filename, version), index=0)


def calV_link_aux(
        roadpath='../../output/paper/wuhan_road/WUHAN_new.csv',
        trapath='../../output/paper/velocity/tra_all.csv',
        bufR=0.00001,
        bufRList={},
        filename='',
        outfile='',
        ):
    filename+='bufR=%s'%bufR
    if outfile == '':
        outfile = '../../output/paper/'
    count = 0
    road_all = pd.read_csv(roadpath)
    data_all = pd.read_csv(trapath)
    if 'roadid' in list(data_all):
        data_all.drop(['roadid'], axis=1, inplace=True)  # TODO
    for j in tqdm(range(len(road_all))):
        link_id = road_all.loc[j, 'link_id']
        label = road_all.loc[j, 'maxspeed']
        if not label in [20, 30, 40, 50, 60, 70, 80]:  # 暂时不考虑10、15、25
            continue
        shp = loads(road_all.loc[j, 'geometry'])
        lon0, lat0, lon1, lat1 = shp.bounds
        delta = 0.0001
        lon0 -= delta
        lat0 -= delta
        lon1 += delta
        lat1 += delta
        buf = shp.buffer(bufR, cap_style=2)
        tmpTra = data_all[(data_all.lon > lon0) & (data_all.lon <= lon1) & (
            data_all.lat > lat0) & (data_all.lat <= lat1)]
        tmp = tmpTra[tmpTra.apply(lambda x:buf.contains(Point(x['lon'], x['lat'])), axis=1)]
        tmp['maxspeed'] = label
        tmp['link_id'] = link_id
        if count == 0:
            tmp.to_csv(tu.pathCheck(outfile)+'%s.csv' %filename, mode='a', index=0)
            count += 1
        else:
            tmp.to_csv(tu.pathCheck(outfile)+'%s.csv' % filename, mode='a', index=0, header=None)


def calV_link_v1(trapath='../../output/paper/tra_linkid_aux/bufR=1e-05.csv',
                 version=0,
                 minN=50,
                 minNdist={20:55,30:55,40:55,50:65,60:85,70:95,80:110},
                 dropWifi=False,
                 filename='tra_bylinkid_v',
                 wfilename='problem_road',
                 velocityName='velocity',
                 outfile='',
                 woutfile='',
                 vlimit=[5, 120],
                 toSingleFile=False):
    bufR=trapath.split('bufR=')[-1].split('.')[0]
    if outfile == '':
        outfile = '../../output/paper/'
    if woutfile == '':
        woutfile = '../../output/paper/'
    count = 0
    data_all = pd.read_csv(trapath)
    if dropWifi == True:
        data_all = data_all[data_all.type == 0]
        data_all.reset_index(drop=True, inplace=True)
    wlist = []
    speedversion = 'speed%s' % version
    if version == 0:
        data_all[speedversion] = data_all.apply(
            lambda a: a[velocityName] if a.type == 1 else a.speed, axis=1)
    if version == 1:
        data_all[speedversion] = data_all.apply(
            lambda a: a[velocityName], axis=1)
    if version == 2:
        data_all = genNewspeed(
            data_all, velocityName=velocityName, newSpeedName=speedversion)
    data_all = tu.vFilter(data_all, vlimt=vlimit, speedName=speedversion)
    gs=data_all.groupby('link_id')
    for lid,g in tqdm(gs):
        if toSingleFile == True:
            if len(g) <= minN:
                wlist.append([lid, len(g)])
                continue
            fname = '%s%s_Pts_minN=%s_bufR=%s_dropWifi=%s_velocity=%s' % (
                filename, version, minN, bufR, dropWifi, velocityName)
            if count == 0:
                g.to_csv(tu.pathCheck(outfile)+'%s.csv' %
                           fname, mode='a', index=0)
                count += 1
            else:
                g.to_csv(tu.pathCheck(outfile)+'%s.csv' %
                           fname, mode='a', index=0, header=None)
        else:
            label=g['maxspeed'].unique()[0]
            g.to_csv(tu.pathCheck('%s%s%s_Pts_minN=%s_bufR=%s_dropWifi=%s_velocity=%s/maxspeed=%d/' % (outfile,
                                                                                                         filename, version, minN, bufR, dropWifi, velocityName, label))+'%s.csv' % lid, index=0)  # TODO version

    pd.DataFrame(wlist, columns=['wrong_link_id', 'num_of_data']).to_csv(
        tu.pathCheck('%s%s_%s/' % (woutfile, wfilename, velocityName))+'%s%s.csv' % (filename, version), index=0)
