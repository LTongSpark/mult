#-*-encoding:utf-8-*-
import gzip
import re
import time
import pandas as pd
#%%
# 判断是否为爬虫记录
def is_spider(log_record,spiders):
    detect_result = [True if log_record.find(spider) == -1 else False for spider in spiders]
    is_exist = True if all(detect_result) else False
    return is_exist
#%%
# 判断是否为UA记录
def is_ua_record(log_record):
    is_ua = True if log_record.find('GET /__ua.gif?') != -1 else False
    return is_ua
#%%
# 解析每条日志数据
def split_ua_data(line):
    # 定义不同日志分割的正则表达式
    ip_pat = '[\d.]*'  # 定义IP规则，例如203.208.60.230
    time_pat = '\[[^\[\]]*\]'  # 定义时间规则，例如[02/Mar/2016:14:00:23 +0800]
    request_pat = '\"[^\"]*\"'  # 定义请求规则
    status_pat = '\d+'  # 定义返回的状态码规则，例如200
    bytes_pat = '\d+'  # 返回的字节数，例如326
    refer_pat = '\"[^\"]*\"'  # 定义refer规则
    user_agent_pat = '\"[^\"]*\"'  # 定义user agnet规则
    # 原理：主要通过空格和-来区分各不同项目，各项目内部写各自的匹配表达式
    re_pattern = re.compile('(%s)\ -\ -\ (%s)\ (%s)\ (%s)\ (%s)\ (%s)\ (%s)' % (
        ip_pat, time_pat, request_pat, status_pat, bytes_pat, refer_pat, user_agent_pat),
                                re.VERBOSE)  # 完整表达式模式
    matchs = re_pattern.match(line)  # 匹配
    if matchs != None:  # 如果不为空
        allGroups = matchs.groups()  # 获得所有匹配的列表
        return allGroups[0],allGroups[1],allGroups[2],allGroups[3],allGroups[4],allGroups[5],allGroups[6]
    else: # 否则返回空
        return '','','','','','',''
#%%
# 读取日志数据
def get_ua_data(file,spiders):
    ua_data = []
    with gzip.open(file, 'rt') as fn:  # 打开要读取的日志文件对象
        content = fn.readlines()  # 以列表形式读取日志数据
    for single_log in content:  # 循环判断每天记录
        rule1 = is_spider(single_log,spiders)
        rule2 = is_ua_record(single_log)
        if rule1 and rule2:  # 如果同时符合2条规则，则执行
            ua_data.append(split_ua_data(single_log))
    ua_pd = pd.DataFrame(ua_data)
    return ua_pd
#%%
#主程序
if __name__ == '__main__':
    file = 'dataivy.cn-Feb-2018.gz'  # 定义原始日志的文件名
    spiders = [
    'AhrefsBot',
    'archive.org_bot',
    'baiduspider',
    'Baiduspider',
    'bingbot',
    'DeuSu',
    'DotBot',
    'Googlebot',
    'iaskspider',
    'MJ12bot',
    'msnbot',
    'Slurp',
    'Sogou web spider',
    'Sogou Push Spider',
    'SputnikBot',
    'Yahoo! Slurp China',
    'Yahoo! Slurp',
    'YisouSpider',
    'YodaoBot',
    'bot.html'
]
    ua_pd = get_ua_data(file,spiders)
    ua_pd.columns = ['ip_add','requet_time','request_info','status','bytes_info','referral','ua']
    output_file = 'ua_result_{0}.xlsx'.format(time.strftime('%Y%m%d%H%M%S',time.localtime(time.time())))
    ua_pd.to_excel(output_file, index=False)
    print('excel file {0} generated!'.format(output_file))