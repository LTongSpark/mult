#-*-encoding:utf-8-*-
# 导入库
import requests  # 用于发出HTML请求
from bs4 import BeautifulSoup  # 用于HTML格式化处理
import re  # 用于解析HTML配合查找条件
import time  # 用于文件名保存
import pandas as pd  # 格式化数据


# %%
class WebParse:
    # 初始化对象
    def __init__(self, headers):
        self.headers = headers
        self.article_list = []
        self.home_page = 'http://www.dataivy.cn/'
        self.nav_page = 'http://www.dataivy.cn/page/{0}/'
        self.art_title = None
        self.art_time = None
        self.art_cat = None
        self.art_tags = None

    # 获取页面数量
    def get_max_page_number(self):
        res = requests.get(self.home_page, headers=self.headers)  # 发送请求
        html = res.text  # 获得请求中的返回文本信息
        html_soup = BeautifulSoup(html, "html.parser")  # 建立soup对象
        page_num_code = html_soup.findAll('a', attrs={"class": "page-numbers"})
        num_sets = [re.findall(r'(\d+)', i.text)
                    for i in page_num_code]  # 获得页面字符串类别
        num_int = [int(i[0]) for i in num_sets if len(i) > 0]  # 获得数值页码
        return max(num_int)  # 最大页码

    # 获得文章列表
    def find_all_articles(self, i):
        url = self.nav_page.format(i)
        res = requests.get(url, headers=headers)  # 发送请求
        html = res.text  # 获得请求中的返回文本信息
        html_soup = BeautifulSoup(html, "html.parser")  # 建立soup对象，用于处理HTML
        self.article_list = html_soup.findAll('article')

    # 解析单文章
    def parse_single_article(self, article):
        self.art_title = article.find('h2', attrs={"class": "entry-title"}).text
        self.art_time = article.find('time', attrs={
            "class": {"entry-date published", "entry-date published updated"}}).text
        self.art_cat = article.find('a', attrs={"rel": "category tag"}).text
        tags_code = article.find('span', attrs={"class": "tags-links"})
        self.art_tags = '' if tags_code is None else WebParse._parse_tags(self, tags_code)

    # 内部用解析tag函数
    def _parse_tags(self, tags_code):
        tag_strs = ''
        for i in tags_code.findAll('a'):
            tag_strs = tag_strs + '/' + i.text
        return tag_strs

    # 格式化数据
    def format_data(self):
        return [self.art_title,
                self.art_time,
                self.art_cat,
                self.art_tags]


# %%
if __name__ == '__main__':
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36'}  # 定义头信息
    data_cols = ['title', 'time', 'cat', 'tags']
    app = WebParse(headers)
    max_num = app.get_max_page_number()
    data_list = []
    for ind in range(max_num):
        app.find_all_articles(ind + 1)  # ind从0开始，因此要加1
        for article in app.article_list:
            app.parse_single_article(article)
            data_list.append(app.format_data())
    data_pd = pd.DataFrame(data_list, columns=data_cols)
    print(data_pd.head(2))