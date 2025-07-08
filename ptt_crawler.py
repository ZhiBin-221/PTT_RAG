#coding=utf-8
import requests
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime
import logging

class PTTCrawler:
    def __init__(self):
        self.base_url = "https://www.ptt.cc"
        self.gossiping_url = "https://www.ptt.cc/bbs/Gossiping/index.html"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # 設定logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def get_page_content(self, url):
        """取得頁面內容"""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            self.logger.error(f"取得頁面失敗: {url}, 錯誤: {e}")
            return None
    
    def parse_article_list(self, html_content):
        """解析文章列表頁面"""
        soup = BeautifulSoup(html_content, 'html.parser')
        articles = []
        
        # 找到所有文章項目
        article_items = soup.find_all('div', class_='r-ent')
        
        for item in article_items:
            try:
                # 取得標題和連結
                title_element = item.find('div', class_='title')
                if not title_element or not title_element.find('a'):
                    continue
                    
                title_link = title_element.find('a')
                title = title_link.get_text(strip=True)
                article_url = self.base_url + title_link['href']
                
                # 取得作者
                author_element = item.find('div', class_='author')
                author = author_element.get_text(strip=True) if author_element else "匿名"
                
                # 取得時間
                date_element = item.find('div', class_='date')
                date = date_element.get_text(strip=True) if date_element else ""
                
                # 跳過已刪除的文章
                if title.startswith('(本文已被刪除)'):
                    continue
                
                articles.append({
                    'title': title,
                    'author': author,
                    'date': date,
                    'url': article_url
                })
                
            except Exception as e:
                self.logger.error(f"解析文章項目失敗: {e}")
                continue
        
        return articles
    
    def parse_article_content(self, html_content):
        """解析文章內容"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 找到文章內容區域
        main_content = soup.find('div', id='main-content')
        if not main_content:
            return None
        
        # 移除不需要的元素
        elements_to_remove = main_content.find_all(['div', 'span'], class_=['article-metaline', 'article-metaline-right', 'push'])
        for element in elements_to_remove:
            if hasattr(element, 'decompose'):
                element.decompose()
        
        # 取得純文字內容
        content = main_content.get_text(strip=True)
        
        # 清理內容
        content = re.sub(r'\n+', '\n', content)
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()
    
    def crawl_daily_articles(self, pages=30):
        """爬取每日前N頁文章"""
        self.logger.info(f"開始爬取PTT八卦版前{pages}頁文章")
        
        all_articles = []
        current_url = self.gossiping_url
        
        for page in range(pages):
            self.logger.info(f"正在爬取第{page + 1}頁: {current_url}")
            
            # 取得頁面內容
            html_content = self.get_page_content(current_url)
            if not html_content:
                continue
            
            # 解析文章列表
            articles = self.parse_article_list(html_content)
            
            # 爬取每篇文章的詳細內容
            for article in articles:
                try:
                    self.logger.info(f"正在爬取文章: {article['title']}")
                    
                    # 取得文章內容
                    article_html = self.get_page_content(article['url'])
                    if article_html:
                        content = self.parse_article_content(article_html)
                        if content:
                            article['content'] = content
                            all_articles.append(article)
                    
                    # 避免請求過於頻繁
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"爬取文章內容失敗: {article['title']}, 錯誤: {e}")
                    continue
            
            # 找到下一頁連結
            soup = BeautifulSoup(html_content, 'html.parser')
            prev_link = soup.find('a', string='‹ 上頁')
            if prev_link and prev_link.get('href'):
                current_url = self.base_url + prev_link.get('href')
            else:
                break
            
            # 避免請求過於頻繁
            time.sleep(2)
        
        self.logger.info(f"爬取完成，共取得{len(all_articles)}篇文章")
        return all_articles
    
    def get_today_articles(self):
        """取得今日文章（用於測試）"""
        return self.crawl_daily_articles(pages=1)

if __name__ == "__main__":
    crawler = PTTCrawler()
    articles = crawler.get_today_articles()
    
    for article in articles:
        print(f"標題: {article['title']}")
        print(f"作者: {article['author']}")
        print(f"時間: {article['date']}")
        print(f"內容: {article['content'][:100]}...")
        print("-" * 50) 