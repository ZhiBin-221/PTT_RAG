import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
from typing import List, Dict, Any

class DatabaseManager:
    def __init__(self, db_path: str = "ptt_articles.db"):
        self.db_path = db_path
        self.conn = None
        self.setup_logging()
        self.init_database()
    
    def setup_logging(self):
 
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def init_database(self):

        try:
            self.conn = sqlite3.connect(self.db_path)
            self.create_tables()
            self.logger.info("資料庫初始化完成")
        except Exception as e:
            self.logger.error(f"資料庫初始化失敗: {e}")
            raise
    
    def create_tables(self):
        cursor = self.conn.cursor()
        
        # 建立文章資料表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                author TEXT,
                date TEXT,
                content TEXT,
                url TEXT UNIQUE,
                title_vector TEXT,
                content_vector TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        #索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_url ON articles(url)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_title ON articles(title)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON articles(date)')
        
        self.conn.commit()
        self.logger.info("資料表建立完成")
    
    def insert_articles(self, articles: List[Dict[str, Any]]) -> int:

        if not articles:
            return 0
        
        cursor = self.conn.cursor()
        inserted_count = 0
        
        for article in articles:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO articles 
                    (title, author, date, content, url) 
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    article.get('title', ''),
                    article.get('author', ''),
                    article.get('date', ''),
                    article.get('content', ''),
                    article.get('url', '')
                ))
                
                if cursor.rowcount > 0:
                    inserted_count += 1
                    
            except Exception as e:
                self.logger.error(f"插入文章失敗: {article.get('title', '')}, 錯誤: {e}")
                continue
        
        self.conn.commit()
        self.logger.info(f"成功插入 {inserted_count} 篇新文章")
        return inserted_count
    
    def update_vectors(self, article_id: int, title_vector: List[float], content_vector: List[float]):

        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE articles 
                SET title_vector = ?, content_vector = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (
                json.dumps(title_vector),
                json.dumps(content_vector),
                article_id
            ))
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"更新詞向量失敗: article_id={article_id}, 錯誤: {e}")
    
    def get_articles_without_vectors(self) -> List[Dict[str, Any]]:

        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, title, content 
            FROM articles 
            WHERE title_vector IS NULL OR content_vector IS NULL
        ''')
        
        articles = []
        for row in cursor.fetchall():
            articles.append({
                'id': row[0],
                'title': row[1],
                'content': row[2]
            })
        
        return articles
    
    def get_all_articles(self) -> pd.DataFrame:

        query = '''
            SELECT id, title, author, date, content, url, title_vector, content_vector, created_at
            FROM articles
            ORDER BY created_at DESC
        '''
        return pd.read_sql_query(query, self.conn)
    
    def search_articles_by_keyword(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:

        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, title, author, date, content, url
            FROM articles
            WHERE title LIKE ? OR content LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (f'%{keyword}%', f'%{keyword}%', limit))
        
        articles = []
        for row in cursor.fetchall():
            articles.append({
                'id': row[0],
                'title': row[1],
                'author': row[2],
                'date': row[3],
                'content': row[4],
                'url': row[5]
            })
        
        return articles
    
    def get_articles_by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:

        query = '''
            SELECT id, title, author, date, content, url, created_at
            FROM articles
            WHERE date BETWEEN ? AND ?
            ORDER BY date DESC
        '''
        return pd.read_sql_query(query, self.conn, params=[start_date, end_date])
    
    def get_statistics(self) -> Dict[str, Any]:

        cursor = self.conn.cursor()
        
        # 總文章數
        cursor.execute('SELECT COUNT(*) FROM articles')
        total_articles = cursor.fetchone()[0]
        
        # 有詞向量的文章數
        cursor.execute('SELECT COUNT(*) FROM articles WHERE title_vector IS NOT NULL AND content_vector IS NOT NULL')
        articles_with_vectors = cursor.fetchone()[0]
        
        # 今日新增文章數
        cursor.execute('SELECT COUNT(*) FROM articles WHERE DATE(created_at) = DATE("now")')
        today_articles = cursor.fetchone()[0]
        
        # 作者統計
        cursor.execute('SELECT author, COUNT(*) FROM articles GROUP BY author ORDER BY COUNT(*) DESC LIMIT 10')
        top_authors = cursor.fetchall()
        
        return {
            'total_articles': total_articles,
            'articles_with_vectors': articles_with_vectors,
            'today_articles': today_articles,
            'top_authors': top_authors
        }
    
    def cleanup_old_articles(self, days: int = 30):
        cursor = self.conn.cursor()
        cursor.execute('''
            DELETE FROM articles 
            WHERE created_at < DATE("now", "-{} days")
        '''.format(days))
        
        deleted_count = cursor.rowcount
        self.conn.commit()
        self.logger.info(f"清理了 {deleted_count} 篇舊文章")
        return deleted_count
    
    def close(self):
        if self.conn:
            self.conn.close()
            self.logger.info("資料庫連線已關閉")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == "__main__":
    db = DatabaseManager("test_ptt.db")
    # 測試統計資訊
    stats = db.get_statistics()
    print("資料庫統計資訊:")
    print(f"總文章數: {stats['total_articles']}")
    print(f"有詞向量的文章數: {stats['articles_with_vectors']}")
    print(f"今日新增文章數: {stats['today_articles']}")
    
    db.close() 
