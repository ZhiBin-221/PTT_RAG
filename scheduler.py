import schedule
import time
import logging
from datetime import datetime, timedelta
import threading
from typing import Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ptt_crawler import PTTCrawler
from database_manager import DatabaseManager
from vector_processor import VectorProcessor

class PTTScheduler:
    def __init__(self, 
                 db_path: str = "ptt_articles.db",
                 pages_to_crawl: int = 30,
                 vector_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):

        self.db_path = db_path#db_path: 資料庫路徑
        self.pages_to_crawl = pages_to_crawl  #每次爬取的頁數
        self.vector_model_name = vector_model_name #詞向量模型名稱
        
        self.crawler = None
        self.db_manager = None
        self.vector_processor = None
        self.is_running = False
        
        self.setup_logging()
        self.init_components()
    
    def setup_logging(self):
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"ptt_scheduler_{datetime.now().strftime('%Y%m%d')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def init_components(self):
        try:
            self.logger.info("正在初始化PTT排程器組件...")
            
            # 初始化爬蟲
            self.crawler = PTTCrawler()
            self.logger.info("PTT爬蟲初始化完成")
            
            # 初始化資料庫管理器
            self.db_manager = DatabaseManager(self.db_path)
            self.logger.info("資料庫管理器初始化完成")
            
            # 初始化詞向量處理器
            self.vector_processor = VectorProcessor(self.vector_model_name)
            self.logger.info("詞向量處理器初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化組件失敗: {e}")
            raise
    
    def daily_crawl_task(self):
        try:
            self.logger.info("開始執行每日爬取任務")
            start_time = datetime.now()
            
            # 1. 爬取PTT文章
            self.logger.info(f"開始爬取PTT八卦版前{self.pages_to_crawl}頁文章")
            articles = self.crawler.crawl_daily_articles(pages=self.pages_to_crawl)
            
            if not articles:
                self.logger.warning("未爬取到任何文章")
                return
            
            self.logger.info(f"成功爬取 {len(articles)} 篇文章")
            
            # 2. 寫入資料庫
            self.logger.info("開始寫入資料庫...")
            inserted_count = self.db_manager.insert_articles(articles)
            self.logger.info(f"成功寫入 {inserted_count} 篇新文章到資料庫")
            
            # 3. 計算詞向量
            self.logger.info("開始計算詞向量...")
            articles_without_vectors = self.db_manager.get_articles_without_vectors()
            
            if articles_without_vectors:
                self.logger.info(f"需要計算詞向量的文章數: {len(articles_without_vectors)}")
                
                # 批次計算詞向量
                vector_results = self.vector_processor.batch_compute_vectors(articles_without_vectors)
                
                # 更新資料庫中的詞向量
                for result in vector_results:
                    self.db_manager.update_vectors(
                        result['id'],
                        result['title_vector'],
                        result['content_vector']
                    )
                
                self.logger.info(f"成功計算並更新 {len(vector_results)} 篇文章的詞向量")
            else:
                self.logger.info("所有文章都已計算詞向量")
            
            # 4. 清理舊文章（可選）
            # self.db_manager.cleanup_old_articles(days=30)
            
            end_time = datetime.now()
            duration = end_time - start_time
            self.logger.info(f"每日爬取任務完成，耗時: {duration}")
            
            # 5. 輸出統計資訊
            stats = self.db_manager.get_statistics()
            self.logger.info(f"資料庫統計: 總文章數={stats['total_articles']}, "
                           f"有詞向量文章數={stats['articles_with_vectors']}, "
                           f"今日新增={stats['today_articles']}")
            
        except Exception as e:
            self.logger.error(f"每日爬取任務失敗: {e}")
    
    def manual_crawl(self, pages: Optional[int] = None):
        if pages is None:
            pages = self.pages_to_crawl
        
        self.logger.info(f"手動執行爬取任務，爬取 {pages} 頁")
        self.daily_crawl_task()
    
    def setup_schedule(self):
        # 每日零時執行爬蟲
        schedule.every().day.at("00:00").do(self.daily_crawl_task)
        
        # 也可以設定其他時間，例如：
        # schedule.every().hour.do(self.daily_crawl_task)  # 每小時執行
        # schedule.every().monday.at("00:00").do(self.daily_crawl_task)  # 每週一零時執行
        
        self.logger.info("排程設定完成：每日零時自動執行爬取任務")
    
    def start_scheduler(self):
        if self.is_running:
            self.logger.warning("排程器已在運行中")
            return
        
        self.is_running = True
        self.setup_schedule()
        
        self.logger.info("PTT排程器已啟動")
        self.logger.info("按 Ctrl+C 停止排程器")
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # 每分鐘檢查一次
                
        except KeyboardInterrupt:
            self.logger.info("收到停止信號，正在關閉排程器...")
            self.stop_scheduler()
    
    def start_scheduler_background(self):
        scheduler_thread = threading.Thread(target=self.start_scheduler, daemon=True)
        scheduler_thread.start()
        self.logger.info("PTT排程器已在背景啟動")
        return scheduler_thread
    
    def stop_scheduler(self):
        self.is_running = False
        schedule.clear()
        self.logger.info("PTT排程器已停止")
    
    def get_next_run_time(self) -> Optional[datetime]:
        jobs = schedule.get_jobs()
        if jobs:
            return jobs[0].next_run
        return None
    
    def get_schedule_info(self) -> dict:
        jobs = schedule.get_jobs()
        next_run = self.get_next_run_time()
        
        return {
            'is_running': self.is_running,
            'next_run_time': next_run.isoformat() if next_run else None,
            'job_count': len(jobs),
            'pages_to_crawl': self.pages_to_crawl
        }
    
    def close(self):
        self.stop_scheduler()
        if self.db_manager:
            self.db_manager.close()
        self.logger.info("PTT排程器已關閉")

def main():
    print("PTT八卦版自動爬取排程器")
    print("=" * 50)
    
    # 建立排程器
    scheduler = PTTScheduler()
    
    try:
        # 詢問是否立即執行一次爬取
        choice = input("是否立即執行一次爬取任務？(y/n): ").lower().strip()
        if choice == 'y':
            scheduler.manual_crawl()
        
        # 詢問是否啟動自動排程
        choice = input("是否啟動自動排程（每日零時執行）？(y/n): ").lower().strip()
        if choice == 'y':
            scheduler.start_scheduler()
        else:
            print("排程器未啟動，程式結束")
            
    except KeyboardInterrupt:
        print("\n程式被中斷")
    except Exception as e:
        print(f"發生錯誤: {e}")
    finally:
        scheduler.close()

if __name__ == "__main__":
    main() 
