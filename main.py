#coding=utf-8
import os
import sys
import logging
from datetime import datetime
import argparse
from typing import Optional

# 添加當前目錄到Python路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ptt_crawler import PTTCrawler
from database_manager import DatabaseManager
from vector_processor import VectorProcessor
from rag_system import RAGSystem
from scheduler import PTTScheduler

class PTTRAGMain:
    def __init__(self, db_path: str = r"C:\Users\BIN\Desktop\政大畢業\PTT_RAG_System_Output\ptt_articles.db"):
        """
        初始化PTT RAG主系統
        
        Args:
            db_path: 資料庫路徑
        """
        self.db_path = db_path
        self.setup_logging()
        
        # 組件將在需要時初始化
        self.crawler = None
        self.db_manager = None
        self.vector_processor = None
        self.rag_system = None
        self.scheduler = None
    
    def setup_logging(self):
        """設定logging"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"ptt_rag_main_{datetime.now().strftime('%Y%m%d')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def init_crawler(self):
        """初始化爬蟲"""
        if self.crawler is None:
            self.crawler = PTTCrawler()
            self.logger.info("PTT爬蟲初始化完成")
        return self.crawler
    
    def init_database(self):
        """初始化資料庫"""
        if self.db_manager is None:
            self.db_manager = DatabaseManager(self.db_path)
            self.logger.info("資料庫管理器初始化完成")
        return self.db_manager
    
    def init_vector_processor(self):
        """初始化詞向量處理器"""
        if self.vector_processor is None:
            self.vector_processor = VectorProcessor()
            self.logger.info("詞向量處理器初始化完成")
        return self.vector_processor
    
    def init_rag_system(self):
        """初始化RAG系統"""
        if self.rag_system is None:
            self.rag_system = RAGSystem(db_path=self.db_path)
            self.logger.info("RAG系統初始化完成")
        return self.rag_system
    
    def init_scheduler(self):
        """初始化排程器"""
        if self.scheduler is None:
            self.scheduler = PTTScheduler(db_path=self.db_path)
            self.logger.info("排程器初始化完成")
        return self.scheduler
    
    def crawl_articles(self, pages: int = 30):
        """爬取文章並儲存到資料庫"""
        try:
            self.logger.info(f"開始爬取PTT八卦版前{pages}頁文章")
            
            crawler = self.init_crawler()
            articles = crawler.crawl_daily_articles(pages=pages)
            
            if not articles:
                self.logger.warning("未爬取到任何文章")
                return 0
            
            self.logger.info(f"成功爬取 {len(articles)} 篇文章")
            
            # 自動儲存到資料庫
            inserted_count = self.save_to_database(articles)
            self.logger.info(f"成功儲存 {inserted_count} 篇新文章到資料庫")
            
            return inserted_count
            
        except Exception as e:
            self.logger.error(f"爬取文章失敗: {e}")
            return 0
    
    def save_to_database(self, articles):
        """儲存文章到資料庫"""
        try:
            self.logger.info("開始儲存文章到資料庫")
            
            db_manager = self.init_database()
            inserted_count = db_manager.insert_articles(articles)
            
            self.logger.info(f"成功儲存 {inserted_count} 篇新文章到資料庫")
            return inserted_count
            
        except Exception as e:
            self.logger.error(f"儲存到資料庫失敗: {e}")
            return 0
    
    def compute_vectors(self):
        """計算詞向量"""
        try:
            self.logger.info("開始計算詞向量")
            
            db_manager = self.init_database()
            vector_processor = self.init_vector_processor()
            
            articles_without_vectors = db_manager.get_articles_without_vectors()
            
            if not articles_without_vectors:
                self.logger.info("所有文章都已計算詞向量")
                return 0
            
            self.logger.info(f"需要計算詞向量的文章數: {len(articles_without_vectors)}")
            
            # 批次計算詞向量
            vector_results = vector_processor.batch_compute_vectors(articles_without_vectors)
            
            # 更新資料庫中的詞向量
            for result in vector_results:
                db_manager.update_vectors(
                    result['id'],
                    result['title_vector'],
                    result['content_vector']
                )
            
            self.logger.info(f"成功計算並更新 {len(vector_results)} 篇文章的詞向量")
            return len(vector_results)
            
        except Exception as e:
            self.logger.error(f"計算詞向量失敗: {e}")
            return 0
    
    def full_pipeline(self, pages: int = 30):
        """執行完整流程：爬取 -> 儲存 -> 計算詞向量"""
        try:
            self.logger.info("開始執行完整流程")
            start_time = datetime.now()
            
            # 1. 爬取文章
            crawler = self.init_crawler()
            articles = crawler.crawl_daily_articles(pages=pages)
            
            if not articles:
                self.logger.warning("未爬取到任何文章")
                return
            
            # 2. 儲存到資料庫
            inserted_count = self.save_to_database(articles)
            
            # 3. 計算詞向量
            vector_count = self.compute_vectors()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info(f"完整流程執行完成")
            self.logger.info(f"爬取文章數: {len(articles)}")
            self.logger.info(f"新增到資料庫: {inserted_count}")
            self.logger.info(f"計算詞向量: {vector_count}")
            self.logger.info(f"總耗時: {duration}")
            
        except Exception as e:
            self.logger.error(f"完整流程執行失敗: {e}")
    
    def start_chat(self, top_k: int = 10):
        """啟動聊天介面，支援指定top_k"""
        try:
            self.logger.info("啟動RAG聊天介面")
            rag_system = self.init_rag_system()
            rag_system.interactive_chat()
        except Exception as e:
            self.logger.error(f"啟動聊天介面失敗: {e}")
    
    def start_scheduler(self):
        """啟動自動排程"""
        try:
            self.logger.info("啟動自動排程")
            scheduler = self.init_scheduler()
            scheduler.start_scheduler()
        except Exception as e:
            self.logger.error(f"啟動排程器失敗: {e}")
    
    def show_statistics(self):
        """顯示系統統計資訊"""
        try:
            db_manager = self.init_database()
            stats = db_manager.get_statistics()
            
            print("\n" + "="*50)
            print("PTT八卦版RAG系統統計資訊")
            print("="*50)
            print(f"總文章數: {stats['total_articles']}")
            print(f"有詞向量的文章數: {stats['articles_with_vectors']}")
            print(f"今日新增文章數: {stats['today_articles']}")
            print(f"詞向量完成率: {stats['articles_with_vectors']/stats['total_articles']*100:.1f}%" if stats['total_articles'] > 0 else "0%")
            
            if stats['top_authors']:
                print("\n熱門作者 (前5名):")
                for i, (author, count) in enumerate(stats['top_authors'][:5], 1):
                    print(f"  {i}. {author}: {count} 篇")
            
            print("="*50)
            
        except Exception as e:
            self.logger.error(f"取得統計資訊失敗: {e}")
    
    def search_articles(self, keyword: str, limit: int = 10):
        """搜尋文章"""
        try:
            db_manager = self.init_database()
            articles = db_manager.search_articles_by_keyword(keyword, limit)
            
            print(f"\n搜尋關鍵字 '{keyword}' 的結果:")
            print("-" * 50)
            
            for i, article in enumerate(articles, 1):
                print(f"{i}. {article['title']}")
                print(f"   作者: {article['author']} | 時間: {article['date']}")
                print(f"   內容: {article['content'][:100]}...")
                print()
            
        except Exception as e:
            self.logger.error(f"搜尋文章失敗: {e}")
    
    def close(self):
        """關閉系統"""
        if self.db_manager:
            self.db_manager.close()
        if self.rag_system:
            self.rag_system.close()
        if self.scheduler:
            self.scheduler.close()
        self.logger.info("PTT RAG主系統已關閉")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="PTT八卦版RAG系統")
    parser.add_argument("--action", choices=["crawl", "vectors", "chat", "scheduler", "stats", "search", "full"], 
                       help="執行動作")
    parser.add_argument("--pages", type=int, default=30, help="爬取頁數")
    parser.add_argument("--keyword", type=str, help="搜尋關鍵字")
    parser.add_argument("--limit", type=int, default=10, help="搜尋結果數量限制")
    
    args = parser.parse_args()
    
    # 建立主系統
    main_system = PTTRAGMain()
    
    try:
        if args.action == "crawl":
            # 只爬取文章
            main_system.crawl_articles(args.pages)
            
        elif args.action == "vectors":
            # 只計算詞向量
            main_system.compute_vectors()
            
        elif args.action == "chat":
            # 啟動聊天介面
            main_system.start_chat()
            
        elif args.action == "scheduler":
            # 啟動排程器
            main_system.start_scheduler()
            
        elif args.action == "stats":
            # 顯示統計資訊
            main_system.show_statistics()
            
        elif args.action == "search":
            # 搜尋文章
            if not args.keyword:
                print("請提供搜尋關鍵字: --keyword")
                return
            main_system.search_articles(args.keyword, args.limit)
            
        elif args.action == "full":
            # 執行完整流程
            main_system.full_pipeline(args.pages)
            
        else:
            # 互動式選單
            show_interactive_menu(main_system)
            
    except KeyboardInterrupt:
        print("\n程式被中斷")
    except Exception as e:
        print(f"發生錯誤: {e}")
    finally:
        main_system.close()

def show_interactive_menu(main_system):
    """顯示互動式選單"""
    while True:
        print("\n" + "="*50)
        print("PTT八卦版RAG系統")
        print("="*50)
        print("1. 爬取PTT文章")
        print("2. 計算詞向量")
        print("3. 執行完整流程")
        print("4. 啟動聊天介面")
        print("5. 啟動自動排程")
        print("6. 顯示統計資訊")
        print("7. 搜尋文章")
        print("0. 退出")
        print("="*50)
        
        try:
            choice = input("請選擇操作 (0-7): ").strip()
            
            if choice == "0":
                print("再見！")
                break
            elif choice == "1":
                pages = input("請輸入爬取頁數 (預設30): ").strip()
                pages = int(pages) if pages.isdigit() else 30
                main_system.crawl_articles(pages)
            elif choice == "2":
                main_system.compute_vectors()
            elif choice == "3":
                pages = input("請輸入爬取頁數 (預設30): ").strip()
                pages = int(pages) if pages.isdigit() else 30
                main_system.full_pipeline(pages)
            elif choice == "4":
                main_system.start_chat()
            elif choice == "5":
                main_system.start_scheduler()
            elif choice == "6":
                main_system.show_statistics()
            elif choice == "7":
                keyword = input("請輸入搜尋關鍵字: ").strip()
                if keyword:
                    limit = input("請輸入結果數量限制 (預設10): ").strip()
                    limit = int(limit) if limit.isdigit() else 10
                    main_system.search_articles(keyword, limit)
                else:
                    print("關鍵字不能為空")
            else:
                print("無效選擇，請重新輸入")
                
        except KeyboardInterrupt:
            print("\n程式被中斷")
            break
        except Exception as e:
            print(f"發生錯誤: {e}")

if __name__ == "__main__":
    main() 