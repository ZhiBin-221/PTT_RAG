#coding=utf-8
import json
import logging
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import pandas as pd
from database_manager import DatabaseManager
from vector_processor import VectorProcessor

class RAGSystem:
    def __init__(self, 
                 taide_model_path: str = "taide/TAIDE-LX-7B-Chat",
                 db_path: str = "ptt_articles.db",
                 vector_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化RAG系統
        
        Args:
            taide_model_path: TAIDE模型路徑
            db_path: 資料庫路徑
            vector_model_name: 詞向量模型名稱
        """
        self.taide_model_path = taide_model_path
        self.db_path = db_path
        self.vector_model_name = vector_model_name
        
        self.tokenizer = None
        self.model = None
        self.db_manager = None
        self.vector_processor = None
        
        self.setup_logging()
        self.load_components()
    
    def setup_logging(self):
        """設定logging"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def load_components(self):
        """載入所有組件"""
        try:
            # 載入TAIDE模型
            self.logger.info("正在載入TAIDE模型...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.taide_model_path, use_fast=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.taide_model_path, 
                device_map="auto", 
                load_in_4bit=True
            )
            self.logger.info("TAIDE模型載入完成")
            
            # 初始化資料庫管理器
            self.logger.info("正在初始化資料庫...")
            self.db_manager = DatabaseManager(self.db_path)
            self.logger.info("資料庫初始化完成")
            
            # 初始化詞向量處理器
            self.logger.info("正在初始化詞向量處理器...")
            self.vector_processor = VectorProcessor(self.vector_model_name)
            self.logger.info("詞向量處理器初始化完成")
            
        except Exception as e:
            self.logger.error(f"載入組件失敗: {e}")
            raise
    
    def search_relevant_articles(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        搜尋相關文章
        
        Args:
            query: 查詢文字
            top_k: 返回前k個最相關的文章
            
        Returns:
            List[Dict]: 相關文章列表
        """
        try:
            # 計算查詢的詞向量
            query_vector = self.vector_processor.compute_title_vector(query)
            
            if not query_vector:
                self.logger.warning("無法計算查詢詞向量")
                return []
            
            # 取得所有文章的詞向量
            articles_df = self.db_manager.get_all_articles()
            
            if articles_df.empty:
                self.logger.warning("資料庫中沒有文章")
                return []
            
            # 準備文章向量資料
            article_vectors = []
            for _, row in articles_df.iterrows():
                if pd.notna(row['title_vector']) and pd.notna(row['content_vector']):
                    try:
                        title_vector = json.loads(row['title_vector'])
                        content_vector = json.loads(row['content_vector'])
                        
                        article_vectors.append({
                            'id': row['id'],
                            'title': row['title'],
                            'title_vector': title_vector,
                            'content_vector': content_vector
                        })
                    except:
                        continue
            
            # 找到相似文章
            similar_articles = self.vector_processor.find_similar_articles(
                query_vector, article_vectors, top_k
            )
            
            # 取得完整文章資訊
            results = []
            for article in similar_articles:
                article_info = self.db_manager.search_articles_by_keyword(
                    article['title'], limit=1
                )
                if article_info:
                    article_info[0]['similarity'] = article['similarity']
                    results.append(article_info[0])
            
            return results
            
        except Exception as e:
            self.logger.error(f"搜尋相關文章失敗: {e}")
            return []
    
    def generate_context(self, relevant_articles: List[Dict[str, Any]]) -> str:
        """
        根據相關文章生成上下文
        
        Args:
            relevant_articles: 相關文章列表
            
        Returns:
            str: 生成的上下文
        """
        if not relevant_articles:
            return ""
        
        # 過濾掉板規/置底/公告類文章
        filtered_articles = [
            article for article in relevant_articles
            if not any(
                kw in article['title'] or kw in article['content']
                for kw in ['板規', '置底', '公告']
            )
        ]
        
        context_parts = []
        context_parts.append("根據以下PTT八卦版文章資訊回答問題（不包含板規/置底/公告）：\n")
        
        for i, article in enumerate(filtered_articles, 1):
            similarity = article.get('similarity', 0)
            context_parts.append(f"文章{i} (相似度: {similarity:.3f}):")
            context_parts.append(f"標題: {article['title']}")
            context_parts.append(f"作者: {article['author']}")
            context_parts.append(f"時間: {article['date']}")
            context_parts.append(f"內容: {article['content'][:500]}...")  # 限制內容長度
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def TAIDE_Chat(self, input_text: str, use_rag: bool = True, top_k: int = 10) -> str:
        """
        使用TAIDE模型進行對話，可選擇是否使用RAG
        
        Args:
            input_text: 輸入文字
            use_rag: 是否使用RAG功能
            top_k: RAG搜尋的文章數量
            
        Returns:
            str: 模型回應
        """
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if use_rag:
                # 使用RAG功能
                self.logger.info("使用RAG功能搜尋相關文章...")
                relevant_articles = self.search_relevant_articles(input_text, top_k)
                
                if relevant_articles:
                    context = self.generate_context(relevant_articles)
                    enhanced_input = f"{context}\n\n問題: {input_text}"
                    self.logger.info(f"找到 {len(relevant_articles)} 篇相關文章")
                else:
                    enhanced_input = input_text
                    self.logger.info("未找到相關文章，使用原始輸入")
            else:
                enhanced_input = input_text
            
            # 準備對話格式
            messages = [
                {"role": "system", "content": "你是一個有用的助手，專門回答關於PTT八卦版文章的問題。"},
                {"role": "user", "content": enhanced_input}
            ]
            
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            model_input = self.tokenizer(text, return_tensors="pt").to(device)
            generated_ids = self.model.generate(
                model_input.input_ids, 
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"TAIDE對話失敗: {e}")
            return f"抱歉，處理您的問題時發生錯誤: {str(e)}"
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        取得系統統計資訊
        
        Returns:
            Dict: 系統統計資訊
        """
        try:
            db_stats = self.db_manager.get_statistics()
            
            return {
                'database': db_stats,
                'model_info': {
                    'taide_model': self.taide_model_path,
                    'vector_model': self.vector_model_name
                }
            }
        except Exception as e:
            self.logger.error(f"取得系統統計失敗: {e}")
            return {}
    
    def interactive_chat(self):
        """互動式聊天介面"""
        print("歡迎使用PTT八卦版RAG系統！")
        print("輸入 'exit' 退出，輸入 'stats' 查看統計資訊")
        print("-" * 50)
        import re
        while True:
            try:
                user_input = input("您: ")
                if user_input.lower() == 'exit':
                    print("再見！")
                    break
                elif user_input.lower() == 'stats':
                    stats = self.get_system_statistics()
                    print("\n系統統計資訊:")
                    print(f"總文章數: {stats.get('database', {}).get('total_articles', 0)}")
                    print(f"有詞向量的文章數: {stats.get('database', {}).get('articles_with_vectors', 0)}")
                    print(f"今日新增文章數: {stats.get('database', {}).get('today_articles', 0)}")
                    print("-" * 50)
                    continue
                print("正在處理您的問題...")
                # 嘗試解析"N篇"需求
                match = re.search(r'(\d+)\s*篇', user_input)
                top_k = int(match.group(1)) if match else 10
                response = self.TAIDE_Chat(user_input, top_k=top_k)
                print(f"TAIDE: {response}")
                print("-" * 50)
            except KeyboardInterrupt:
                print("\n再見！")
                break
            except Exception as e:
                print(f"發生錯誤: {e}")
                print("-" * 50)
    
    def close(self):
        """關閉系統"""
        if self.db_manager:
            self.db_manager.close()
        self.logger.info("RAG系統已關閉")

if __name__ == "__main__":
    # 測試RAG系統
    rag_system = RAGSystem()
    
    # 測試對話
    test_questions = [
        "最近PTT八卦版有什麼熱門話題？",
        "有沒有人討論天氣？",
        "八卦版今天有什麼新聞？"
    ]
    
    for question in test_questions:
        print(f"問題: {question}")
        response = rag_system.TAIDE_Chat(question)
        print(f"回答: {response}")
        print("-" * 50)
    
    # 互動式聊天
    # rag_system.interactive_chat()
    
    rag_system.close() 