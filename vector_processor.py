import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any, Tuple
import json

class VectorProcessor:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        self.setup_logging()
        self.load_model()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        try:
            self.logger.info(f"正在載入詞向量模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.logger.info("詞向量模型載入完成")
        except Exception as e:
            self.logger.error(f"載入詞向量模型失敗: {e}")
            raise
    
    def compute_vectors(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        
        try:
            # 使用模型計算詞向量
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            self.logger.info(f"成功計算 {len(texts)} 個文字的詞向量")
            return embeddings
        except Exception as e:
            self.logger.error(f"計算詞向量失敗: {e}")
            raise
    
    def compute_title_vector(self, title: str) -> List[float]:
        if not title or not title.strip():
            return []
        
        try:
            vector = self.compute_vectors([title])
            return vector[0].tolist()
        except Exception as e:
            self.logger.error(f"計算標題詞向量失敗: {title}, 錯誤: {e}")
            return []
    
    def compute_content_vector(self, content: str) -> List[float]:
        if not content or not content.strip():
            return []
        
        try:
            # 如果內容太長，可以分段處理
            max_length = 512  # 模型的最大輸入長度
            if len(content) > max_length:
                # 簡單的分段策略：按句子分割
                sentences = self.split_content(content)
                vectors = []
                for sentence in sentences:
                    if len(sentence.strip()) > 10:  # 過濾太短的句子
                        vector = self.compute_vectors([sentence])
                        vectors.append(vector[0])
                
                if vectors:
                    # 取平均向量
                    avg_vector = np.mean(vectors, axis=0)
                    return avg_vector.tolist()
                else:
                    return []
            else:
                vector = self.compute_vectors([content])
                return vector[0].tolist()
                
        except Exception as e:
            self.logger.error(f"計算內容詞向量失敗: {content[:50]}..., 錯誤: {e}")
            return []
    
    def split_content(self, content: str) -> List[str]:
        # 簡單的分句策略
        import re
        
        # 按句號、問號、驚嘆號分割
        sentences = re.split(r'[。！？]', content)
        
        # 過濾空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def compute_article_vectors(self, title: str, content: str) -> Tuple[List[float], List[float]]:
        title_vector = self.compute_title_vector(title)
        content_vector = self.compute_content_vector(content)
        
        return title_vector, content_vector
    
    def compute_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        if not vector1 or not vector2:
            return 0.0
        
        try:
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            
            # 計算餘弦相似度
            cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return float(cosine_sim)
        except Exception as e:
            self.logger.error(f"計算相似度失敗: {e}")
            return 0.0
    
    def find_similar_articles(self, query_vector: List[float], article_vectors: List[Dict], top_k: int = 10) -> List[Dict]:
        if not query_vector or not article_vectors:
            return []
        
        similarities = []
        
        for article in article_vectors:
            title_sim = self.compute_similarity(query_vector, article.get('title_vector', []))
            content_sim = self.compute_similarity(query_vector, article.get('content_vector', []))
            
            # 綜合相似度（標題權重0.3，內容權重0.7）
            combined_sim = 0.3 * title_sim + 0.7 * content_sim
            
            similarities.append({
                'id': article['id'],
                'title': article.get('title', ''),
                'similarity': combined_sim,
                'title_similarity': title_sim,
                'content_similarity': content_sim
            })
        
        # 按相似度排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def batch_compute_vectors(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        
        for i, article in enumerate(articles):
            try:
                self.logger.info(f"正在處理第 {i+1}/{len(articles)} 篇文章")
                
                title_vector = self.compute_title_vector(article.get('title', ''))
                content_vector = self.compute_content_vector(article.get('content', ''))
                
                results.append({
                    'id': article['id'],
                    'title_vector': title_vector,
                    'content_vector': content_vector
                })
                
            except Exception as e:
                self.logger.error(f"處理文章失敗: {article.get('title', '')}, 錯誤: {e}")
                continue
        
        return results

if __name__ == "__main__":
    # 測試詞向量功能
    processor = VectorProcessor()
    
    # 測試計算詞向量
    test_title = "測試標題"
    test_content = "這是一個測試內容，用來驗證詞向量計算功能是否正常運作。"
    
    title_vector, content_vector = processor.compute_article_vectors(test_title, test_content)
    
    print(f"標題詞向量長度: {len(title_vector)}")
    print(f"內容詞向量長度: {len(content_vector)}")
    print(f"標題詞向量前5個值: {title_vector[:5]}")
    print(f"內容詞向量前5個值: {content_vector[:5]}")
    
    # 測試相似度計算
    similarity = processor.compute_similarity(title_vector, content_vector)
    print(f"標題與內容相似度: {similarity:.4f}") 
