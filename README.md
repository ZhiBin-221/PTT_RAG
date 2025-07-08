# PTT八卦版RAG系統

本專案自動化爬取PTT八卦版文章，並結合語意檢索與生成式AI（RAG），提供智慧化的資訊查詢與問答服務。

## 主要功能
- 自動爬取PTT八卦版文章，支援自訂頁數與去重
- SQLite資料庫儲存與檢索，支援全文搜尋與統計
- 使用Sentence Transformers計算標題/內容詞向量
- 整合TAIDE-LX-7B-Chat大型語言模型，實現RAG問答
- 支援每日自動排程、批次處理與日誌記錄

## 專案結構
```
PTT_RAG_System_Output/
├── ptt_crawler.py         # PTT爬蟲
├── database_manager.py    # 資料庫管理
├── vector_processor.py    # 詞向量計算
├── rag_system.py          # RAG整合
├── scheduler.py           # 排程
├── main.py                # 主程式
├── requirements.txt       # 依賴套件
└── README.md              # 說明文件
```

## 安裝方式
1. 下載專案
```bash
cd <your_path>/PTT_RAG_System_Output
```
2. 安裝依賴
```bash
pip install -r requirements.txt
```

## 使用說明
- 啟動互動式選單：
  ```bash
  python main.py
  ```
- 常用指令：
  - 爬取文章：`python main.py --action crawl --pages 30`
  - 計算詞向量：`python main.py --action vectors`
  - 啟動聊天：`python main.py --action chat`
  - 啟動排程：`python main.py --action scheduler`
  - 搜尋文章：`python main.py --action search --keyword "天氣" --limit 10`

## 技術細節
- **語言模型**：TAIDE-LX-7B-Chat
- **詞向量模型**：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **資料庫**：SQLite，內建全文檢索與向量欄位
- **硬體建議**：Python 3.8+，8GB RAM，CUDA GPU

## 常見問題
- CUDA記憶體不足：可切換CPU或減少batch size
- 網路問題/模型下載失敗：請檢查連線或手動下載
- 資料庫錯誤：檢查權限與磁碟空間

## 資安與上傳建議
- 已內建 .gitignore，會自動忽略：
  - 資料庫檔案（如 .db, .sqlite3, ptt_articles.db）
  - 日誌檔案（logs/、*.log）
  - 個人化/暫存/快取檔（__pycache__/、.vscode/、.idea/ 等）
  - 機器學習模型檔（*.bin, *.pt, *.ckpt 等）
  - 環境與密鑰檔（.env, config.json, config.yaml 等）
  - 你的本地路徑與個人資料夾（如政大畢業/、桌面提示/等）
- 請勿將任何密碼、API金鑰、個人資料、真實資料庫檔案上傳至GitHub。
- 若需分享設定，請提供範本（如 .env.example）。
- 上傳前請再次檢查敏感資訊是否已排除。

## 貢獻與授權
- 僅供學術研究與學習用途，請遵守PTT及相關網站規範
- 歡迎提交Issue與PR改善專案 