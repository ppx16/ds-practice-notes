import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.lm import MLE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertTokenizer, BertModel
import warnings
warnings.filterwarnings('ignore')



# ----------------------
# 1. 数据加载与预处理  
# ----------------------
def load_data():
    """加载训练集、测试集和原始文章数据"""
    train_df = pd.read_excel("training.xlsx")
    test_df = pd.read_excel("test.xlsx")
    articles_df = pd.read_excel("all_articles.xlsx")
    
    # 关联数据（通过article_id连接原始文章信息）
    train_df = train_df.merge(articles_df, on="article_id", how="left")
    test_df = test_df.merge(articles_df, on="article_id", how="left")
    return train_df, test_df

def text_preprocess(text):
    """文本预处理：清洗、分词、去停用词（针对英文文本）"""
    # 确保输入是字符串
    if pd.isna(text) or text is None:
        return [], ""
    
    text = str(text)
    # 去除特殊字符和数字，转为小写
    text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
    # 分词
    words = word_tokenize(text)
    # 去停用词
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]
    return words, text  # 返回分词结果和清洗后的完整文本

# ----------------------
# 2. 特征工程实现
# ----------------------
class FeatureEngineer:
    def __init__(self):
        # 加载BERT模型（使用英文模型处理英文数据）
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        
        # 简单的语法质量评估（不需要Java）
        # self.lang_tool = language_tool_python.LanguageTool('en-US')  # 移除Java依赖
        
        # 衔接词列表（用于逻辑结构特征）
        self.cohesive_words = {"however", "therefore", "in addition", "nevertheless", "furthermore", "thus", "besides"}
    
    def length_features(self, row):
        """文本长度与比例特征"""
        summary_words, _ = text_preprocess(row["summary_text"])
        article_words, _ = text_preprocess(row["article_text"])
        question_words, _ = text_preprocess(row["summary_question"])
        
        # 基础长度
        summary_len = len(summary_words)
        summary_sent_len = len(sent_tokenize(row["summary_text"]))
        
        # 相对比例
        article_len = len(article_words)
        ratio_vs_article = summary_len / article_len if article_len > 0 else 0
        
        question_len = len(question_words)
        ratio_vs_question = summary_len / question_len if question_len > 0 else 0
        
        return {
            "summary_word_count": summary_len,
            "summary_sent_count": summary_sent_len,
            "ratio_vs_article": ratio_vs_article,
            "ratio_vs_question": ratio_vs_question
        }
    
    def relevance_features(self, row):
        """内容相关性特征（TF-IDF相似度+BERT嵌入相似度）"""
        # TF-IDF相似度（需预先拟合训练集的TF-IDF向量器）
        if not hasattr(self, "tfidf_vectorizer"):
            return {}  # 需在fit阶段拟合
        
        # 提取文本
        summary_text = row["summary_text"]
        article_text = row["article_text"]
        question_text = row["summary_question"]
        
        # TF-IDF向量
        tfidf_summary = self.tfidf_vectorizer.transform([summary_text])
        tfidf_article = self.tfidf_vectorizer.transform([article_text])
        tfidf_question = self.tfidf_vectorizer.transform([question_text])
        
        # 余弦相似度
        sim_article = cosine_similarity(tfidf_summary, tfidf_article)[0][0]
        sim_question = cosine_similarity(tfidf_summary, tfidf_question)[0][0]
        
        # BERT嵌入相似度
        def bert_embedding(text):
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        emb_summary = bert_embedding(summary_text)
        emb_article = bert_embedding(article_text)
        emb_question = bert_embedding(question_text)
        
        bert_sim_article = cosine_similarity(emb_summary, emb_article)[0][0]
        bert_sim_question = cosine_similarity(emb_summary, emb_question)[0][0]
        
        return {
            "tfidf_sim_article": sim_article,
            "tfidf_sim_question": sim_question,
            "bert_sim_article": bert_sim_article,
            "bert_sim_question": bert_sim_question
        }
    
    def language_quality_features(self, row):
        """语言质量特征（语法错误、词汇多样性等）"""
        summary_text = row["summary_text"]
        words, cleaned_text = text_preprocess(summary_text)
        if not words:
            return {"gram_error_count": 0, "ttr": 0, "avg_sent_len": 0, "perplexity": 0}
        
        # 简单的语法质量评估（不需要Java）
        gram_error_count = self.simple_grammar_check(summary_text)
        
        # 词汇多样性（TTR）
        ttr = len(set(words)) / len(words)
        
        # 平均句长
        sents = sent_tokenize(summary_text)
        avg_sent_len = len(words) / len(sents) if sents else 0
        
        # 简单n-gram困惑度（示例用bigram）
        if not hasattr(self, "ngram_model"):
            perplexity = 0  # 需在fit阶段训练
        else:
            perplexity = self.calculate_perplexity(cleaned_text, self.ngram_model)
        
        return {
            "gram_error_count": gram_error_count,
            "ttr": ttr,
            "avg_sent_len": avg_sent_len,
            "perplexity": perplexity
        }
    
    def logic_features(self, row):
        """逻辑结构特征（衔接词频率等）"""
        summary_words, _ = text_preprocess(row["summary_text"])
        # 衔接词频率
        cohesive_count = sum(1 for w in summary_words if w in self.cohesive_words)
        cohesive_ratio = cohesive_count / len(summary_words) if summary_words else 0
        
        return {
            "cohesive_word_ratio": cohesive_ratio
        }
    
    def article_level_features(self, row, train_data=None):
        """文章层面特征（目标编码）"""
        article_id = row["article_id"]
        if train_data is None:
            # 测试集使用训练集的统计结果
            article_stats = self.article_stats.get(article_id, {
                "article_avg_score": 0,
                "article_score_std": 0,
                "article_avg_word_len": 0
            })
        else:
            # 训练集计算统计量
            article_data = train_data[train_data["article_id"] == article_id]
            article_avg_score = article_data["score"].mean() if not article_data.empty else 0
            article_score_std = article_data["score"].std() if not article_data.empty else 0
            
            # 原始文章平均单词长度
            article_text = row["article_text"]
            article_words, _ = text_preprocess(article_text)
            avg_word_len = np.mean([len(w) for w in article_words]) if article_words else 0
            
            article_stats = {
                "article_avg_score": article_avg_score,
                "article_score_std": article_score_std,
                "article_avg_word_len": avg_word_len
            }
        
        return article_stats
    
    def simple_grammar_check(self, text):
        """简单的语法质量评估（不需要Java）"""
        error_count = 0
        
        # 检查常见语法问题
        # 1. 重复单词
        words = text.lower().split()
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                error_count += 1
        
        # 2. 句子开头没有大写
        sentences = sent_tokenize(text)
        for sent in sentences:
            sent = sent.strip()
            if sent and not sent[0].isupper():
                error_count += 1
        
        # 3. 句子过长（超过40个单词可能表示句子结构有问题）
        for sent in sentences:
            words_in_sent = len(word_tokenize(sent))
            if words_in_sent > 40:
                error_count += 1
        
        # 4. 检查基本标点问题
        if text.count('(') != text.count(')'):
            error_count += 1
        if text.count('[') != text.count(']'):
            error_count += 1
        if text.count('{') != text.count('}'):
            error_count += 1
        
        return error_count
    
    def fit(self, train_df):
        """拟合训练集，初始化需要训练的组件（TF-IDF、ngram模型等）"""
        # 拟合TF-IDF向量器
        all_texts = pd.concat([
            train_df["summary_text"],
            train_df["article_text"],
            train_df["summary_question"]
        ])
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        self.tfidf_vectorizer.fit(all_texts)
        
        # 训练n-gram语言模型（用于困惑度计算）
        valid_texts = [text for text in train_df["summary_text"].dropna() if text and len(str(text).strip()) > 0]
        if valid_texts:
            tokenized_texts = []
            for text in valid_texts:
                tokens = word_tokenize(str(text).lower())
                if len(tokens) > 1:  # 至少需要2个token才能训练bigram
                    tokenized_texts.append(tokens)
            
            if tokenized_texts and len(tokenized_texts) > 0:
                try:
                    from nltk.lm.preprocessing import padded_everygram_pipeline
                    train_data, vocab = padded_everygram_pipeline(2, tokenized_texts)
                    self.ngram_model = MLE(2)
                    self.ngram_model.fit(train_data, vocab)
                except Exception as e:
                    print(f"N-gram模型训练失败: {e}")
                    self.ngram_model = None
            else:
                self.ngram_model = None
        else:
            self.ngram_model = None
        
        # 计算文章层面统计量（目标编码）
        self.article_stats = {}
        for article_id in train_df["article_id"].unique():
            article_data = train_df[train_df["article_id"] == article_id]
            # 计算该文章的平均单词长度
            if not article_data.empty:
                word_lengths = []
                for _, row in article_data.iterrows():
                    article_words, _ = text_preprocess(row["article_text"])
                    word_lengths.extend([len(w) for w in article_words])
                avg_word_len = np.mean(word_lengths) if word_lengths else 0
            else:
                avg_word_len = 0
            
            self.article_stats[article_id] = {
                "article_avg_score": article_data["score"].mean() if not article_data.empty else 0,
                "article_score_std": article_data["score"].std() if not article_data.empty else 0,
                "article_avg_word_len": avg_word_len
            }
    
    def transform(self, df, is_train=True):
        """转换数据生成特征"""
        features = []
        for _, row in df.iterrows():
            # 合并所有特征
            feat = {}
            feat.update(self.length_features(row))
            feat.update(self.relevance_features(row))
            feat.update(self.language_quality_features(row))
            feat.update(self.logic_features(row))
            
            # 文章层面特征
            if is_train:
                feat.update(self.article_level_features(row, train_data=df))
            else:
                feat.update(self.article_level_features(row))
            
            features.append(feat)
        
        return pd.DataFrame(features)
    
    @staticmethod
    def calculate_perplexity(text, ngram_model):
        """计算n-gram模型的困惑度（简化版）"""
        try:
            return ngram_model.perplexity(word_tokenize(text.lower()))
        except:
            return float("inf")

# ----------------------
# 3. 执行特征工程
# ----------------------
if __name__ == "__main__":
    # 加载数据 
    train_df, test_df = load_data()
    
    # 初始化特征工程器
    fe = FeatureEngineer()
    
    # 拟合训练集
    fe.fit(train_df)
    
    # 生成训练集特征
    train_features = fe.transform(train_df, is_train=True)
    train_features = pd.concat([train_df[["student_id", "article_id", "score"]], train_features], axis=1)
    
    # 生成测试集特征
    test_features = fe.transform(test_df, is_train=False)
    test_features = pd.concat([test_df[["student_id", "article_id"]], test_features], axis=1)
    
    # 保存特征
    train_features.to_csv("train_features.csv", index=False)
    test_features.to_csv("test_features.csv", index=False)
    print("特征工程完成，已保存至train_features.csv和test_features.csv")