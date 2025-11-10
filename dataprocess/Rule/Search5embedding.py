import openai
import numpy as np
from scipy.spatial.distance import cdist
import json
from tqdm import tqdm
from typing import List, Dict, Any
import time

API_KEY = "sk-X" 
BASE_URL = ""

EMBEDDING_MODEL = "text-embedding-3-small"

QUESTIONS_FILE_PATH = "dataprocess/Rule/train_with_search5.json"
RULES_FILE_PATH = "dataprocess/Rule/TUR.json"
OUTPUT_FILE_PATH = "dataprocess/Rule/train_with_search5.json"

def get_openai_embeddings(texts: List[str], client: openai.OpenAI, model: str) -> np.ndarray:
    try:
        response = client.embeddings.create(input=texts, model=model)
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
    except Exception as e:
        print(f"调用 OpenAI API 时出错: {e}")
        raise

class OpenAIRetriever:
    def __init__(self, api_key: str, base_url: str, model: str = EMBEDDING_MODEL):
        self.model = model
        try:
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        except Exception as e:
            print(f"OpenAI 客户端初始化失败: {e}")
            raise
        
        self.rule_embeddings = None
        self.original_rules = None

    def build_rule_index(self, rules: List[str]):
        print(f"正在为 {len(rules)} 条规则创建向量索引...")
        self.original_rules = rules
        self.rule_embeddings = get_openai_embeddings(rules, self.client, self.model)

    def search(self, questions: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if self.rule_embeddings is None:
            raise RuntimeError("规则索引尚未建立。请在搜索前调用 .build_rule_index(rules) 方法。")

        question_texts = [q['question'] for q in questions]
        question_embeddings = get_openai_embeddings(question_texts, self.client, self.model)

        cosine_distances = cdist(question_embeddings, self.rule_embeddings, 'cosine')
        cosine_similarities = 1 - cosine_distances

        for i in tqdm(range(len(questions))):
            scores_for_this_question = cosine_similarities[i]
            k_for_search = min(top_k, len(self.original_rules))
            top_rule_indices = np.argsort(scores_for_this_question)[::-1][:k_for_search]
            
            top_5_rules = [self.original_rules[idx] for idx in top_rule_indices]
            questions[i]['Search5TUR'] = top_5_rules

        return questions

def main():
    try:
        with open(QUESTIONS_FILE_PATH, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        with open(RULES_FILE_PATH, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
    except FileNotFoundError as e:
        print(f"错误: 找不到输入文件 {e.filename}。")
        return

    retriever = OpenAIRetriever(api_key=API_KEY, base_url=BASE_URL)

    retriever.build_rule_index(rules_data)

    updated_questions = retriever.search(questions_data, top_k=5)

    try:
        print(f"将结果保存到 '{OUTPUT_FILE_PATH}'...")
        with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(updated_questions, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"保存文件时出错: {e}")


if __name__ == '__main__':
    main()