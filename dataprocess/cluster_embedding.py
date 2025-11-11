import json
from openai import OpenAI
from sklearn.cluster import KMeans
import numpy as np

ANYWHERE_API_KEY = "sk-X"
ANYWHERE_BASE_URL = ""
MODEL_TO_USE_FOR_EXTRACTION = "gpt-4o"
MODEL_TO_USE_FOR_EMBEDDING = "text-embedding-3-small"

client = OpenAI(api_key=ANYWHERE_API_KEY, base_url=ANYWHERE_BASE_URL)

def get_chat_response(prompt, model_name):
    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a linguistic analyst who extracts core concepts from text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"  -> Chat API Error: {e}")
        return None

def get_openai_embeddings(texts_to_embed, model_name):
    try:
        response = client.embeddings.create(model=model_name, input=texts_to_embed)
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"  -> Embedding API Error: {e}")
        return None

extraction_prompt_template = """
# Role:
You are a linguistic analyst. Your task is to extract the single core subject or concept from a survey question, ignoring all framing, introductory phrases, and question marks.

# Instructions:
1. Read the user's [Survey Question].
2. Identify the central person, group, or idea being evaluated.
3. Return ONLY this core concept as a concise noun phrase. Do not add any explanation or other text.

# Examples:
- Input: "Would you not like to have drug addicts as neighbors?"
- Output: drug addicts
---
- Input: "How do you feel about having a strong leader who does not have to bother with parliament and elections as a way of governing this country?"
- Output: a strong leader who does not have to bother with parliament and elections
---
- Input: "Do you agree that when a mother works for pay, the children suffer?"
- Output: a working mother's effect on children

# Begin Extraction:

[Survey Question]: {survey_question}
"""

def write_json(file_path, s):
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(s, f, ensure_ascii=False, indent=4)

def main(input_file_path='dataprocess/concept_expand_rule.json', k=8):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        all_questions = json.load(f)
    
    extracted_concepts = []
    original_questions_map = {}

    print("--- 步骤 1: 提取核心概念 ---")
    for item in all_questions:
        q_id = item.get("q_id", "N/A")
        question_text = item.get("question", "")
        if not question_text: continue

        print(f"正在提取 Q_id: {q_id} 的概念...")
        prompt = extraction_prompt_template.format(survey_question=question_text)
        concept = get_chat_response(prompt, MODEL_TO_USE_FOR_EXTRACTION)
        
        if concept:
            extracted_concepts.append(concept)
            original_questions_map[concept] = question_text
            item["concept"] = concept
            print(f"  -> 提取到概念: {concept}")
    write_json("concept_expand_rule.json", all_questions)
    
    for q in all_questions:
        extracted_concepts.append(q.get("concept"))
        original_questions_map[q.get("concept")] = q.get("question")

    print("\n--- 步骤 2: 正在向量化所有核心概念... ---")
    embeddings = get_openai_embeddings(extracted_concepts, MODEL_TO_USE_FOR_EMBEDDING)

    if embeddings:
        print("\n--- 步骤 3: 正在对概念向量进行聚类... ---")
        embedding_array = np.array(embeddings)
        num_clusters = k
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        kmeans.fit(embedding_array)
        
        clusters = kmeans.labels_
        clustered_results = {i: [] for i in range(num_clusters)}
        for concept, cluster_id in zip(extracted_concepts, clusters):
            original_question = original_questions_map[concept]
            clustered_results[cluster_id].append(original_question)

        for i in range(num_clusters):
            print(f"\n--- 概念类别 {i} ---")
            for question in clustered_results[i]:
                print(f"- {question}")
        write_json("dataprocess/clustered_results.json", clustered_results)
    else:
        print("获取向量失败，聚类中止。")

from fire import Fire

if __name__ == "__main__":
    Fire(main)