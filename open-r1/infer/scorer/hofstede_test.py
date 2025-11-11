from config import vsm13_data
import fire, time
import re, math
import codecs, csv
from config import model_dict
import json

def computeMetrics(ans_list):
    pdi = 35 * (ans_list[7-1] - ans_list[2-1]) + 25 * (ans_list[20-1] - ans_list[23-1])
    idv = 35 * (ans_list[4-1] - ans_list[1-1]) + 35 * (ans_list[9-1] - ans_list[6-1])
    mas = 35 * (ans_list[5-1] - ans_list[3-1]) + 25 * (ans_list[8-1] - ans_list[10-1])
    uai = 40 * (ans_list[18-1] - ans_list[15-1]) + 25 * (ans_list[21-1] - ans_list[24-1])
    lto = 40 * (ans_list[13-1] - ans_list[14-1]) + 25 * (ans_list[19-1] - ans_list[22-1])
    ivr = 35 * (ans_list[12-1] - ans_list[11-1]) + 40 * (ans_list[17-1] - ans_list[16-1])

    return pdi+50, idv+50, mas+50, uai+50, lto+50, ivr+50

def write_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def run(culture, file_path, engine=None):
    country_dict = {'Arabic': 'Iraq', 'Bengali': 'Bangladesh', 'Chinese': 'China', 'German': 'Germany', 'Korean': 'Korea South', 'Portuguese': 'Brazil', 'Spanish': 'Argentina', 'Turkish': 'Turkey', 'USA': 'U.S.A.'}

    ans_dict = dict()
    with codecs.open(f'CulturePark/data/6-dimensions-for-website-2015-08-16.csv', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f, skipinitialspace=True, delimiter=';'):
            country = row['country']
            pdi = row['pdi']
            idv = row['idv']
            mas = row['mas']
            uai = row['uai']
            lto = row['ltowvs']
            ivr = row['ivr']
            ans_dict[country] = {'pdi': pdi, 'idv': idv, 'mas': mas, 'uai': uai, 'lto': lto, 'ivr': ivr}
    
    cur_country = country_dict[culture]
    human_ans = ans_dict[cur_country]
    print('Human', human_ans)
    with open(file_path, 'r') as f:
        res = json.load(f)
    # model_dict = getModel(culture)
    # test_model = model_dict[culture]

    ans_list = []
    for response in res:
        try:
            ans_list.append(int(re.findall(r'\d', response["generated_text"])[0]))
        except:
            ans_list.append(10)
            print(response)
    pdi, idv, mas, uai, lto, ivr = computeMetrics(ans_list)
    cur_ans = {'pdi': pdi, 'idv': idv, 'mas': mas, 'uai': uai, 'lto': lto, 'ivr': ivr}
    print('Cur', pdi, idv, mas, uai, lto, ivr)

    missed_key = []
    human_point = []
    cur_point = []
    for key in human_ans.keys():
        v = human_ans[key]
        if '#' in v:
            missed_key.append(key)
        else:
            human_point.append(int(v))
    for key in cur_ans.keys():
        v = cur_ans[key]
        if key not in missed_key:
            cur_point.append(v)
    
    distance = math.sqrt(sum([(x - y) ** 2 for x, y in zip(human_point, cur_point)]))

    print('Dis: ', distance)

    OUTPUT_DIR = "open-r1/infer/"
    path_parts = file_path.strip("/").split('/')[: -1]
    num = -1
    while num >= -len(path_parts):
        part = path_parts[num]
        if "train_" in part or "Qwen2.5-" in part or "PT_" in part or "gpt-" in part: break
        num -= 1
    else: num = -1
    
    import os
    OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, path_parts[num])
    
    if "Qwen2.5-3B-Instruct" in file_path:
        OUTPUT_DIR = os.path.join(OUTPUT_DIR, culture.lower())
    
    if path_parts[-1].startswith("checkpoint"):
        OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, path_parts[-1])
    
    output_path = os.path.join(OUTPUT_FOLDER, "all_results.json")
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            results_list = json.load(f)
    else:
        results_list = {}

    if "vsm13" not in results_list.keys():
        results_list["vsm13"] = distance
        write_json(output_path, results_list)

    # with jsonlines.open(f'results/hofstede.jsonl',mode='a') as writer:
    #     cur_dict = {'culture': culture, 'engine': engine, 'distance': distance}
    #     writer.write(cur_dict)


if __name__ == '__main__':
    fire.Fire(run)  