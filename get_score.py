# DentVLM evaluation.
#
# This script calculates the final accuracy, hit rate, and Intersection over
# Union (IoU) scores for the model's predictions. It is used to evaluate
# the performance of the trained model on test datasets.


import os
import re
import json
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score

languages = ['zh', 'en']
malocclusion = ['错合类型', '面型', '安氏分类', '骨性分类', '拥挤', '牙列间隙', '深覆合', '深覆盖', '开合', '牙前突', '上颌前突', '上颌发育不足', '下颌前突', '下颌后缩', '前牙反合', '后牙锁合', '后牙反合', '中线', '矢状关系']
disease = ['龋坏', '牙周病', '楔状缺损', '脱钙', '软垢', '牙磨损', '阻生牙', '修复冠', '根管填充', '填充物', '连接桥', '根尖周炎', '残根', '种植体', '残冠', '乳牙萌出空间不足', '牙结石']
dental_term_dict = {
    "错合类型": "Types of Malocclusion",
    "面型": "Facial Profile",
    "安氏分类": "Angle's Classification",
    "骨性分类": "Skeletal Classification",
    "拥挤": "Dental Crowding",
    "牙列间隙": "Dental Spacing",
    "深覆合": "Deep Overbite",
    "深覆盖": "Deep Overjet",
    "开合": "Open Bite",
    "牙前突": "Dental Protrusion",
    "上颌前突": "Maxillary Protrusion",
    "上颌发育不足": "Maxillary Hypoplasia",
    "下颌前突": "Mandibular Protrusion",
    "下颌后缩": "Mandibular Retrognathism",
    "前牙反合": "Anterior Crossbite",
    "后牙锁合": "Posterior Scissor Bite",
    "后牙反合": "Posterior Crossbite",
    "中线": "Midline Deviation",
    "矢状关系": "Sagittal Relationship",
    "龋坏": "Caries",
    "牙周病": "Periodontal Disease",
    "楔状缺损": "Wedge-shaped Defect",
    "脱钙": "Demineralization",
    "软垢": "Soft Deposit",
    "牙磨损": "Tooth Wear",
    "阻生牙": "Impacted Tooth",
    "修复冠": "Prosthetic Crown",
    "根管填充": "Root Canal Therapy",
    "填充物": "Fillings",
    "连接桥": "Prosthetic Bridge",
    "根尖周炎": "Apical Periodontitis",
    "残根": "Residual Root",
    "种植体": "Implant",
    "残冠": "Residual Crown",
    "乳牙萌出空间不足": "Insufficient Space for Primary Tooth Eruption",
    "牙结石": "Calculus",
}

area_term_dict = {
    "上、下牙列的右侧后牙区": "the right posterior region of both the upper and lower dentition",
    "上、下牙列的前牙区": "the anterior region of both the upper and lower dentition",
    "上、下牙列的左侧后牙区": "the left posterior region of both the upper and lower dentition",
    "上牙列的右侧后牙区": "the right posterior region of the upper dentition",
    "上牙列的前牙区": "the anterior region of the upper dentition",
    "上牙列的左侧后牙区": "the left posterior region of the upper dentition",
    "下牙列的右侧后牙区": "the right posterior region of the lower dentition",
    "下牙列的前牙区": "the anterior region of the lower dentition",
    "下牙列的左侧后牙区": "the left posterior region of the lower dentition",
}


def get_args():
    parser = argparse.ArgumentParser(description="Final Evaluation.")
    parser.add_argument('--input_path', type=str, default="answer/answer.json", help="Path of the answer file")
    parser.add_argument('--output_path', type=str, default="answer/answer_score.json", help="Path of the output file")
    parser.add_argument('--proc_total', type=int, default=6, help="Total number of parallel processes.")
    return parser.parse_args()


def get_hit_rate(pred, gt, lan_cls, data_type):
    def ex_keys(text, refer_list):
        cur_keys = []
        for item in refer_list:
            if item in text:
                cur_keys.append(item)
        return sorted(cur_keys)
    
    if data_type == 'Facial Profile':
        if lan_cls == "zh":
            cur_refer_list = malocclusion[10:-5]+["正常"]
        elif lan_cls == "en":
            cur_refer_list = [dental_term_dict[mal_item].lower() for mal_item in malocclusion[10:-5]]+["normal"]
    elif data_type == 'Types of Malocclusion':
        if lan_cls == "zh":
            cur_refer_list = malocclusion[4:-2]
        elif lan_cls == "en":
            cur_refer_list = [dental_term_dict[mal_item].lower() for mal_item in malocclusion[4:-2]]
    else:
        raise ValueError(f"{data_type} is error!")

    pred_keys, gt_keys = ex_keys(pred.lower(), cur_refer_list), ex_keys(gt.lower(), cur_refer_list)
    hitted_cnt = 0
    for pred_key in pred_keys:
        if pred_key not in gt_keys:
            return 0
        hitted_cnt += 1
    return hitted_cnt/len(gt_keys)


def calc_iou(pred, gt, lan_cls):
    def trans_text_to_areaList(text, refer_list):
        areaList = [0, 0, 0, 0, 0, 0]
        for item_id, item in enumerate(refer_list):
            if item in text:
                if item_id < 3:
                    areaList[item_id%3] = 1
                    areaList[item_id%3+3] = 1
                elif item_id < 6:
                    areaList[item_id%3] = 1
                else:
                    areaList[item_id%3+3] = 1
        return areaList
    
    if lan_cls == "zh":
        cur_refer_list = list(area_term_dict.keys())
    elif lan_cls == "en":
        cur_refer_list = list(area_term_dict.values())

    # print("pred:", pred.lower())
    # print("gt:", gt.lower())
    # print("cur_refer_list:", cur_refer_list)
    pred_area_list, gt_area_list = trans_text_to_areaList(pred.lower(), cur_refer_list), trans_text_to_areaList(gt.lower(), cur_refer_list)
    iarea, uarea = 0, 0
    for pred_item, gt_item in zip(pred_area_list, gt_area_list):
        if pred_item == 1 and gt_item == 1:
            iarea += 1
        if pred_item == 1 or gt_item == 1:
            uarea += 1
    return iarea / uarea


def get_ans_value(text, lan_cls, data_type):
    if data_type in ['Angle\'s Classification', 'Skeletal Classification']:
        roman_to_number = {"I": 1, "II": 2, "III": 3}
        text = text.replace("Ⅲ", "III").replace("Ⅱ", "II").replace("Ⅰ", "I")
        pattern = re.compile(r'(?:I{1,3})')
        text_match = pattern.search(text)
        if text_match is None:
            return 0
        return roman_to_number[text_match.group()]
    else:
        text = text.lower()
        if lan_cls == "zh":
            lbl1, lbl2 = "是", "否"
            if data_type in ['Midline Deviation', 'Sagittal Relationship']:
                lbl1, lbl2 = "正常", "异常"
        elif lan_cls == "en":
            lbl1, lbl2 = "yes", 'no'
            if data_type in ['Midline Deviation', 'Sagittal Relationship']:
                lbl1, lbl2 = "normal", "abnormal"
        if lbl2 in text and lbl1 in text.replace(lbl2, ""):
            return 0
        if lbl2 in text:
            return 2
        if lbl1 in text:
            return 1
        return 0


def draw_excel(score_data, out_file, out_sheet):
    df = pd.DataFrame(columns=["Category", "Type", "Disease", "Value"])
    rows = []
    for category, value in score_data["hit_rate"].items():
        rows.append({"Category": "hit_rate", "Type": category, "Disease": "", "Value": value})
    for category, value in score_data["area_iou"].items():
        rows.append({"Category": "area_iou", "Type": category, "Disease": "", "Value": value})
    for sub_category in ["mal", "dis"]:
        for disease, value in score_data["acc"][sub_category].items():
            rows.append({"Category": "acc", "Type": sub_category, "Disease": disease, "Value": value})

    df = pd.DataFrame(rows)

    if os.path.isfile(out_file):
        with pd.ExcelWriter(out_file, engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, sheet_name=out_sheet, index=False)
    else:
        with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=out_sheet, index=False)


if __name__ == "__main__":
    args = get_args()
    test_data = []
    if args.proc_total == 0:
        with open(args.input_path, "r", encoding='utf-8') as F:
            test_data += json.load(F)
    else:
        for proc_id in range(args.proc_total):
            with open(f"{args.input_path[:-5]}{proc_id}.json", "r", encoding='utf-8') as F:
                for line in F:
                    test_data.append(json.loads(line))

    hit_rate_dict = {lan: {item: 0 for item in ['Types of Malocclusion', 'Facial Profile']} for lan in languages}
    hit_rate_cnt_dict = {lan: {item: 0 for item in ['Types of Malocclusion', 'Facial Profile']} for lan in languages}
    mal_pred = {lan: {dental_term_dict[item]: [] for item in malocclusion} for lan in languages}
    mal_gt = {lan: {dental_term_dict[item]: [] for item in malocclusion} for lan in languages}
    dis_pred = {lan: {dental_term_dict[item]: [] for item in disease} for lan in languages}
    dis_gt = {lan: {dental_term_dict[item]: [] for item in disease} for lan in languages}
    area_iou_dict = {lan: {dental_term_dict[item]: 0 for item in disease} for lan in languages}
    area_iou_cnt_dict = {lan: {dental_term_dict[item]: 0 for item in disease} for lan in languages}

    ans_data = []
    for item in tqdm(test_data, total=len(test_data)):
        lan_cls, data_cls, data_type = item['id'].split('_')[0], item['id'].split('_')[1], item['id'].split('_')[-1]
        pred_response = item.get('pred_response', item.get('pred_rat', ''))
        if data_type in ['Types of Malocclusion', 'Facial Profile']:
            score = get_hit_rate(pred_response.split("\n")[0], item['gt_ans'], lan_cls, data_type)
            hit_rate_dict[lan_cls][data_type] += score
            hit_rate_cnt_dict[lan_cls][data_type] += 1
        else:
            pred_value = get_ans_value(pred_response.split("\n")[0], lan_cls, data_type)
            gt_value = get_ans_value(item['gt_ans'], lan_cls, data_type)
            if pred_value == 0:
                print(item['id'], "pred_response: ", pred_response)
            if gt_value == 0:
                print(item['id'], "gt_ans: ", item['gt_ans'])
            score = 1 if pred_value == gt_value else 0
            if data_cls == "mal":
                mal_pred[lan_cls][data_type].append(pred_value)
                mal_gt[lan_cls][data_type].append(gt_value)
            elif data_cls == "dis":
                dis_pred[lan_cls][data_type].append(pred_value)
                dis_gt[lan_cls][data_type].append(gt_value)
        cur_ans_item = {
            "id": item['id'],
            "question": item['question'],
            "pred_response": pred_response,
            "pred_ans": pred_response.split("\n")[0],
            "gt_ans": item['gt_ans'],
            "ans_score": score,
            "images": item['images']
        }
        if "area" in item.keys() and cur_ans_item['ans_score'] == 1:
            score = calc_iou(pred_response, item['area'], lan_cls)
            area_iou_dict[lan_cls][data_type] += score
            area_iou_cnt_dict[lan_cls][data_type] += 1
            cur_ans_item['area'] = item['area']
            cur_ans_item['area_score'] = score
        ans_data.append(cur_ans_item)

    mal_acc = {lan: {dental_term_dict[item]: 0 for item in malocclusion} for lan in languages}
    dis_acc = {lan: {dental_term_dict[item]: 0 for item in disease} for lan in languages}
    for lan_cls in languages:
        for mal in malocclusion:
            mal = dental_term_dict[mal]
            if mal in ['Types of Malocclusion', 'Facial Profile']:
                mal_acc[lan_cls].pop(mal)
            else:
                mal_acc[lan_cls][mal] = accuracy_score(mal_gt[lan_cls][mal], mal_pred[lan_cls][mal])
        mal_acc[lan_cls]["average"] = sum(list(mal_acc[lan_cls].values())) / len(mal_acc[lan_cls].values())
        for dis in disease:
            dis = dental_term_dict[dis]
            dis_acc[lan_cls][dis] = accuracy_score(dis_gt[lan_cls][dis], dis_pred[lan_cls][dis])
            area_iou_dict[lan_cls][dis] = area_iou_dict[lan_cls][dis] / area_iou_cnt_dict[lan_cls][dis] if area_iou_cnt_dict[lan_cls][dis]!=0 else 0
        dis_acc[lan_cls]["average"] = sum(list(dis_acc[lan_cls].values())) / len(dis_acc[lan_cls].values())
        area_iou_dict[lan_cls]["average"] = sum(list(area_iou_dict[lan_cls].values())) / len(area_iou_dict[lan_cls].values())

    score_data = {
        lan_cls: {
            "hit_rate": {
                'Types of Malocclusion': hit_rate_dict[lan_cls]['Types of Malocclusion']/max(hit_rate_cnt_dict[lan_cls]['Types of Malocclusion'], 1),
                'Facial Profile': hit_rate_dict[lan_cls]['Facial Profile']/max(hit_rate_cnt_dict[lan_cls]['Facial Profile'], 1)
            },
            "area_iou": area_iou_dict[lan_cls],
            "acc": {"mal": mal_acc[lan_cls], "dis": dis_acc[lan_cls]},
        } for lan_cls in languages
    }
    ans_data = [score_data] + ans_data
    
    for lan_cls in languages:
        draw_excel(score_data[lan_cls], f"answer/score_output_{lan_cls}.xlsx", '-'.join(args.input_path.split("/")[1:3]))
    with open(args.output_path, "w", encoding='utf-8') as F:
        json.dump(ans_data, F, indent=2, ensure_ascii=False)
