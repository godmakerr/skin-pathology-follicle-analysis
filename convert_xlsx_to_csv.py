import pandas as pd
import os

# ================= 配置 =================
# 你标注好的 Excel 文件名
INPUT_XLSX = "scene_labels.xlsx" 

# 转换后输出的 CSV 文件名 (训练脚本要用这个)
OUTPUT_CSV = "scene_labels.csv"
# =======================================

def main():
    if not os.path.exists(INPUT_XLSX):
        print(f"[错误] 找不到文件: {INPUT_XLSX}")
        print("请确保你把标注好的 xlsx 文件放在当前目录下，并且名字改对了。")
        return

    print(f"正在读取 {INPUT_XLSX} ...")
    
    try:
        # 读取 Excel
        df = pd.read_excel(INPUT_XLSX)
        
        # 简单的数据检查
        print(f"读取成功，共 {len(df)} 行数据。")
        print("数据预览（前5行）：")
        print(df.head())
        
        # 检查必要的列名是否存在
        required_columns = ["filename", "slope_label", "layer_label"]
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            print("\n[警告] 你的 Excel 表头似乎不对！")
            print(f"缺少的列名: {missing}")
            print(f"当前的列名: {list(df.columns)}")
            print("请去 Excel 里把第一行表头改成: filename, slope_label, layer_label")
            # 这里不强制退出，万一你用了别的列名但知道自己在做什么
        
        # 保存为 CSV
        # index=False 表示不保存行号（0,1,2...）
        # encoding='utf-8' 确保中文或特殊字符不乱码
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        
        print(f"\n[成功] 已转换为: {OUTPUT_CSV}")
        print("现在你可以直接运行 python train_global_models.py 了。")

    except Exception as e:
        print(f"[失败] 转换过程中出错: {e}")

if __name__ == "__main__":
    main()