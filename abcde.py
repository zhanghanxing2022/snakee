import pandas as pd
import re

def parse_test_results(row):
    public = str(row['public'])
    private = str(row['private'])
    total_tests = 36
    # 情况1：都为空，通过0个测试
    if pd.isna(row['public']) and pd.isna(row['private']):
        return 0, total_tests

    # 情况2：包含数字（负数表示失败数）
    if public.replace('-', '').replace('.0', '').isdigit() and private.replace('-', '').replace('.0', '').isdigit():
        failed = abs(int(float(public))) + abs(int(float(private)))
        passed = total_tests - failed
        return passed, total_tests

    # 情况3：包含a+b格式
    if '+' in public:
        passed = sum(int(x) for x in re.findall(r'\d+', public))
        return passed, total_tests# 默认情况
    return 0, total_tests

# 读取Excel文件
file_path = '/Users/zhanghanxing/Downloads/副本2024-12-19T1605_评分-COMP130014.01_编译_ck(1).xlsx'
df = pd.read_excel(file_path)

# 创建新的DataFrame
new_data = []
for _, row in df.iterrows():
    student_id = str(row['SIS Login ID'])  # 修改这里，使用SIS Login ID
    name = row['Student']
    passed_tests, total_tests = parse_test_results(row)
    failed_tests = total_tests - passed_tests

    new_data.append({
        'student_id': student_id,
        'name': name,
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests
    })

# 创建新的DataFrame并保存为CSV
new_df = pd.DataFrame(new_data)
new_df.to_csv('test_ref.csv', index=False)
print("文件已保存为 test_ref.csv")
print("\n预览前几行：")
print(new_df.head())