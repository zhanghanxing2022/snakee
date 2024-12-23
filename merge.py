import pandas as pd

# 读取两个CSV文件
results_df = pd.read_csv('/Users/zhanghanxing/同步空间/test_results.csv')
ref_df = pd.read_csv('/Users/zhanghanxing/Desktop/work/abv/Snakeee/test_ref.csv')

# 合并数据，使用student_id作为键
merged_df = pd.merge(
    ref_df[['student_id', 'name', 'passed_tests']], 
    results_df[['student_id', 'passed_tests']], 
    on='student_id',
    suffixes=('_ref', '_results')
)

# 添加是否需要注意的标记
merged_df['needs_attention'] = merged_df['passed_tests_ref'] != merged_df['passed_tests_results']
merged_df['total_tests'] =36
merged_df['failed_tests'] = 36 - merged_df['passed_tests_ref']

# 保存结果
merged_df.to_csv('result.csv', index=False)

# 打印需要注意的记录
attention_needed = merged_df[merged_df['needs_attention']]
if not attention_needed.empty:
    print("\n需要注意的记录：")
    print(attention_needed)
else:
    print("\n所有记录都匹配")