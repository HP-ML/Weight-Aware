import torch

def sort_matrix_rows_optimized(matrix):
    """
    将矩阵按值从小到大排序，并填充到矩阵行中。

    参数:
        matrix (torch.Tensor): 输入矩阵
    
    返回:
        transformed_matrix (torch.Tensor): 按值排序后的矩阵
        original_indices (torch.Tensor): 原始值的索引，用于恢复
    """
    # 展平矩阵并排序
    flat_matrix = matrix.flatten()  # 等价于 view(-1)
    sorted_values, original_indices = torch.sort(flat_matrix)  # 从小到大排序

    # 将排序后的值逐行填充到矩阵
    transformed_matrix = sorted_values.view(matrix.shape)
    print("************")
    return transformed_matrix, original_indices

def restore_matrix_optimized(transformed_matrix, original_indices):
    """
    将排序后的矩阵恢复为原始矩阵。

    参数:
        transformed_matrix (torch.Tensor): 排序后的矩阵
        original_indices (torch.Tensor): 原始值的索引
    
    返回:
        restored_matrix (torch.Tensor): 恢复的原始矩阵
    """
    # 展平排序矩阵
    flat_transformed = transformed_matrix.flatten()

    # 创建恢复矩阵并填充原始位置
    restored_flat = torch.zeros_like(flat_transformed)
    restored_flat[original_indices] = flat_transformed
    print("_____________")

    # 恢复形状
    return restored_flat.view(transformed_matrix.shape)

# 示例
matrix = torch.tensor(
    [[1.2, -2.3,  0.5, -0.7],
     [-0.8,  1.0,  2.0, -1.1],
     [0.6, -1.2, -1.0,  0.4],
     [0.9, -0.3,  0.7, -0.6]],
    dtype=torch.float32
)

# # 转换操作
# transformed_matrix, original_indices = sort_matrix_rows_optimized(matrix)
# print("Original Matrix:\n", matrix)
# print("\nTransformed Matrix:\n", transformed_matrix)
#
# # 恢复操作
# restored_matrix = restore_matrix_optimized(transformed_matrix, original_indices)
# print("\nRestored Matrix:\n", restored_matrix)
