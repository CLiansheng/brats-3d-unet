
import numpy as np
import nibabel as nib
import pyvista as pv
from pathlib import Path


# ==========================
# 1. 配置文件路径
# ==========================

ROOT = Path(__file__).resolve().parent.parent
patient = "BraTS-GLI-00046-000"

data_dir = ROOT / "data" / "raw" / patient

mri_path = data_dir / f"{patient}-t2f.nii.gz"
seg_path = data_dir / f"{patient}-seg.nii.gz"


# ==========================
# 2. 读取 MRI 和 Seg
# ==========================

# MRI：保存每个体素的信号强度
mri_file = nib.load(str(mri_path))
mri = mri_file.get_fdata(dtype=np.float32)

# Seg：保存每个体素的肿瘤类别
seg_file = nib.load(str(seg_path))
seg = np.asanyarray(seg_file.dataobj)

print("MRI shape:", mri.shape)
print("Seg shape:", seg.shape)
print("标签类别:", np.unique(seg))

# 确保 MRI 和 Seg 的形状及空间坐标一致
assert mri.shape == seg.shape
assert np.allclose(mri_file.affine, seg_file.affine)


# ==========================
# 3. 创建三维网格与渲染窗口
# ==========================

# 三维网格，每个数组元素对应一个网格点
grid = pv.ImageData(dimensions=seg.shape)

# 创建 PyVista 渲染器
plotter = pv.Plotter()

# 设置背景颜色
plotter.set_background("white")


# ==========================
# 4. 重建整个大脑
# ==========================

# 非零 MRI 区域近似作为脑部区域
brain_mask = (mri > 0).astype(np.uint8)

# 将三维数组展平成一维，绑定到网格点
grid.point_data["brain_mask"] = brain_mask.ravel(order="F")

# 在 0 和 1 的交界处提取三维表面
brain_surface = grid.contour(
    [0.5],
    scalars="brain_mask"
)

# 将数组坐标转换为真实空间坐标
brain_surface = brain_surface.transform(
    mri_file.affine,
    inplace=False
)

# 添加半透明的大脑
plotter.add_mesh(
    brain_surface,
    color="lightgray",
    opacity=0.10,
    smooth_shading=True,
    label="Brain"
)


# ==========================
# 5. 重建肿瘤的三个区域
# ==========================

# BraTS 2023 的三种肿瘤标签
colors = {
    1: "red",
    2: "green",
    3: "yellow"
}

for label, color in colors.items():

    # 当前类别的位置为 1，其余为 0
    mask = (seg == label).astype(np.uint8)

    # 如果不存在该类别，跳过
    if not np.any(mask):
        continue

    # 将三维 Mask 绑定到网格
    grid.point_data["mask"] = mask.ravel(order="F")

    # 提取当前肿瘤区域的三维表面
    surface = grid.contour(
        [0.5],
        scalars="mask"
    )

    # 转换到真实空间坐标
    surface = surface.transform(
        seg_file.affine,
        inplace=False
    )

    # 添加到三维场景
    plotter.add_mesh(
        surface,
        color=color,
        opacity=0.9,
        smooth_shading=True,
        label=f"Label {label}"
    )


# ==========================
# 6. 显示最终三维模型
# ==========================

# 添加图例
plotter.add_legend(
    size=(0.18, 0.20),
    bcolor="white"
)

# 显示坐标轴
plotter.show_axes()

# 自动调整相机
plotter.reset_camera()

# 启动三维交互窗口
plotter.show()
