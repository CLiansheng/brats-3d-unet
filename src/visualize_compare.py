import numpy as np
import nibabel as nib
import pyvista as pv
from pathlib import Path


# ==========================
# 1. 配置路径
# ==========================

ROOT = Path(__file__).resolve().parent.parent

# 改成你要看的患者
patient_name = "BRATS_446.nii.gz"

# MSD 数据集路径
data_root = ROOT / "data" / "Task01_BrainTumour"
image_path = data_root / "imagesTr" / patient_name
label_path = data_root / "labelsTr" / patient_name

# 预测结果路径
pred_path = ROOT / "outputs" / "evaluation" / patient_name.replace(".nii.gz", "_pred.npy")


# ==========================
# 2. 读取 MRI、真实标签、预测标签
# ==========================

image_file = nib.load(str(image_path))
label_file = nib.load(str(label_path))

image = image_file.get_fdata(dtype=np.float32)   # shape: [H, W, D, 4]
label = label_file.get_fdata().astype(np.int64)  # shape: [H, W, D]
pred = np.load(pred_path).astype(np.int64)       # shape: [H, W, D]

print("MRI shape:", image.shape)
print("GT shape:", label.shape)
print("Pred shape:", pred.shape)
print("GT labels:", np.unique(label))
print("Pred labels:", np.unique(pred))

assert label.shape == pred.shape


# ==========================
# 3. 构建脑部轮廓
# ==========================

# 4个模态只要有一个非0，就认为属于脑区
brain_mask = np.any(image != 0, axis=-1).astype(np.uint8)

grid = pv.ImageData(dimensions=label.shape)


def extract_surface(mask_3d, affine):
    """
    输入:
        mask_3d: 三维二值数组
        affine: nibabel 的仿射矩阵
    输出:
        surface: 提取出的三维表面
    """
    grid.point_data["mask"] = mask_3d.ravel(order="F")
    surface = grid.contour([0.5], scalars="mask")
    surface = surface.transform(affine, inplace=False)
    return surface


brain_surface = extract_surface(brain_mask, image_file.affine)


# ==========================
# 4. 定义绘制函数
# ==========================

colors = {
    1: "red",
    2: "green",
    3: "yellow"
}


def add_tumor_surfaces(plotter, seg, affine, title):
    # 先加脑部轮廓
    plotter.add_mesh(
        brain_surface,
        color="lightgray",
        opacity=0.08,
        smooth_shading=True
    )

    # 再加三类肿瘤
    for tumor_label, color in colors.items():
        mask = (seg == tumor_label).astype(np.uint8)

        if not np.any(mask):
            continue

        surface = extract_surface(mask, affine)

        plotter.add_mesh(
            surface,
            color=color,
            opacity=0.85,
            smooth_shading=True,
            label=f"Label {tumor_label}"
        )

    plotter.add_text(title, font_size=12)
    plotter.show_axes()
    plotter.reset_camera()


# ==========================
# 5. 左右对比显示
# ==========================

plotter = pv.Plotter(shape=(1, 2), window_size=(1400, 700))
plotter.set_background("white")

# 左边：真实标签
plotter.subplot(0, 0)
add_tumor_surfaces(plotter, label, label_file.affine, "Ground Truth")

# 右边：预测标签
plotter.subplot(0, 1)
add_tumor_surfaces(plotter, pred, label_file.affine, "Prediction")

plotter.link_views()
# 保存真实标签与预测结果的对比图片
save_path = (
    ROOT / "outputs" / "evaluation"
    / patient_name.replace(".nii.gz", "_compare.png")
)

# 先初始化并渲染窗口，不立即关闭
plotter.show(auto_close=False, interactive=False)

# 保存截图
plotter.screenshot(str(save_path))

# 截图完成后关闭窗口
plotter.close()

print("图片是否存在:", save_path.exists())
print("对比图片路径:", save_path)