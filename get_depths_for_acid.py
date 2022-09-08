from PIL import Image as PILImage
import torch
from geofree.modules.warp.midas import Midas
from geofree.data.read_write_model import *
from torchvision.utils import save_image
import glob
import cv2

def load_sparse_model_example(img_path, size):
    """
    Parameters
        root        folder containing directory sparse with points3D.bin,
                    images.bin and cameras.bin, and directory images
        img_dst     filename of image in images to be used as destination
        img_src     filename of image in images to be used as source
        size        size to resize image and parameters to. If None nothing is
                    done, otherwise it should be in (h,w) format
    Returns
        example     dictionary containing
            dst_img     destination image as (h,w,c) array in range (-1,1)
            src_img     source image as (h,w,c) array in range (-1,1)
            src_points  sparse set of 3d points for source image as (N,3) array
                        with (:,:2) being pixel coordinates and (:,2) depth values
            K           3x3 camera intrinsics
            K_inv       inverse of camera intrinsics
            R_rel       relative rotation mapping from source to destination
                        coordinate system
            t_rel       relative translation mapping from source to destination
                        coordinate system
    """
    # load images
    img = np.array(PILImage.open(img_path))
    print(img.max())
    print(img.min())
    H, W = img.shape[:2]
    center_crop_size = min(H, W)
    center = [H/2, W/2]
    x = center[1] - center_crop_size/2
    y = center[0] - center_crop_size/2
    img = img[int(y):int(y+center_crop_size), int(x):int(x+center_crop_size)]

    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    # img = img.resize((size[1],size[0]),
    #                         resample=PILImage.LANCZOS)
    img = img/127.5-1.0
    print(img.max())
    print(img.min())

    return img


root = "/media/data6/shengqu/datasets/acid/train/frames/*"
img_list = glob.glob(os.path.join(root, "*"))
print(len(img_list))

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
midas = Midas().to(device)
midas.eval()
midas.train = disabled_train

for img in img_list:
    rgb = load_sparse_model_example(img, size=None)
    rgb = rgb.astype(np.float32)

    with torch.no_grad():
        depth = midas(torch.Tensor(rgb).to(device).unsqueeze(0).permute(0, 3, 1, 2))
        # c, _ = midas.warp(x=torch.Tensor(example["src_img"]).to(device).unsqueeze(0).permute(0, 3, 1, 2), points=torch.Tensor(example["src_points"]).to(device).unsqueeze(0),
        #                 R=torch.Tensor(example["R_rel"]).to(device).unsqueeze(0), t=torch.Tensor(example["t_rel"]).to(device).unsqueeze(0),
        #                 K_src_inv=torch.Tensor(example["K_inv"]).to(device).unsqueeze(0), K_dst=torch.Tensor(example["K"]).to(device).unsqueeze(0))
    # print(c.shape)
    depth_min = depth.min()
    depth_max = depth.max()
    depth = (depth - depth_min)/(depth_max-depth_min)
    # save_image(c[0], "warped.png")
    # ori_img = torch.Tensor(example["src_img"]).to(device).unsqueeze(0).permute(0, 3, 1, 2)[0]
    # ori_img_min = ori_img.min()
    # ori_img_max = ori_img.max()
    # ori_img = (ori_img - ori_img_min)/(ori_img_max-ori_img_min)
    # save_image(ori_img, "original.png")
    # saving_root = os.path.dirname(img)
    img_saving_path = img.replace("frames", "128")
    saving_path = os.path.dirname(img_saving_path)
    print(saving_path)
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    save_image(depth[0], img_saving_path[:-4] + "_d.png")
    # save_image((rgb + 1.0) * 127.5, img_saving_path[:-4] + "_rgb.png")
    rgb = (rgb + 1.0) * 127.5
    # im = PILImage.fromarray(rgb)
    # im.save(img_saving_path[:-4] + "_rgb.png")
    cv2.imwrite(img_saving_path[:-4] + "_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# cv2.imwrite(os.path.join(testsavedir, f'{this_i_test:03d}' + "_rgb" + ".jpg"), cv2.cvtColor(utils.to8b(this_rgb), cv2.COLOR_RGB2BGR))
# write depth
# depth = depth[0]
# cv2.imwrite("depth.png", utils.to16b(np.nan_to_num(1.0 / np.clip(depth, 0.01, 100.0))))