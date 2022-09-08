from PIL import Image as PILImage
import torch
from geofree.modules.warp.midas import Midas
from geofree.data.read_write_model import *
from torchvision.utils import save_image
import glob

def load_sparse_model_example(root, img_dst, img_src, size):
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
    # load sparse model
    model = os.path.join(root, "sparse")
    try:
        cameras, images, points3D = read_model(path=model, ext=".bin")
    except Exception as e:
        raise Exception(f"Failed to load sparse model {model}.") from e


    # load camera parameters and image size
    cam = cameras[1]
    h = cam.height
    w = cam.width
    params = cam.params
    K = np.array([[params[0], 0.0, params[2]],
                  [0.0, params[1], params[3]],
                  [0.0,       0.0,       1.0]])

    # find keys of desired dst and src images
    key_dst = [k for k in images.keys() if images[k].name==img_dst]
    assert len(key_dst)==1, (img_dst, key_dst)
    key_src = [k for k in images.keys() if images[k].name==img_src]
    assert len(key_src)==1, (img_src, key_src)
    keys = [key_dst[0], key_src[0]]

    # load extrinsics
    Rs = np.stack([images[k].qvec2rotmat() for k in keys])
    ts = np.stack([images[k].tvec for k in keys])

    # load sparse 3d points to be able to estimate scale
    sparse_points = [None, None]
    #for i in range(len(keys)):
    for i in [1]: # only need it for source
        key = keys[i]
        xys = images[key].xys
        p3D = images[key].point3D_ids
        pmask = p3D > 0
        # if verbose: print("Found {} 3d points in sparse model.".format(pmask.sum()))
        xys = xys[pmask]
        p3D = p3D[pmask]
        worlds = np.stack([points3D[id_].xyz for id_ in p3D]) # N, 3
        # project to current view
        worlds = worlds[..., None] # N,3,1
        pixels = K[None,...]@(Rs[i][None,...]@worlds+ts[i][None,...,None])
        pixels = pixels.squeeze(-1) # N,3

        # instead of using provided xys, one could also project pixels, ie
        # xys ~ pixels[:,:2]/pixels[:,[2]]
        points = np.concatenate([xys, pixels[:,[2]]], axis=1)
        sparse_points[i] = points

        # code to convert to sparse depth map
        # xys = points[:,:2]
        # xys = np.round(xys).astype(np.int)
        # xys[:,0] = xys[:,0].clip(min=0,max=w-1)
        # xys[:,1] = xys[:,1].clip(min=0,max=h-1)
        # indices = xys[:,1]*w+xys[:,0]
        # flatdm = np.zeros(h*w)
        # flatz = pixels[:,2]
        # np.put_along_axis(flatdm, indices, flatz, axis=0)
        # sparse_dm = flatdm.reshape(h,w)

    # load images
    im_root = os.path.join(root, "images")
    im_paths = [os.path.join(im_root, images[k].name) for k in keys]
    ims = list()
    for path in im_paths:
        im = PILImage.open(path)
        ims.append(im)

    if size is not None and (size[0] != h or size[1] != w):
        # resize
        ## K
        K[0,:] = K[0,:]*size[1]/w
        K[1,:] = K[1,:]*size[0]/h
        ## points
        points[:,0] = points[:,0]*size[1]/w
        points[:,1] = points[:,1]*size[0]/h
        ## img
        for i in range(len(ims)):
            ims[i] = ims[i].resize((size[1],size[0]),
                                   resample=PILImage.LANCZOS)


    for i in range(len(ims)):
        ims[i] = np.array(ims[i])/127.5-1.0


    # relative camera
    R_dst = Rs[0]
    t_dst = ts[0]
    R_src_inv = Rs[1].transpose(-1,-2)
    t_src = ts[1]
    R_rel = R_dst@R_src_inv
    t_rel = t_dst-R_rel@t_src
    K_inv = np.linalg.inv(K)

    # collect results
    example = {
        "dst_img": ims[0],
        "src_img": ims[1],
        "src_points": sparse_points[1],
        "K": K,
        "K_inv": K_inv,
        "R_rel": R_rel,
        "t_rel": t_rel,
    }

    return example

def pad_points(points, N):
    padded = -1*np.ones((N,3), dtype=points.dtype)
    padded[:points.shape[0],:] = points
    return padded

def read_model(path, ext=""):
    # try to detect the extension automatically
    if ext == "":
        if detect_model_format(path, ".bin"):
            ext = ".bin"
        elif detect_model_format(path, ".txt"):
            ext = ".txt"
        else:
            print("Provide model format: '.bin' or '.txt'")
            return

    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3d_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D


def detect_model_format(path, ext):
    if os.path.isfile(os.path.join(path, "cameras"  + ext)) and \
       os.path.isfile(os.path.join(path, "images"   + ext)) and \
       os.path.isfile(os.path.join(path, "points3D" + ext)):
        print("Detected model format: '" + ext + "'")
        return True

    return False

# root = "/scratch_net/biwidl212/shecai/datasets/acid_sparse/train/011aa706b20c2958"
root = "/media/data6/shengqu/datasets/acid_single_sparse/train/d7fca566a7941495"
# img_dir = os.path.join(root, "images")
img_dir = "/media/data6/shengqu/datasets/acid_single/train/frames/d7fca566a7941495"
print(img_dir)
img_list = [os.path.basename(x) for x in glob.glob(os.path.join(img_dir, "*"))]
print(img_list)

for img in img_list:
    # img_src = "4442604833.png"
    # img_dst = "4424653567.png"
    print(img)
    img_src = img
    img_dst = img
    example = load_sparse_model_example(
        root=root, img_dst=img_dst, img_src=img_src, size=None)

    for k in example:
        example[k] = example[k].astype(np.float32)

    example["src_points"] = pad_points(example["src_points"],
                                        16384)
    # example["seq"] = seq
    example["label"] = 0
    example["dst_fname"] = img_dst
    example["src_fname"] = img_src


    def disabled_train(self, mode=True):
        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        return self

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    midas = Midas().to(device)
    midas.eval()
    midas.train = disabled_train

    with torch.no_grad():
        depth = midas.scaled_depth(torch.Tensor(example["src_img"]).to(device).unsqueeze(0).permute(0, 3, 1, 2), torch.Tensor(example["src_points"]).to(device).unsqueeze(0), True)
        # c, _ = midas.warp(x=torch.Tensor(example["src_img"]).to(device).unsqueeze(0).permute(0, 3, 1, 2), points=torch.Tensor(example["src_points"]).to(device).unsqueeze(0),
        #                 R=torch.Tensor(example["R_rel"]).to(device).unsqueeze(0), t=torch.Tensor(example["t_rel"]).to(device).unsqueeze(0),
        #                 K_src_inv=torch.Tensor(example["K_inv"]).to(device).unsqueeze(0), K_dst=torch.Tensor(example["K"]).to(device).unsqueeze(0))
    # print(c.shape)
    # c_min = c.min()
    # c_max = c.max()
    # c = (c - c_min)/(c_max-c_min)
    # save_image(c[0], "warped.png")
    # ori_img = torch.Tensor(example["src_img"]).to(device).unsqueeze(0).permute(0, 3, 1, 2)[0]
    # ori_img_min = ori_img.min()
    # ori_img_max = ori_img.max()
    # ori_img = (ori_img - ori_img_min)/(ori_img_max-ori_img_min)
    # save_image(ori_img, "original.png")
    save_image(depth[0], os.path.join("/media/data6/shengqu/datasets/acid_single_seq/d", img))
    print(depth.min())
    print(depth.max())

# cv2.imwrite(os.path.join(testsavedir, f'{this_i_test:03d}' + "_rgb" + ".jpg"), cv2.cvtColor(utils.to8b(this_rgb), cv2.COLOR_RGB2BGR))
# write depth
# depth = depth[0]
# cv2.imwrite("depth.png", utils.to16b(np.nan_to_num(1.0 / np.clip(depth, 0.01, 100.0))))