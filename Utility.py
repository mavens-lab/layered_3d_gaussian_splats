import numpy as np
from plyfile import PlyData, PlyElement
import math
import os
import struct
import collections
from PIL import Image
from typing import NamedTuple

#### from Inria
def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5
def sigmoid(l):
    return np.array([1 / (1 + math.exp(-op)) for op in l])
def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose() # correct, R is numpy
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def transformPoint4x4(point3d, matrix4):
    x = point3d[0]
    y = point3d[1]
    z = point3d[2]
    transformed = [
        matrix4[0] * x + matrix4[4] * y + matrix4[8] * z + matrix4[12],
        matrix4[1] * x + matrix4[5] * y + matrix4[9] * z + matrix4[13],
        matrix4[2] * x + matrix4[6] * y + matrix4[10] * z + matrix4[14],
        matrix4[3] * x + matrix4[7] * y + matrix4[11] * z + matrix4[15]
    ]
    return transformed
def transformPoint4x3(point3d, matrix4):
    x = point3d[0]
    y = point3d[1]
    z = point3d[2]
    transformed = [
        matrix4[0] * x + matrix4[4] * y + matrix4[8] * z + matrix4[12],
        matrix4[1] * x + matrix4[5] * y + matrix4[9] * z + matrix4[13],
        matrix4[2] * x + matrix4[6] * y + matrix4[10] * z + matrix4[14],   
    ]
    return transformed

def ndc2Pix(v, S):
    return ((v+1.0) * S - 1.0) * 0.5

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def build_rotation(r):
    norm = np.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = np.zeros((q.shape[0], 3, 3))

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R
def build_scaling_rotation(s, r):
    L = np.zeros((s.shape[0], 3, 3))
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]
    L = R @ L
    return L
def strip_symmetric(sym):
    return strip_lowerdiag(sym)
def strip_lowerdiag(L):
    uncertainty = np.zeros((L.shape[0], 6))

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty
def build_covariance_from_scaling_rotation(scaling, rotation):
    scaling_modifier = 1
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = np.matmul(L, np.transpose(L, axes=(0, 2, 1)))
    symm = strip_symmetric(actual_covariance)
    return symm

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])
max_sh_degree = 3
class Images(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)
    
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
        
class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
        
class GaussianModel(NamedTuple):
    xyz : np.array
    features_dc : np.array
    features_rest : np.array
    scaling : np.array
    rotation : np.array
    opacity : np.array
    
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Images(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    print("Reading images from folder:", images_folder)
    for idx, key in enumerate(cam_extrinsics):
#         print('\r')
        # the exact output you're looking for:
#         print("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
#         sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec)) # correct, transpose on np
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
#     sys.stdout.write('\n')
    return cam_infos

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}
def sigmoid(z):
    return 1/(1 + np.exp(-z))
def load_ply(path):
#     plydata = PlyData.read(path)
#     vertices = plydata['vertex']
#     props = [p.name for p in vertices.properties]

#     positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
#     if 'red' in props:
#         colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
#     elif 'f_dc_0' in props:
#         c = np.vstack([vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2']]).T
#         colors = SH2RGB(c)
#     normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
#     return BasicPointCloud(points=positions, colors=colors, normals=normals)
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    opacities = sigmoid(opacities)

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    
    return GaussianModel(xyz=xyz, 
                         features_dc = features_dc,
                         features_rest = features_extra,
                         scaling = scales,
                         rotation = rots,
                         opacity = opacities)

def getRect(point_image, my_radius, grid):
    rect_min = {
        min(grid.x, max(0, (int)((p.x - max_radius) / BLOCK_X))),
        min(grid.y, max(0, (int)((p.y - max_radius) / BLOCK_Y)))
    }
    rect_max = {
        min(grid.x, max(0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
        min(grid.y, max(0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
    }
    return rect_min, rect_max
def computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix):
    t = transformPoint4x3(p_orig, viewmatrix.flatten())
#     print(t)
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[0] / t[2]
    tytz = t[1] / t[2]
    t[0] = min(limx, max(-limx, txtz)) * t[2]
    t[1] = min(limy, max(-limy, tytz)) * t[2]
    
    # Compute Jacobian matrix J
    J = np.array([[focal_x / t[2], 0.0, -(focal_x * t[0]) / (t[2] ** 2)],
                  [0.0, focal_y / t[2], -(focal_y * t[1]) / (t[2] ** 2)],
                  [0.0, 0.0, 0.0]])
#     print(J)
    
    # Compute transformation matrix T
    T = np.dot(viewmatrix[:3, :3], J)
#     print(T)
    
    Vrk = np.array([[cov3D[0], cov3D[1], cov3D[2]],
                    [cov3D[1], cov3D[3], cov3D[4]],
                    [cov3D[2], cov3D[4], cov3D[5]]])
    cov = np.dot(np.transpose(T), np.dot(np.transpose(Vrk), T))

    # Apply low-pass filter
    cov[0, 0] += 0.3
    cov[1, 1] += 0.3

    return cov[0, 0], cov[0, 1], cov[1, 1]

# from inria
def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = (np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return np.transpose(resized_image,(2, 0, 1))
    else:
        temp = np.expand_dims(resized_image, -1)
        return np.transpose(temp, (2,0,1))  # torch.permute is equivalent to np.transpose
#         return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    
def loadCam(id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

#     if args.resolution in [1, 2, 4, 8]:
#         resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
#     else:  # should be a type that converts to float
    
    resolution = -1
    
    if resolution == -1:
        if orig_w > 1600:
            print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                "If this is not desired, please explicitly specify '--resolution/-r' as 1")
            global_down = orig_w / 1600
        else:
            global_down = 1
    else:
        global_down = orig_w / resolution
        
    scale = float(global_down) * float(resolution_scale)
    resolution = (int(orig_w / scale), int(orig_h / scale))

    print('scale', scale, 'resolution', resolution)
    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    

#     print(resized_image_rgb.shape)
    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
        
#     print(gt_image.shape)
    print("colmap_id=",cam_info.uid, 
          "R=",cam_info.R, 
          "T=", cam_info.T, 
          "FoVx=", cam_info.FovX, 
          "FoVy=", cam_info.FovY, 
#           "image=",gt_image, 
          "gt_alpha_mask=",loaded_mask,
          "image_name=",cam_info.image_name, 
          "uid=",id)
#     return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
#                   FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
#                   image=gt_image, gt_alpha_mask=loaded_mask,
#                   image_name=cam_info.image_name, uid=id)

def cameraList_from_camInfos(cam_infos, resolution_scale):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(id, c, resolution_scale))

    return camera_list

#### from Inria

def calculate_overlap_area(image_width, image_height, point_image, radius):
    rect_center = (0.5*image_width, 0.5*image_height)
    rect_width = image_width
    rect_height = image_height
        
    rect_x_min = rect_center[0] - 0.5*rect_width
    rect_y_min = rect_center[1] - 0.5*rect_height
    
    rect_x_max = rect_center[0] + 0.5*rect_width
    rect_y_max = rect_center[1] + 0.5*rect_height
    
    square_x_min = point_image[0] - radius
    square_y_min = point_image[1] - radius
    
    square_x_max = point_image[0] + radius
    square_y_max = point_image[1] + radius 
    
    # Calculate overlap region
    overlap_x1 = max(rect_x_min, square_x_min)
    overlap_y1 = max(rect_y_min, square_y_min)
    overlap_x2 = min(rect_x_max, square_x_max)
    overlap_y2 = min(rect_y_max, square_y_max)

    # Check if there is overlap
    if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
        # Calculate the overlap area
        overlap_width = overlap_x2 - overlap_x1
        overlap_height = overlap_y2 - overlap_y1
        overlap_area = overlap_width * overlap_height
    else:
        # No overlap
        overlap_area = 0.0
    
    return overlap_area


#### starting point
root = "/Users/yizhen91/Document/UCR/3d_volumetric_video_streaming/"
cam_path = f"{root}/gaussian-splatting/tandt/train/sparse/0/"
img_path = f"{root}/gaussian-splatting/tandt/train/images/"
ply_path = f"{root}/3D_splat/train_135000_180000_v_important_score_sort_progressive_from180k/point_cloud/iteration_5000/point_cloud.ply"

# read camera parameters
cameras_extrinsic_file = os.path.join(cam_path, "images.bin")
cameras_intrinsic_file = os.path.join(cam_path, "cameras.bin")
cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

# read camera poses
cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=img_path)
cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
train_cam_infos = cam_infos
nerf_normalization = getNerfppNorm(train_cam_infos)

pcd = load_ply(ply_path)
scene_info = SceneInfo(point_cloud=pcd,
                       train_cameras=train_cam_infos, # TODO: change to use our pose traces
                       test_cameras=[],
                       nerf_normalization=nerf_normalization,
                       ply_path=ply_path)

cameras_extent = scene_info.nerf_normalization["radius"]

# use first camera pose as the user pose, TODO: need to change to our pose traces
idx = 0
view = scene_info.train_cameras[idx]

tanfovx = math.tan(view.FovX * 0.5)
tanfovy = math.tan(view.FovY * 0.5)

# use it as viewport size for now
image_height = view.image.size[1] 
image_width = view.image.size[0] 

zfar = 100.0
znear = 0.01
trans=np.array([0.0, 0.0, 0.0]) 
scale=1.0
viewmatrix = getWorld2View2(view.R, view.T, trans, scale).transpose()
projmatrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=view.FovX, fovY=view.FovY).transpose()
full_proj_transform = np.matmul(np.expand_dims(viewmatrix, 0), np.expand_dims(projmatrix, 0)).squeeze(0)
campos = np.linalg.inv(viewmatrix)[3, :3]

scaling = np.exp(scene_info.point_cloud.scaling)
cov3Ds = build_covariance_from_scaling_rotation(scaling, scene_info.point_cloud.rotation)

focal_y = image_height / (2.0 * tanfovy)
focal_x = image_width / (2.0 * tanfovx)


# utility calculation, give 0 if the splat is has no overlapping part
splats = []
for i, (point, cov3D, alpha) in enumerate(zip(scene_info.point_cloud.xyz, cov3Ds, scene_info.point_cloud.opacity)):
    
    # frustum culling
    p_view = transformPoint4x3(point, viewmatrix.flatten())
#     if (p_view[2] <= 0.2):
#         continue
    
    # Transform point by projecting
    p_hom = transformPoint4x4(point, full_proj_transform.flatten())
    p_w = 1.0 / (p_hom[3]+ 0.0000001)
    p_proj = [p_hom[0] * p_w, p_hom[1] * p_w, p_hom[2] * p_w]
    point_image = [ndc2Pix(p_proj[0], image_width), ndc2Pix(p_proj[1], image_height)]
    
    # 2D point inside viewport
#     if (point_image[0] < 0 or point_image[0] > image_width or 
#         point_image[1] < 0 or point_image[1] > image_height):
#         continue
    
    # Compute 2D screen-space covariance matrix
    cov = computeCov2D(point, focal_x, focal_y, tanfovx, tanfovy, cov3D, viewmatrix)

    # Invert covariance (EWA algorithm)
    det = (cov[0] * cov[2] - cov[1] * cov[1])
#     if (det == 0.0):
#         continue
        
    det_inv = 1.0 / det
    conic = { cov[2] * det_inv, -cov[1] * det_inv, cov[0] * det_inv }
    mid = 0.5 * (cov[0] + cov[2])
    lambda1 = mid + math.sqrt(max(0.1, mid * mid - det))
    lambda2 = mid - math.sqrt(max(0.1, mid * mid - det))
    ax1 = math.sqrt(abs(lambda1))
    ax2 = math.sqrt(abs(lambda2))
    
    # using a circle to approximate the area of a ellipse
    # lambda1 and lambda2 is the real 2 axes of the ellipse
    # use 3*standard deviation to cover 99.7% of the area
    my_radius = math.ceil(3.0 * math.sqrt(max(lambda1, lambda2))) # radius
    
    # omit pi
    ellipse_area = abs(ax1)*abs(ax2)
    # circle_area = my_radius*my_radius
    
    # overlap with viewport
    overlap = calculate_overlap_area(image_width, image_height, point_image, my_radius)
    
    distance = euclidean_distance(p_view, campos)
    p = {
        'idx': i,
        'view':{'x': p_view[0], 'y': p_view[1], 'z': p_view[2]},
        'uv': {'x': point_image[0], 'y': point_image[1]},
        'axe1': ax1,
        'axe2': ax2,
        'radii': my_radius,
        'ellipse': ellipse_area,
        # 'circle': circle_area,
        'closeness': 1/distance,
        'opacity': alpha[0],
        'overlap': overlap,
        # 'utility': overlap*1/distance,
        'util_op': overlap*1/distance*alpha[0]
    }
    splats.append(p)
    
# get utility for all splats
utility = [p['utility'] for p in splats]