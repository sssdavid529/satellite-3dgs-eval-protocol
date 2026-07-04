import dsmr
import iio
import rasterio
import numpy as np
import os

from typing import Optional

def dsm_pointwise_diff(in_dsm_path: str, gt_dsm_path: str, dsm_metadata, water_mask_path: Optional[str] = None, vis_mask_path: Optional[str] = None, tree_mask_path: Optional[str] = None):
    """
    in_dsm_path is a string with the path to the NeRF generated dsm
    gt_dsm_path is a string with the path to the reference lidar dsm
    bbx_metadata is a 4-valued array with format (x, y, s, r)
    where [x, y] = offset of the dsm bbx, s = width = height, r = resolution (m per pixel)
    """

    # read gt dsm
    with rasterio.open(gt_dsm_path, "r") as f:
        gt_dsm = f.read()[0, :, :]

    # read dsm metadata
    xoff, yoff = dsm_metadata[0], dsm_metadata[1]
    xsize, ysize = int(dsm_metadata[2]), int(dsm_metadata[2])
    resolution = dsm_metadata[3]

    # define projwin for gdal translate
    ulx, uly, lrx, lry = xoff, yoff + ysize * resolution, xoff + xsize * resolution, yoff

    # load the corresponding in_dsm crop
    with rasterio.open(in_dsm_path, 'r') as f:
        profile = f.profile
        transform = f.transform

        # Compute the window corresponding to the window
        ulx, uly = ~transform * (ulx, uly)
        lrx, lry = ~transform * (lrx, lry)
        window = rasterio.windows.Window(ulx, uly, lrx-ulx, lry-uly)

        pred_dsm = f.read(1, window=window)

        profile.update(height=lry-uly, width=lrx-ulx, transform=f.window_transform(window))

    if water_mask_path is not None:
        with rasterio.open(water_mask_path, "r") as f:
            mask = f.read()[0, :, :]
            water_mask = mask.copy()
            water_mask[mask != 9] = 0
            water_mask[mask == 9] = 1
            water_mask = water_mask.astype(bool)
        if ("CLS.tif" in water_mask_path) and (os.path.exists(water_mask_path.replace("CLS.tif", "WATER.png"))):
            #print("found alternative water mask!")
            mask = iio.read(water_mask_path.replace("CLS.tif", "WATER.png"))[..., 0]
            water_mask = mask == 0

        water_mask = water_mask[:pred_dsm.shape[0], :pred_dsm.shape[1]]
        pred_dsm[water_mask] = np.nan

    # Use visibility mask if available
    if vis_mask_path is not None:
        vis_mask = rasterio.open(vis_mask_path).read()[0,...] > 0.5
        pred_dsm[vis_mask] = np.nan

    if tree_mask_path is not None:
        tree_mask = rasterio.open(tree_mask_path).read()[0,...] > 0.5
        pred_dsm[np.logical_not(tree_mask)] = np.nan

    # register and compute mae
    transform = dsmr.compute_shift(gt_dsm, pred_dsm, scaling=False)
    pred_rdsm = dsmr.apply_shift(pred_dsm, *transform)

    h = min(pred_rdsm.shape[0], gt_dsm.shape[0])
    w = min(pred_rdsm.shape[1], gt_dsm.shape[1])
    max_gt_alt = gt_dsm.max()
    min_gt_alt = gt_dsm.min()
    pred_rdsm = np.clip(pred_rdsm, min_gt_alt - 10, max_gt_alt + 10)
    err = pred_rdsm[:h, :w] - gt_dsm[:h, :w]

    return err, pred_rdsm, profile

def compute_mae(pred_dsm_path: str, gt_dir: str, aoi_id: str, enable_vis_mask: bool = True, filter_tree: bool = True):
    gt_dsm_path = os.path.join(gt_dir, "{}_DSM.tif".format(aoi_id))

    # if a v2 exists, use it
    if os.path.exists(os.path.join(gt_dir, "{}_CLS_v2.tif".format(aoi_id))):
        gt_seg_path = os.path.join(gt_dir, "{}_CLS_v2.tif".format(aoi_id))
    else:
        gt_seg_path = os.path.join(gt_dir, "{}_CLS.tif".format(aoi_id))

    assert os.path.exists(gt_dsm_path), f"{gt_dsm_path} not found"
    assert os.path.exists(gt_seg_path), f"{gt_seg_path} not found"

    # Check whether a txt file exists. If so, use it
    # Otherwise assume that the DSM is geolocalized
    if os.path.exists(os.path.join(gt_dir, "{}_DSM.txt".format(aoi_id))):
        # Mostly DFC2019 scenes
        gt_roi_path = os.path.join(gt_dir, "{}_DSM.txt".format(aoi_id))
        gt_roi_metadata = np.loadtxt(gt_roi_path)
    else:
        # mostly IARPA scenes
        src = rasterio.open(gt_dsm_path)
        gt_roi_metadata = np.array([src.bounds.left, src.bounds.bottom, min(src.height, src.width), src.res[0]])
        del src

    if enable_vis_mask:
        vis_mask_path = os.path.join(os.path.dirname(__file__), f"vis_masks/{aoi_id}.tif".format(aoi_id))
        if not os.path.exists(vis_mask_path):
            vis_mask_path = None
    else:
        vis_mask_path = None

    if filter_tree:
        tree_mask_path = os.path.join(os.path.dirname(__file__), f"tree_masks/{aoi_id}.png".format(aoi_id))
        if not os.path.exists(tree_mask_path):
            tree_mask_path = None
    else:
        tree_mask_path = None

    diff, rdsm, profile = dsm_pointwise_diff(pred_dsm_path, gt_dsm_path, gt_roi_metadata, water_mask_path=None, vis_mask_path=vis_mask_path, tree_mask_path=tree_mask_path)

    mae = np.nanmean(abs(diff.ravel()))
    return mae, diff, rdsm, profile


def compute_mae_and_save_dsm_diff(pred_dsm_path: str, gt_dir: str, aoi_id: str, out_dir: Optional[str] = None, enable_vis_mask: bool = False, filter_tree: bool = False) -> float:
    mae, diff, rdsm, profile = compute_mae(pred_dsm_path, gt_dir, aoi_id, enable_vis_mask=enable_vis_mask, filter_tree=filter_tree)
    if out_dir is not None:
        rdsm_diff_path = os.path.join(out_dir, "{}_rdsm_diff_{}.tif".format(aoi_id, mae))
        with rasterio.open(rdsm_diff_path, 'w', **profile) as dst:
            dst.write(diff, 1)

        rdsm_path = os.path.join(out_dir, "{}_rdsm.tif".format(aoi_id))
        with rasterio.open(rdsm_path, 'w', **profile) as dst:
            dst.write(rdsm, 1)

    return mae

if __name__ == "__main__":
    import tyro
    print("MAE:", tyro.cli(compute_mae_and_save_dsm_diff))
