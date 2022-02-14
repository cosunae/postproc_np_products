#!/usr/bin/python
import numpy as np
import cfgrib
import xarray as xr
import matplotlib.pyplot as plt
import cProfile
import pstats
import io
import time
from pstats import SortKey
import dask
from collections.abc import Iterable
import itertools


dask.config.set(scheduler="threads", num_workers=8)

# pr = cProfile.Profile()
# pr.enable()

pc_g = 9.80665


def destagger(u, du):
    du[1:-1, :] += u[2:, :] + u[0:-2, :]


def level_range(index, short_name):
    # print(index.header_values)
    levels = index.subindex(
        filter_by_keys={"shortName": short_name, "typeOfLevel": "generalVerticalLayer"}
    ).header_values["level:float"]

    return (min(levels), max(levels))


def fthetavi(p, t, qv):
    return [fthetav(p * (1 + i * 0.01), t, qv) for i in range(10)]


def fthetav(p, t, qv):
    pc_r_d = 287.05
    pc_r_v = 461.51  # Gas constant for water vapour[J kg-1 K-1]
    pc_cp_d = 1005.0
    pc_rvd = pc_r_v / pc_r_d

    pc_rdocp = pc_r_d / pc_cp_d
    pc_rvd_o = pc_rvd - 1.0

    # Reference surface pressure for computation of potential temperature
    p0 = 1.0e5
    return (p0 / p) ** pc_rdocp * t * (1.0 + (pc_rvd_o * qv / (1.0 - qv)))


def fbrn(p, t, qv, u, v, hhl, hsurf):
    thetav = fthetav(p, t, qv)

    thetav_sum = thetav.isel(generalVerticalLayer=slice(None, None, -1)).cumsum(
        dim="generalVerticalLayer"
    )

    nlevels_xr = xr.DataArray(
        data=np.arange(nlevels, 0, -1), dims=["generalVerticalLayer"]
    )

    brn = (
        pc_g
        * (hhl - hsurf)
        * (thetav - thetav.isel(generalVerticalLayer=79))
        / ((thetav_sum / nlevels_xr) * (u ** 2 + v ** 2))
    )
    return brn


def profile(pr):
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


def load_data(fields, chunk_size=10):

    chunk_arg = {}
    if chunk_size:
        chunk_arg = {"chunks": {"generalVerticalLayer": chunk_size}}

    dss = cfgrib.open_datasets(
        data_dir + "/lfff00000000",
        backend_kwargs={
            "read_keys": ["typeOfLevel", "gridType"],
            "filter_by_keys": {"typeOfLevel": "generalVerticalLayer"},
        },
        encode_cf=("time", "geography", "vertical"),
        **chunk_arg,
    )

    massds = dss[0]
    uds = cfgrib.open_dataset(
        data_dir + "/lfff00000000",
        backend_kwargs={
            "read_keys": ["cfVarName"],
            "filter_by_keys": {"cfVarName": "u"},
        },
        encode_cf=("time", "geography", "vertical"),
        **chunk_arg,
    )
    vds = cfgrib.open_dataset(
        data_dir + "/lfff00000000",
        backend_kwargs={
            "read_keys": ["cfVarName"],
            "filter_by_keys": {"cfVarName": "v"},
        },
        encode_cf=("time", "geography", "vertical"),
        **chunk_arg,
    )
    hsurf_ds = cfgrib.open_dataset(
        data_dir + "/lfff00000000c",
        backend_kwargs={
            "read_keys": ["shortName"],
            "filter_by_keys": {"shortName": "HSURF"},
        },
        encode_cf=("time", "geography", "vertical"),
        **chunk_arg,
    )
    if chunk_size:
        chunk_arg = {"chunks": {"generalVertical": chunk_size}}

    cds = cfgrib.open_dataset(
        data_dir + "/lfff00000000c",
        backend_kwargs={
            "read_keys": ["typeOfLevel", "gridType"],
            "filter_by_keys": {"typeOfLevel": "generalVertical"},
        },
        encode_cf=("time", "geography", "vertical"),
        **chunk_arg,
    )
    hhl = cds["HHL"].rename({"generalVertical": "generalVerticalLayer"})

    return (
        massds["P"],
        massds["T"],
        massds["QV"],
        hhl,
        hsurf_ds["HSURF"],
        uds["U"],
        vds["V"],
    )


import itertools


def split_by_chunks(dataset):
    chunk_slices = {}

    for dim, chunks in dataset.chunks.items():
        slices = []
        start = 0
        for chunk in chunks:
            if start >= dataset.sizes[dim]:
                break
            stop = start + chunk
            slices.append(slice(start, stop))
            start = stop
        chunk_slices[dim] = slices
    for slices in itertools.product(*chunk_slices.values()):
        selection = dict(zip(chunk_slices.keys(), slices))
        yield dataset[selection]


def create_filepath(ds, root_path="."):
    """
    Generate a filepath when given an xarray dataset
    """
    varname = [x for x in ds.data_vars][0]

    start = ds.generalVerticalLayer.data[0]
    end = ds.generalVerticalLayer.data[-1]
    filepath = f"{root_path}/{varname}_{start}_{end}.nc"
    return filepath


# implementation extracted from https://github.com/dask/dask/issues/1887
def map_blocks(func, *arrs, nout=None, pure=False):
    arr0 = arrs[0]
    nr_common_dims = min([len(arr.shape) for arr in arrs])
    common_chunks = arr0.chunks[:nr_common_dims]
    arr_collections = [a.to_delayed() for a in arrs]
    common_blockshape = arr_collections[0].shape
    flat_arr_collections = [np.ravel(a, order="C") for a in arr_collections]
    blockshapes_common_dims = list(
        itertools.product(*common_chunks)
    )  # Ordering needs to fit to np.ravel
    dfunc = dask.delayed(func, nout=nout, pure=pure)
    flat_res_collections = [dfunc(*args) for args in zip(*flat_arr_collections)]
    if nout is not None:
        flat_res_collections = list(zip(*flat_res_collections))
    else:
        flat_res_collections = [flat_res_collections]
    # Unsolved problem: if `func` creates or destroys dimensions,
    # we can't know it automatically before evaluation. Refer also to `dask.array.Array.map_blocks`
    res_blockss = [
        [
            dask.array.from_delayed(c, shape=s, dtype=arr0.dtype)
            for c, s in zip(fc, blockshapes_common_dims)
        ]
        for fc in flat_res_collections
    ]

    # for iteridx, blocklen in enumerate(common_blockshape):
    #     res_blockss = [
    #         [
    #             dask.array.concatenate(
    #                 y[idx * blocklen : (idx + 1) * blocklen], axis=(-1 - iteridx)
    #             )
    #             for idx in range(len(y) // blocklen)
    #         ]
    #         for y in res_blockss
    #     ]
    # res_blockss = tuple([r[0] for r in res_blockss])
    # if nout is None:
    #     return res_blockss[0]
    return res_blockss


if __name__ == "__main__":

    scheduler = "distributed"
    cluster = None
    if scheduler == "distributed":
        from dask.distributed import Client

        from dask_jobqueue import SLURMCluster

        cluster = SLURMCluster(
            queue="postproc",
            cores=16,
            memory="24GB",
            job_extra=["--exclusive"],
        )
        cluster.scale(jobs=2)
        client = Client(cluster)
    elif scheduler == "localcluster":
        from dask.distributed import Client, LocalCluster

        cluster = LocalCluster(n_workers=16, threads_per_worker=2)
        client = Client(cluster)
    elif scheduler == "threads":
        from multiprocessing.pool import ThreadPool

        dask.config.set(pool=ThreadPool(1))
        # dask.config.set(scheduler="threads")
    elif scheduler == "synchronous":
        dask.config.set(
            scheduler="synchronous"
        )  # overwrite default with single-threaded scheduler
    elif scheduler == "processes":
        from multiprocessing.pool import Pool

        dask.config.set(pool=Pool(2))

    data_dir = "/scratch/cosuna/postproc_np_products/grib_files/cosmo-1e/"

    index = cfgrib.open_fileindex(
        "grib_files/cosmo-1e/lfff00000000",
        index_keys=cfgrib.dataset.INDEX_KEYS
        + ["time", "step"]
        + ["shortName", "paramId"],
    )

    if cluster:
        while cluster.status != dask.distributed.core.Status.running:
            time.sleep(1)
        print("CLUSTER ALLOCATED", cluster.status, cluster.workers)

        import sys

        sys.stdin.read(1)

    start = time.time()

    levels = level_range(index, "T")
    nlevels = int(levels[1]) - int(levels[0]) + 1

    start = time.time()
    # with dask.distributed.get_task_stream(
    #     plot="save", filename="task-stream_localc_p16_2t_chunk4.html"
    # ) as ts:
    # pr = cProfile.Profile()
    # pr.enable()
    p, t, qv, hhl, hsurf, u, v = load_data([], chunk_size=10)
    # profile(pr)

    end = time.time()
    print("Time elapsed (load data):", end - start)

    p = p.drop(["valid_time", "time", "step"])
    t = t.drop(["valid_time", "time", "step"])
    qv = qv.drop(["valid_time", "time", "step"])

    pd = p.data.to_delayed()
    td = t.data.to_delayed()
    qvd = qv.data.to_delayed()

    pd = pd.reshape(pd.size)
    td = td.reshape(td.size)
    qvd = qvd.reshape(qvd.size)

    # with dask.distributed.get_task_stream(
    #     plot="save", filename="task-stream_dist_c16_2n_chunk4_map_block.html"
    # ) as ts:
    start = time.time()

    res = map_blocks(fthetavi, p.data, t.data, qv.data, nout=10, pure=True)

    res_ds = [
        xr.Dataset(data_vars={"thetav" + str(i): (p.dims, block)})
        for i, vararr in enumerate(res)
        for block in vararr
    ]

    paths = ["path" + str(i) + ".nc" for i in range(len(res_ds))]

    writes = [
        xr.save_mfdataset([dat], paths=[apath], format="NETCDF4", compute=False)
        for dat, apath in zip(res_ds, paths)
    ]

    dask.compute(*writes)

    end = time.time()

    print("Time elapsed (thetav):", end - start)
