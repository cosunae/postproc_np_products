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
    # thetav.data.visualize(filename='thetav.svg')

    thetav_sum = thetav.isel(generalVerticalLayer=slice(None, None, -1)).cumsum(
        dim="generalVerticalLayer"
    )

    # dask.delayed(thetav_sum.data).visualize(filename='thetasum.svg')

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


def fbrn2(p, t, qv, u, v, hhl, hsurf):
    thetav = fthetav(p, t, qv)
    # thetav.data.visualize(filename='thetav.svg')

    # thetav_sum = thetav.isel(generalVerticalLayer=slice(None, None, -1)).cumsum(
    #     dim="generalVerticalLayer"
    # )

    # dask.delayed(thetav_sum.data).visualize(filename='thetasum.svg')

    # nlevels_xr = xr.DataArray(
    #     data=np.arange(nlevels, 0, -1), dims=["generalVerticalLayer"]
    # )

    brn = (
        pc_g
        * (hhl - hsurf)
        * (thetav - thetav.isel(generalVerticalLayer=79))
        / (u ** 2 + v ** 2)
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
        **chunk_arg
    )

    massds = dss[0]
    uds = cfgrib.open_dataset(
        data_dir + "/lfff00000000",
        backend_kwargs={
            "read_keys": ["cfVarName"],
            "filter_by_keys": {"cfVarName": "u"},
        },
        encode_cf=("time", "geography", "vertical"),
        **chunk_arg
    )
    vds = cfgrib.open_dataset(
        data_dir + "/lfff00000000",
        backend_kwargs={
            "read_keys": ["cfVarName"],
            "filter_by_keys": {"cfVarName": "v"},
        },
        encode_cf=("time", "geography", "vertical"),
        **chunk_arg
    )
    hsurf_ds = cfgrib.open_dataset(
        data_dir + "/lfff00000000c",
        backend_kwargs={
            "read_keys": ["shortName"],
            "filter_by_keys": {"shortName": "HSURF"},
        },
        encode_cf=("time", "geography", "vertical"),
        **chunk_arg
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
        **chunk_arg
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


if __name__ == "__main__":

    scheduler = "synchronous"
    cluster = None
    if scheduler == "distributed":
        from dask.distributed import Client

        from dask_jobqueue import SLURMCluster

        cluster = SLURMCluster(
            queue="postproc",
            cores=2,
            memory="24GB",
            job_extra=["--exclusive"],
        )
        cluster.scale(jobs=4)

        client = None
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

    start = time.time()

    levels = level_range(index, "T")
    nlevels = int(levels[1]) - int(levels[0]) + 1

    start = time.time()
    # with dask.distributed.get_task_stream(
    #     plot="save", filename="task-stream_localc_p16_2t_chunk4.html"
    # ) as ts:
    # pr = cProfile.Profile()
    # pr.enable()
    p, t, qv, hhl, hsurf, u, v = load_data([], chunk_size=4)
    # profile(pr)

    end = time.time()
    print("Time elapsed (load data):", end - start)

    thetav_ds = [
        xr.Dataset(data_vars={"thetav": fthetav(p * (1 + i * 0.01), t, qv)})
        for i in range(10)
    ]

    paths = ["thetav_" + str(i) + ".nc" for i in range(len(thetav_ds))]
    xr.save_mfdataset(thetav_ds, paths=paths, format="NETCDF4")

    # client.profile(filename="dask-profile.html")
    # history = ts.data
    end = time.time()
    print("Time elapsed (compute):", end - start)
