#!/usr/bin/python
import numpy as np
import cfgrib
import xarray as xr
import matplotlib.pyplot as plt
import cProfile
import pstats
import io
from pstats import SortKey
pr = cProfile.Profile()
pr.enable()


def destagger(u, du):
    du[1:-1, :] += u[2:, :] + u[0:-2, :]


def level_range(index, short_name):
    levels = index.subindex(
        filter_by_keys={'shortName': short_name, 'typeOfLevel': 'generalVerticalLayer'}).header_values['level:float']

    return (min(levels), max(levels))


def fthetav(p, t, qv):
    pc_r_d = 287.05
    pc_r_v = 461.51  # Gas constant for water vapour[J kg-1 K-1]
    pc_cp_d = 1005.0
    pc_rvd = pc_r_v / pc_r_d

    pc_rdocp = pc_r_d/pc_cp_d
    pc_rvd_o = pc_rvd - 1.0

    # Reference surface pressure for computation of potential temperature
    p0 = 1.0e5
    return p0 / p ** pc_rdocp * t * (1.+(pc_rvd_o*qv / (1.-qv)))


def brn(dataset):
    pass


if __name__ == '__main__':

    dss = None
    print("LLLLLL")
    index = cfgrib.open_fileindex(
        'grib_files/cosmo-1e/lfff00000000', index_keys=cfgrib.dataset.INDEX_KEYS + ["time", "step"]+["shortName"])

    levels = level_range(index, 't')
    print("LL", levels)
    dss = cfgrib.open_datasets('grib_files/cosmo-1e/lfff00000000',
                               backend_kwargs={'read_keys': ['typeOfLevel', 'gridType'], 'filter_by_keys': {
                                   'typeOfLevel': 'generalVerticalLayer'}})
    print("DONE LOADING")
    massds = dss[0]

    cds = cfgrib.open_dataset('grib_files/cosmo-1e/lfff00000000c',
                              backend_kwargs={'read_keys': ['typeOfLevel', 'gridType'],
                                              'filter_by_keys': {
                                  'typeOfLevel': 'generalVertical'}
                              }
                              )

    hsurf_ds = cfgrib.open_datasets('grib_files/cosmo-1e/lfff00000000c',
                                    backend_kwargs={'read_keys': [
                                        'paramId'], 'filter_by_keys': {'paramId': 3008}}
                                    )

    print("IIII", hsurf_ds)
    print("8888", cds)

    # thetav = None
    print("KKK", levels, len(levels))
    print("IO", cds)
    print("ENDIO")
    nlevels = int(levels[1]) - int(levels[0])
    # for lev in range(nlevels+1):
    #     print("LEV", lev)
    #     thetav_ = fthetav(massds['pres'].isel(generalVerticalLayer=lev), massds['t'].isel(
    #         generalVerticalLayer=lev), massds['q'].isel(generalVerticalLayer=lev))

    #     thetav = xr.concat(
    #         [thetav, thetav_], dim='generalVerticalLayer') if thetav is not None else thetav_

    thetav = fthetav(massds['pres'], massds['t'], massds['q'])

    thetav_sum = thetav.isel(generalVerticalLayer=slice(
        None, None, -1)).cumsum(dim='generalVerticalLayer') / nlevels

    brn = (cds['h'])*(thetav - thetav.isel(generalVerticalLayer=79)) / \
        ((thetav_sum.isel(generalVerticalLayer=0)/80)
         * (massds['u']**2 + massds['v']**2))

    print(thetav)
    # print(dss[0])
    # for lev in range(*(int(x) for x in levels)):
    # gds = xr.Dataset()
    # for lev in range(1, 10):
    #     print("AAAA", lev)
    #     ds = cfgrib.open_datasets(
    #         'grib_files/cosmo-1e/lfff00000000',
    #         backend_kwargs={'read_keys': ['typeOfLevel', 'gridType'], 'filter_by_keys': {
    #             'typeOfLevel': 'generalVerticalLayer', 'level': lev}}  # , 'generalVerticalLayer': 1}},
    #         # chunks={"generalVerticalLayer": 1}
    #         # , engine='cfgrib')
    #     )
    #     massds = ds[0]

    #     thetav(massds, lev, "thetav", gds)

    #     # print(ds['t'].coords['generalVerticalLayer'])
    #     if not dss:
    #         dss = ds
    #     else:
    #         dss = xr.concat([dss, ds], dim='generalVerticalLayer')
    #     # print(rhs)
    #     # dss = cfgrib.xarray_store.merge_datasets(rhs)

    # print(dss)
    # massds = dss[0]

    # theta = thetav(massds)
    # theta = massds
    # thetasum = theta.isel(generalVerticalLayer=slice(
    #     None, None, -1)).cumsum(dim='generalVerticalLayer')

    # print(thetasum)
    # thetasum.isel(generalVerticalLayer=0).plot()
    # integ = t.sum('generalVerticalLayer')
    # , backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})

    # t.isel(generalVerticalLayer=2).plot()
    # integ.plot()
    # plt.show()
    # massv = ds[1]
    # print(massv.attrs)
    # print(ds[0].attrs)
    # print(ds[0])
    print("******************")

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    # print(ds[0]['t'].sel(generalVerticalLayer=slice(0, None)))
    # print(ds.attrs)
    # print(ds[0].attrs['GRIB_gridType'])
    # print(dir(massv))
    # print(ds)

    # print(ds[1])

    # for k in range(0, 80):
    #     destagger(u[:, :, k], du[:, :, k])
