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
pr = cProfile.Profile()
pr.enable()

pc_g = 9.80665

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
    return (p0 / p) ** pc_rdocp * t * (1.+(pc_rvd_o*qv / (1.-qv)))


def brn(dataset):
    pass

def profile():
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


if __name__ == '__main__':


    start = time.time()
    index = cfgrib.open_fileindex(
        'grib_files/cosmo-1e/lfff00000000', index_keys=cfgrib.dataset.INDEX_KEYS + ["time", "step"]+["shortName", "paramId"])

    levels = level_range(index, 't')

    dss = cfgrib.open_datasets('grib_files/cosmo-1e/lfff00000000',
                               backend_kwargs={'read_keys': ['typeOfLevel', 'gridType'], 'filter_by_keys': {
                                   'typeOfLevel': 'generalVerticalLayer'}})
    massds = dss[0]

    uds = cfgrib.open_dataset('grib_files/cosmo-1e/lfff00000000',
                               backend_kwargs={'read_keys': ['cfVarName'], 'filter_by_keys': {
                                   'cfVarName': 'u'}})
    vds = cfgrib.open_dataset('grib_files/cosmo-1e/lfff00000000',
                               backend_kwargs={'read_keys': ['cfVarName'], 'filter_by_keys': {
                                   'cfVarName': 'v'}})

    cds = cfgrib.open_dataset('grib_files/cosmo-1e/lfff00000000c',
                              backend_kwargs={'read_keys': ['typeOfLevel', 'gridType'],
                                              'filter_by_keys': {
                                  'typeOfLevel': 'generalVertical'}
                              }
                              )
    hhl = cds['h'].rename({'generalVertical':'generalVerticalLayer'})

    hsurf_ds = cfgrib.open_dataset('grib_files/cosmo-1e/lfff00000000c',
                                    backend_kwargs={'read_keys': [
                                        'shortName'], 'filter_by_keys': {'shortName': 'HSURF'}}
                                    )
    
    nlevels = int(levels[1]) - int(levels[0])+1
    # for lev in range(nlevels+1):
    #     print("LEV", lev)
    #     thetav_ = fthetav(massds['pres'].isel(generalVerticalLayer=lev), massds['t'].isel(
    #         generalVerticalLayer=lev), massds['q'].isel(generalVerticalLayer=lev))

    #     thetav = xr.concat(
    #         [thetav, thetav_], dim='generalVerticalLayer') if thetav is not None else thetav_

    P = massds['pres']
    T = massds['t']
    QV = massds['q']

    thetav = fthetav(P, T, QV)

    thetav_sum = thetav.isel(generalVerticalLayer=slice(
        None, None, -1)).cumsum(dim='generalVerticalLayer') 

    nlevels_xr =xr.DataArray(data=np.arange(nlevels,0,-1), dims=["generalVerticalLayer"])

    brn = pc_g* (hhl-hsurf_ds['h'])*(thetav - thetav.isel(generalVerticalLayer=79)) / \
        ( (thetav_sum/nlevels_xr)*(uds['u']**2 + vds['v']**2))
    brn.name = "BRN"

    brn.isel(generalVerticalLayer=slice(37,80,1)).to_netcdf(path="brn_out.nc")

    end = time.time()

    print("Time elapsed:", end - start)
