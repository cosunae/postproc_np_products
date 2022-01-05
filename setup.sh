module load python/3.7.4
source /project/g110/spack/user/tsa/spack/share/spack/setup-env.sh

cosmo_eccodes=`spack location -i  cosmo-eccodes-definitions@2.19.0.7%gcc`
eccodes=`spack location -i eccodes@2.19.0%gcc`

export GRIB_DEFINITION_PATH=${cosmo_eccodes}/cosmoDefinitions/definitions/:${eccodes}/share/eccodes/definitions/
