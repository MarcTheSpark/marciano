import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

_current_directory = Path(__file__).parent.resolve()


K_cmb = 2.725 #CMB temperature
K_2_microK = 10**6

try:
    microwave_map = hp.read_map(_current_directory / "downgraded.fits")
except FileNotFoundError:
    print("Downgraded map not found.")
    try:
        print("Trying to load local copy of full map.")
        microwave_map = hp.read_map('COM_CMB_IQU-smica-field-Int_2048_R2.01_full.fits') # read Planck map
    except FileNotFoundError:
        print("Local copy not found; tring to load remote copy.")
        microwave_map = hp.read_map('https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica-field-Int_2048_R2.01_full.fits') # read Planck map
    print("Success. Downgrading map.")
    downsampled_microwave_map = hp.ud_grade(microwave_map, nside_out=hp.get_nside(microwave_map) // 4)
    print("Saving downgraded map for future use.")
    hp.write_map(_current_directory / "downgraded.fits", downsampled_microwave_map)
    microwave_map = downsampled_microwave_map

microwave_map *= K_2_microK
nside = hp.get_nside(microwave_map)


def get_filtered_map(original_map, l_cutoff):
    # Compute the spherical harmonic coefficients (a_lm)
    alm_original = hp.map2alm(original_map)
    lmax = hp.Alm.getlmax(len(alm_original))
    ls = np.arange(lmax + 1)
    Fl = np.exp(-(ls/l_cutoff)**2)

    for l in range(lmax + 1):
        idx = hp.sphtfunc.Alm.getidx(lmax, l, np.arange(min(l, lmax) + 1))
        alm_original[idx] *= Fl[l]

    return hp.alm2map(alm_original, nside)


def get_map_value_at(which_map, theta, phi):
    # Get pixel index
    return which_map[hp.ang2pix(nside, theta, phi)]


try:
    filtered_maps = hp.read_map(_current_directory / "filteredmaps.fits", field=range(14))
except FileNotFoundError:
    filtered_maps = []
    x = 2
    while x < 2000:
        print(f"Generating Filtered map for  l_cutoff={x}")
        filtered_maps.append(get_filtered_map(microwave_map, x))
        x *= 1.7
    print(f"Generated a total of {len(filtered_maps)} maps, including the original")
    hp.write_map(_current_directory / "filteredmaps.fits", filtered_maps)


def get_value_at(sharpness, theta, phi):
    sharpness = max(min(sharpness, 1), 0)
    map_to_use = sharpness * (len(filtered_maps) - 1)
    if map_to_use == int(map_to_use):
        filtered_map = filtered_maps[int(map_to_use)]
    else:
        lower_index = int(map_to_use // 1)
        float_part = map_to_use % 1
        filtered_map = filtered_maps[lower_index] * (1 - float_part) + filtered_maps[lower_index + 1] * float_part
    return get_map_value_at(filtered_map, theta, phi)


if __name__ == '__main__':
    for sharpness in np.linspace(0, 1, 50):
        projection = get_value_at(sharpness, np.linspace(0, np.pi, 180)[:, np.newaxis],  np.linspace(0, 2 * np.pi, 320))
        plt.imshow(projection, cmap='viridis', origin='lower')
        plt.show()

