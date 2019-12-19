from redhawkmaster.rh_inmemory import RedHawkPointCloud
import numpy as np


def acquisition_modelling(
        flying_height=500,
        FOV=75.0,
        SR=225.0,
        pulse_rate=1800000,
        speed_kts=110,
        x_range=100,
        mode="shm",
        density_mode="voxel",
        area_of_circles=1,
        qc="acquisition_modelling.las",
        text_file="aqcuisition_modelling.txt"):  # for 1 meter voxels
    """
    @param flying_height: the height which the aircraft flies above ground in the model
    @param FOV: angle swepth left-to-right in degrees
    @param SR: scan rate (frequency of sweeps) or lines-per-second/LPS in Riegl docs, in Hz
    @param pulse_rate: pulse rate of laser
    @param speed_kts: speed in knots (not mps)
    @param x_range: distance the aircraft flies in the model in meters
    @param mode: mode of oscillation, shm" or "triangular" or "sawtooth"
    @param density_mode:  "none" or "voxel" or "radial" - "none" means we don't calculate densities,
    "voxel" means we use square (x,y)-voxels and "radial" means we use a circle over every point
    Note "circle" is a lot slower.
    @param area_of_circles: area of circles to use if density_mode == "radial" (does nothing otherwise)
    @return:
    """
    if text_file:
        f = open(text_file, "a+")

    def record(statement):
        if text_file:
            f.write(statement)

    speed_mps = 0.514444 * speed_kts  # convert speed to mps
    flight_time = x_range / speed_mps  # figure out how long we will fly for
    num_pts = int(np.ceil(flight_time * pulse_rate))  # find out how many points we need for flight time
    times = np.arange(num_pts) * (1 / pulse_rate)  # what are the times for points?

    # make the aircraft position
    x_ = np.empty(num_pts, dtype=float)
    z_ = np.empty(num_pts, dtype=float)

    x_[:] = 0  # assume aircraft flies parallel to y axis
    y_ = times * speed_mps
    z_[:] = flying_height

    # make the scan position
    x = np.empty(num_pts, dtype=float)
    y = np.empty(num_pts, dtype=float)
    z = np.empty(num_pts, dtype=float)

    angular_amplitude = (FOV / 2) * (np.pi / 180)  # formulas subject to change when more understanding gained
    frequency = SR  # formulas subject to change when more understanding gained

    if mode == "shm":
        function = lambda x: np.cos(2 * np.pi * x)
    if mode == "triangular":
        function = lambda x: (2 * (2 * x - np.floor(2 * x)) - 1) * ((-1) ** ((np.floor(2 * x)).astype(int)))
    if mode == "sawtooth":
        function = lambda x: 2 * (x - np.floor(x)) - 1

    theta = angular_amplitude * function(times * frequency)

    x = z_ * np.tan(theta)
    y[:] = y_[:]
    z[:] = 0

    coords = np.stack((x, y, z), axis=1)
    differences = coords[:-1, :] - coords[1:, :]
    distances_across_track = np.sqrt(np.sum(differences ** 2, axis=1))
    record("")
    record(
        f"max pt spacing across track {np.max(distances_across_track[distances_across_track < 0.5 * (max(x) - min(x))])}")
    record(f"max pt spacing along track speed_{mps / frequency}")
    record("")
    record(f"swath width {np.round(max(x) - min(x), 2)")
    record(f"time elapsed {np.round(max(times) - min(times), 2)}")
    record("")
    unq, ind, inv, cnt = np.unique((np.floor(coords)).astype(int), return_index=True, return_inverse=True,
                                   return_counts=True, axis=0)
    record(f"voxel mode lowest ppm {min(cnt)}")
    record(f"voxel mode highest ppm {max(cnt)}")
    record(f"voxel mode ppm: mean, median = {np.mean(cnt[inv])}, {np.median(cnt[inv])}")
    unq1, ind1, inv1, cnt1 = np.unique((np.floor(coords / 0.1)).astype(int), return_index=True, return_inverse=True,
                                       return_counts=True, axis=0)
    frame = {
        'A': inv,
        'B': inv1
    }
    df = pd.DataFrame(frame)
    nunique_vals = ((df.groupby('A')['B'].nunique()).values)[inv]
    record("")
    record(f"lowest coverage {min(nunique_vals)} percent")
    record(f"highest coverage {max(nunique_vals)} percent")
    record(f"avg coverage {np.round(np.mean(nunique_vals), 2)} percent")
    record("")
    if density_mode == "voxel":
        ppm = cnt[inv]

    if density_mode == "none":
        ppm = np.zeros(num_pts)

    if density_mode == "radial":
        radius = np.sqrt(area_of_circles / np.pi)
    done = np.zeros(num_pts, dtype=bool)
    ppm = np.empty(num_pts, dtype=int)
    k = 1
    while not (done.all()):
        nhbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(coords[:, :2])
        distances, indices = nhbrs.kneighbors(coords[~ done, :2])  # (num_pts,k)
        k_found = distances[:, -1] >= radius
        not_done_save = ~done
        done[~done] = k_found
        ppm_not_done = ppm[not_done_save]
        ppm_not_done[k_found] = k
        ppm[not_done_save] = ppm_not_done
        k += 1
    record(f"radial mode lowest ppm {min(ppm)}")
    record(f"radial mode highest ppm {max(ppm)}")
    record(f"radial mode avg ppm {np.mean(ppm)}")


    newHeader = Header()
    newFile = File("als_model.las", mode="w", header=newHeader)
    newFile.header.scale = [0.0001, 0.0001, 0.0001]  # 4 dp
    newFile.x = x
    newFile.y = y
    newFile.z = z
    newFile.intensity = ppm
    newFile.classification = inv % 32
    newFile.close()

    newFile = RedHawkPointCloud(num_pts)
    newFile.x = x
    newFile.y = y
    newFile.z = z
    newFile.intensity = ppm

    if qc:
        newHeader = Header()
        outFile = File(qc, mode = "w")
        outFile.header.scale = [0.0001, 0.0001, 0.0001]  # 4 dp
        outFile.x = x
        outFile.y = y
        outFile.z = z

    return newFile