from redhawkmaster.rh_inmemory import RedHawkPointCloud


def acquisition_modelling(
        flying_height=500,
        FOV=75.0,  # degrees, for full left-right angle
        SR=225.0,  # scan rate, Hz, lines per second, LPS
        pulse_rate=1800000,
        speed_kts=110,  # enter this in in knots
        x_range=100,  # meters
        mode="shm",  # "shm" or "triangular" or "sawtooth"
        density_mode="voxel",  # "none" or "voxel" or "radial"
        scale=1):  # for 1 meter voxels
    """
    @param flying_height:
    @param FOV:
    @param SR:
    @param pulse_rate:
    @param speed_kts:
    @param x_range:
    @param mode:
    @param density_mode:
    @param scale:
    @return:
    """
    def function()
    speed_mps = 0.514444 * speed_kts  # convert speed to mps
    flight_time = x_range / speed_mps  # figure out how long we will fly for
    num_pts = int(np.ceil(flight_time * pulse_rate))  # find out how many points we need for flight time
    times = np.arange(num_pts) * (1 / pulse_rate)  # what are the times for points?

    # make the aircraft position
    x_ = np.empty(num_pts, dtype=float)
    y_ = np.empty(num_pts, dtype=float)
    z_ = np.empty(num_pts, dtype=float)

    x_[:] = 0  # assume aircraft flies parrallel to y axis
    y_ = times * speed_mps
    z_[:] = flying_height

    # make the scan position
    x = np.empty(num_pts, dtype=float)
    y = np.empty(num_pts, dtype=float)
    z = np.empty(num_pts, dtype=float)
    intensity = np.empty(num_pts, dtype=int)

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
    print("")
    print("max pt spacing across track",
          np.max(distances_across_track[distances_across_track < 0.5 * (max(x) - min(x))]))
    print("max pt spacing along track", speed_mps / frequency)
    print("")
    print("swath width", np.round(max(x) - min(x), 2))
    print("time elapsed", np.round(max(times) - min(times), 2))
    print("")
    unq, ind, inv, cnt = np.unique((np.floor(coords)).astype(int), return_index=True, return_inverse=True,
                                   return_counts=True, axis=0)
    print("voxel mode lowest ppm", min(cnt))
    print("voxel mode highest ppm", max(cnt))
    print("voxel mode avg ppm", np.mean(cnt[inv]), np.median(cnt[inv]))
    unq1, ind1, inv1, cnt1 = np.unique((np.floor(coords / 0.1)).astype(int), return_index=True, return_inverse=True,
                                       return_counts=True, axis=0)
    frame = {
        'A': inv,
        'B': inv1
    }
    df = pd.DataFrame(frame)
    nunique_vals = ((df.groupby('A')['B'].nunique()).values)[inv]
    print("")
    print("lowest coverage", min(nunique_vals), "percent")
    print("highest coverage", max(nunique_vals), "percent")
    print("avg coverage", np.round(np.mean(nunique_vals), 2), "percent")
    print("")
    if density_mode == "voxel":
        ppm = cnt[inv]

    if density_mode == "none":
        ppm = np.zeros(num_pts)

    if density_mode == "radial":
        radius = np.sqrt(scale / np.pi)
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
        print("radial mode lowest ppm", min(ppm))
        print("radial mode highest ppm", max(ppm))
        print("radial mode avg ppm", np.mean(ppm))

    newHeader = Header()
    newFile = File("als_model.las", mode="w", header=newHeader)
    newFile.header.scale = [0.0001, 0.0001, 0.0001]  # 4 dp
    newFile.x = x
    newFile.y = y
    newFile.z = z
    newFile.intensity = ppm
    newFile.classification = inv % 32
    newFile.close()

    outfile = RedHawkPointCloud
