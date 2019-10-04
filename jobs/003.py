from redhawkmaster import rh_io
from redhawkmaster.las_modules import las_range, duplicate_attr, flightline_point_counter, rh_assign, virus


f002 = rh_io.las_input('/home/mcus/Downloads/TestArea.las', mode='r')
backup_input = f002
f002_000_prep = duplicate_attr(f002, 'intensity', 'intensity_snapshot', 'Snapshot of intensity', 5)

mask_000 = las_range(f002_000_prep.intensity, start=630, end=730, reverse=False)
f002_040_Intensity_630_730 = rh_io.las_output('ILIJA_FlightlineTest_Tile000_job003_Intensity_630t0730.las',
                                              f002_000_prep, mask_000)

intensity_040 = flightline_point_counter(f002_040_Intensity_630_730, clip=1.0, nh=40)
f002_040_Intensity_630_730.intensity = intensity_040

mask_060 = las_range(f002_040_Intensity_630_730.intensity, start=40, reverse=False)
f002_060_python_mild = rh_io.las_output('ILIJA_FlightlineTest_Tile000_job003_060_python_mild.las',
                                        f002_040_Intensity_630_730, mask_060)

intensity_090 = flightline_point_counter(f002_060_python_mild, clip=0.25, nh=80)
f002_060_python_mild.intensity = intensity_090

mask_100 = las_range(f002_060_python_mild.intensity, start=80, reverse=False)
f002_060_python_strong = rh_io.las_output('ILIJA_FlightlineTest_Tile000_job003_100_python_strong.las',
                                          f002_060_python_mild, mask_100)

mask_020 = las_range(f002_000_prep.intensity, start=630, end=730, reverse=True)
f002_040_NotIntensity_630_730 = rh_io.las_output('ILIJA_FlightlineTest_Tile000_job003_!Intensity_630t0730.las',
                                                 f002_000_prep, mask_020)


f002_040_NotIntensity_630_730.Classification = rh_assign(f002_040_NotIntensity_630_730.Classification, 1)

mask_060_noise = las_range(f002_040_Intensity_630_730.intensity, start=40, reverse=True)
f002_060_python_mild_noise_merge = rh_io.las_output('ILIJA_FlightlineTest_Tile000_job003_mild_python_noise.las',
                                                    f002_040_Intensity_630_730, mask_060_noise)
f002_060_python_mild_noise_merge.Classification = rh_assign(f002_060_python_mild_noise_merge.Classification, 1)

mask_090_noise = las_range(f002_060_python_mild.intensity, start=80, reverse=True)
f002_090_python_strong_noise_merge = rh_io.las_output('ILIJA_FlightlineTest_Tile000_job003_strong_python_noise.las',
                                                      f002_060_python_mild, mask_090_noise)
f002_090_python_strong_noise_merge.Classification = rh_assign(f002_090_python_strong_noise_merge.Classification, 1)


f002_040_NotIntensity_630_730.close()
f002_060_python_strong.close()
f002_090_python_strong_noise_merge.close()
f002_060_python_mild_noise_merge.close()

rh_io.merge(['ILIJA_FlightlineTest_Tile000_job003_!Intensity_630t0730.las',
             'ILIJA_FlightlineTest_Tile000_job003_100_python_strong.las',
             'ILIJA_FlightlineTest_Tile000_job003_mild_python_noise.las',
             'ILIJA_FlightlineTest_Tile000_job003_strong_python_noise.las'],
            'ILIJA_FlightlineTest_Tile000_job003_merged.las')

f003_merged = rh_io.las_input('ILIJA_FlightlineTest_Tile000_job003_merged.las', mode='rw')
f003_merged.intensity = f002_000_prep.intensity_snapshot
#f003_000_991prep = duplicate_attr(f003_merged, 'intensity_snapshot', 'intensity', 'Snapshot of intensity', 5)

class_merged = virus(f003_merged, clip=0.50, num_itter=1, classif=10)
f003_merged.classification = class_merged

f003_merged.close()
f002_060_python_mild.close()
f002_040_Intensity_630_730.close()
f002_000_prep.close()
