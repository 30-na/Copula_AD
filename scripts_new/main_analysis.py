import sys
sys.path.append("c:/Users/Sina.Mokhtar.XLSCIENTIFIC/Documents/Problems/Copula_AD/scripts_new")

from earthquake_data import EarthquakeData

eq = EarthquakeData(base_path="data\earthquake")
#eq.plot_segments_per_file(station_name="afi", save_dir="figures/afi_segments")
eq.plot_segments_per_file(station_name="funa", save_dir="figures/funa_segments")
eq.plot_segments_per_file(station_name="rao", save_dir="figures/rao_segments")
eq.plot_segments_per_file(station_name="tara", save_dir="figures/tara_segments")
#afi_df = eq.load_station("afi")

#print(afi_df.head())
