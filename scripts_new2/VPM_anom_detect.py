from EDA.my_functions import *



def Anomaly_heatmap_VPM(data, startDate, endDate, order):
    path = r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\VPM"
    file = read_file_category_VPM(data)
    variables = file.columns.drop(["ReceivedTime"])

    windows_anomaly_df = pd.DataFrame()
    windows_dist_df = pd.DataFrame()

    for variable in variables:

        df_raw=file[variable]
        resample_interval = "1min"
        resample = df_raw[~df_raw.index.duplicated(keep='first')].resample(resample_interval).mean().interpolate(method="linear")
        resample = filter_by_date(df=resample, fromDate=startDate, toDate=endDate)
        data = resample
        window_size = pd.Timedelta(minutes=94)  
        overlap_size = pd.Timedelta(minutes=23)  
        windows = create_overlapping_windows(data, window_size, overlap_size)[:-1]

        if order == "Auto":
            try:
                window_coeff = calculate_window_parameter_autoArima_aic(windows)
            except np.linalg.LinAlgError:
                print(f"Skipping variable {variable} due to LU decomposition error.")
                continue
        else:
            try:
                window_coeff = calculate_window_parameter_Arima(windows, order)
            except np.linalg.LinAlgError:
                print(f"Skipping variable {variable} due to LU decomposition error.")
                continue
    
        if windows_anomaly_df.empty:
            windows_anomaly_df = pd.DataFrame(index=window_coeff.index)
        if windows_dist_df.empty:
            windows_dist_df = pd.DataFrame(index=window_coeff.index)

        sigma2 = window_coeff["sigma2"]
        std = np.std(sigma2)
        sigma_dist = sigma2/std
        windows_dist_df.loc[window_coeff.index, variable] = sigma_dist

        threshold = std*3
        mu = np.mean(sigma2)
        sigma_anomaly = sigma2[(sigma2 > mu + threshold) | (sigma2 < mu - threshold)]
        window_anomaly = sigma_anomaly.index

        windows_anomaly_df[variable] = 0
        windows_anomaly_df.loc[window_anomaly, variable] = 1

    #windows_dist_df.to_csv(os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\VPM\SPC", f"{data}_heatmap_{order}_07.csv"))
    #windows_anomaly_df.to_csv(os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\VPM\SPC", f"{data}_anomaly_{order}_07.csv"))
    
    return(windows_dist_df, windows_anomaly_df)

windows_dist_df, windows_anomaly_df = Anomaly_heatmap_VPM(data="AttDet", startDate="2023-09-01", endDate="2023-09-30", order=(1,1,1))

windows_dist_df.columns
windows_dist_df_edit = windows_dist_df.copy()
#windows_dist_df_edit = np.log(windows_dist_df_edit+1.01)
windows_dist_df_edit.index = windows_dist_df.index.date

windows_dist_df_edit = windows_dist_df_edit.drop(columns=["MeasRateValid"])
#windows_dist_df_edit = windows_dist_df_edit.drop(columns=["AllowSunModel", "MomentumTooHigh", "SunPointAngleError"])

plt.figure(figsize=(15, 8))
sns.heatmap(windows_dist_df_edit.T, cmap='Reds')
plt.title('Sigma Heatmap 94 min windows, VPM (July)')
plt.xticks(rotation=45)
plt.locator_params(axis='x', nbins=15)
path = os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\VPM\SPC", "AttDet_09_111_.png") 
plt.savefig(path)
plt.close()
    


# Define the XVID Data path
path = r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\VPM"

file = read_file_category_VPM("AttDet")
variables = file.columns.drop(["ReceivedTime"])

windows_anomaly_df = pd.DataFrame()
windows_dist_df = pd.DataFrame()

for variable in variables:

    df_raw=file[variable]
    resample_interval = "1min"
    resample = df_raw[~df_raw.index.duplicated(keep='first')].resample(resample_interval).mean().interpolate(method="linear")
    resample = filter_by_date(df=resample, fromDate="2023-08-01", toDate="2023-08-31")
    data = resample
    window_size = pd.Timedelta(minutes=94)  
    overlap_size = pd.Timedelta(minutes=23)  
    windows = create_overlapping_windows(data, window_size, overlap_size)[:-1]
    order = (1,1,1)

    try:
        window_coeff = calculate_window_parameter_Arima(windows, order)
        #window_coeff, windows_SE = calculate_window_parameter_SE_Arima(windows, order=order)
        #window_coeff = calculate_window_parameter_autoArima_bic(windows)
    except np.linalg.LinAlgError:
        print(f"Skipping variable {variable} due to LU decomposition error.")
        continue
    
    sigma2 = window_coeff["sigma2"]
    std = np.std(sigma2)
    threshold = std*3
    mu = np.mean(sigma2)
    sigma_anomaly = sigma2[(sigma2 > mu + threshold) | (sigma2 < mu - threshold)]
    sigma_dist = sigma2/std
    window_anomaly = sigma_anomaly.index
    if windows_anomaly_df.empty:
        windows_anomaly_df = pd.DataFrame(index=window_coeff.index)
    if windows_dist_df.empty:
        windows_dist_df = pd.DataFrame(index=window_coeff.index)
    windows_anomaly_df[variable] = 0
    windows_anomaly_df.loc[window_anomaly, variable] = 1
    windows_dist_df.loc[window_coeff.index, variable] = sigma_dist


