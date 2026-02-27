from EDA.my_functions import *

file = read_file_category_XVI("Analogs")

variables = file.columns.drop(["ReceivedTime"])
#variables = [f for f in variables if "Temp" in f]
#variables = ["ConvertedBuckConvTempMon"]

windows_anomaly_df = pd.DataFrame()
windows_dist_df = pd.DataFrame()
variable = variables[1]
for variable in variables:

    df_raw=file[variable]
    resample_interval = "1min"
    resample = df_raw[~df_raw.index.duplicated(keep='first')].resample(resample_interval).mean().interpolate(method="linear")
    resample = filter_by_date(df=resample, fromDate="2023-08-01", toDate="2023-08-31")
    data = resample
    window_size = pd.Timedelta(minutes=94)  
    overlap_size = pd.Timedelta(minutes=23)  
    windows = create_overlapping_windows(data, window_size, overlap_size)[:-1]
    order = (6,1,6)

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

windows_anomaly_df.to_csv(os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC", "Analogs_anomaly_616_08.csv"))
windows_dist_df.to_csv(os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC", "Analogs_heatmap_616_08.csv"))
windows_dist_df = pd.read_csv(os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC", "Analogs_anomaly_Auto.csv"), index_col=0, parse_dates=True)





windows_dist_df.columns
windows_dist_df_edit = windows_dist_df.copy()
#windows_dist_df_edit = np.log(windows_dist_df_edit+1.01)
windows_dist_df_edit.index = windows_dist_df.index.date

windows_dist_df_edit = windows_dist_df_edit.drop(columns=["ConvertedUserAnalog5"])
#windows_dist_df_edit = windows_dist_df_edit.drop(columns=["AllowSunModel", "MomentumTooHigh", "SunPointAngleError"])

plt.figure(figsize=(15, 8))
sns.heatmap(windows_dist_df_edit.T, cmap='Reds')
plt.title('Sigma Heatmap 94 min windows, Auto_ARIMA')
plt.xticks(rotation=45)
plt.locator_params(axis='x', nbins=15)
path = os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC", "Analogs_515.png") 
plt.savefig(path)
plt.close()

# Compute the pairwise distance between variables
# Assuming windows_dist_df_edit is your DataFrame
df = windows_dist_df_edit

# Perform hierarchical clustering
linkage = sch.linkage(df.T, method='average')

# Create a dendrogram to reorder the columns
dendro = sch.dendrogram(linkage, labels=df.columns, no_plot=False)
plt.show()

ordered_columns = [df.columns[i] for i in dendro['leaves']]

df_reordered = df[ordered_columns]

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df_reordered.T, cmap='Reds', cbar_kws={'label': 'Value'})
plt.title('Heatmap of Clustered Variables')
plt.xlabel('Samples')

plt.ylabel('Variables')
plt.show()

sns.heatmap(df_reordered.T, cmap='viridis')
plt.xlabel('Time')
plt.ylabel('Variables')
plt.title('Heatmap with Clustered Variables')
plt.show()


# Melt the DataFrame to long format for easier plotting
# Reset the index and melt the DataFrame
anomaly_data = windows_anomaly_df.reset_index().melt(id_vars=windows_anomaly_df.index.name, var_name='Variable', value_name='Anomaly')

# Filter out non-anomalous points
anomaly_data = anomaly_data[anomaly_data['Anomaly'] == 1]

# Plot the anomalies
plt.figure(figsize=(15, 8))
plt.scatter(anomaly_data[windows_anomaly_df.index.name], anomaly_data['Variable'], color='red', marker='o')
plt.title('Anomalies Over Time by Variable 94 min windows (August), ARIMA(202)')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.xticks(rotation=45)
plt.grid(True)

# Show the plot
path = os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC", "Anomaly_94windows_212_11.png") 
plt.savefig(path)
plt.close()



# Clean the data
file = read_file_category("Analogs")
df_raw=file["Converted_3v3CurrMon"]
df_raw=file["Converted_3v3CurrMon"]
resample_interval = "1min"
resample = df_raw[~df_raw.index.duplicated(keep='first')].resample(resample_interval).mean().interpolate(method="linear")
resample = filter_by_date(df=resample, fromDate="2024-01-01", toDate="2024-01-31")
# Check Stationary
check_stationary(data=resample, table=True, alpha=.05)

# If its not Stationary do Transformation
# 01 Temprature Data
dominent_period = seasonal_period_fourior(df=resample) # It need work
# 19 for 5 min resampling 
dominent_period = 94
seasonalRemoved = seasonal_differencing(series=resample.values, period=dominent_period)
df_resample = resample.to_frame()
df_resample["seasonalRemoved"] = seasonalRemoved
df_resample = df_resample.dropna()

# 02 Not Temprature
df_resample = resample.to_frame()
df_resample["diff1"] = df_resample.diff()
df_resample = df_resample.dropna()

# Check Stationary
check_stationary(data=df_resample.iloc[:, 1], table=True, alpha=.05)

#Chech data visually
Show_TimeSeries_plot(df_resample)
data = df_resample.iloc[:, 1]


# Make windows
window_size = pd.Timedelta(minutes=94)  
overlap_size = pd.Timedelta(minutes=23)  
windows = create_overlapping_windows(data, window_size, overlap_size)[:-1]

# Find the min AIC order
#auto_model = pm.auto_arima(data, seasonal=True, trace=True, error_action='ignore', suppress_warnings=True, stepwise=False)
#order = auto_model.order
order = (3,0,1)

# Fit the same order to all windows
window_coeff = calculate_window_parameter_Arima(windows, order=order)

# plot the result
#plot_Control_Chart(window_coeff, C=3, file_name = "SPC\\test", T = "", reset=None, xlab="Time", ylab="")


#window_coeff, windows_SE = calculate_window_parameter_SE_Arima(windows, order=order)
#window_coeff, windows_SE = calculate_window_parameter_SE_autoArima(windows)


# plot the result
plot_Control_Chart(df=window_coeff, C=3, file_name = "SPC\\test", T = "", reset=pd.DatetimeIndex([
    '2023-08-11 01:30:00',
    '2023-08-26 15:38:41'],
    dtype='datetime64[ns, UTC]', freq=None), xlab="Time", ylab="")

plot_Control_Chart_SE(window_coeff, C=3, file_name = "SPC\\Converted_3v3CurrMon09_94min", T = "", reset=pd.DatetimeIndex([
    '2023-09-13 17:40:00',
    '2023-09-16 09:26:00'],
    dtype='datetime64[ns, UTC]', freq=None), xlab="Time", ylab="")



fault_202308 = pd.DatetimeIndex(["2023-08-04 19:23:10", 
                                             "2023-08-11 01:54:00", 
                                             '2023-08-24 20:01:10', 
                                             "2023-08-26 15:38:40",
                                             "2023-08-30 15:33:30", 
                                             "2023-08-31 18:37:10"])

fault_202309 = pd.DatetimeIndex(["2023-09-06 07:28:30", 
                                             "2023-09-07 21:55:40", 
                                             '2023-09-08 04:51:40', 
                                             "2023-09-11 14:42:50",
                                             "2023-09-13 17:41:00", 
                                             "2023-09-15 21:41:10",
                                             "2023-09-16 09:26:20",
                                             "2023-09-16 09:49:10",
                                             "2023-09-24 04:22:50",
                                             "2023-09-25 16:39:30",
                                             "2023-09-26 16:55:40",
                                             "2023-09-27 19:01:20"])

fault_202309 = pd.DatetimeIndex(["2023-10-07 16:20:00", 
                                             "2023-10-08 08:07:30", 
                                             '2023-10-15 04:20:00', 
                                             "2023-10-20 20:03:40",
                                             "2023-10-28 18:37:40", 
                                             "2023-10-30 13:06:20"])


## Generating the Artificial data
# Data Simulation
# Given coefficients
c = 0
phi1 = .5
phi2 = .2
phi3 = .3
theta1 = .9
sigma =2
# Number of time series to simulate
series_length = 45000
simulated_data = np.zeros(series_length)

noise = np.random.normal(0, 1, series_length)
# Generate simulated data
simulated_data[0] = 0
simulated_data[1] = 0
simulated_data[2] = 0
for t in range(3, series_length):
    simulated_data[t] =  c + phi1*simulated_data[t-1] + phi2*simulated_data[t-2] + phi3*simulated_data[t-3] + sigma*noise[t] + theta1*noise[t-1]
 
sim_diff = np.diff(simulated_data)

armiaModel = pm.auto_arima(sim_diff)
order = armiaModel.order
order = (3,0,1)


model = sm.tsa.ARIMA(sim_diff, order=order)
fit = model.fit()
fit.summary()
fit.params
fit 

time_index = pd.date_range(start='2023-01-01', periods=len(sim_diff), freq='min')

# Convert sim_diff to a pandas Series with the time index
sim_diff = pd.Series(sim_diff, index=time_index)
check_stationary(data=sim_diff, table=True, alpha=.05)
# Make windows
window_size = pd.Timedelta(minutes=10*94)  
overlap_size = pd.Timedelta(minutes=10*46)  
windows = create_overlapping_windows(sim_diff, window_size, overlap_size)[:-1]
#windows_stationary = calculate_window_stationary(windows=windows)
#window_coeff, windows_SE = calculate_window_parameter_SE_Arima(windows, order=order)
window_coeff, windows_SE = calculate_window_parameter_SE_autoArima(windows)
# plot the result
plot_Control_Chart_SE(df=window_coeff, SE=windows_SE, file_name = "SPC\\sim", T = "", reset=None, xlab="Time", ylab="")



# Plot the simulated data
plt.figure(figsize=(10, 6))
plt.plot(sim_diff, color='b', linestyle='-', linewidth=1.5)
plt.title('Simulated Data based on ARIMA(3,0,1) Model')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.show()











# Check stationary of windows
windows_stationary = calculate_window_stationary(windows=windows)
station_table = windows_stationary.copy()
station_table["Window"] = station_table["Window"].apply(lambda x: x.time().strftime('%H:%M:%S'))
for column in station_table.columns[1:]:
    # Check if column is numeric
    if pd.api.types.is_numeric_dtype(station_table[column]):
        station_table[column] = station_table[column].apply(lambda x: f'{x:.1e}' if pd.notnull(x) else 'NaN')
    else:
        # For non-numeric columns, just ensure 'NaN' is properly represented
        station_table[column] = station_table[column].apply(lambda x: 'NaN' if pd.isnull(x) else x)
print(tabulate(station_table, headers='keys', tablefmt='latex', showindex=False))

# Fit ARIMA 
window_coeff = calculate_window_parameter_Arima(windows, order=order)  
coef_table = window_coeff.copy()
coef_table.index = coef_table.index.to_series().apply(lambda x: x.time().strftime('%H:%M:%S'))
df_formatted = coef_table.copy()
for column in coef_table.columns[1:]:
    df_formatted[column] = coef_table[column].apply(lambda x: f'{x:.1e}' if pd.notnull(x) else 'NaN')

# Print LaTeX table
print(tabulate(df_formatted, headers='keys', tablefmt='latex', showindex=True))

plot_Control_Chart(window_coeff, C=3, file_name = "Arima_ConvertedPayloadTemp2_seasonalRemoved_5min120min_maxpq_08b", T = "", reset=pd.DatetimeIndex(["2023-08-04 19:23:10", 
                                             "2023-08-11 01:54:00", 
                                             '2023-08-24 20:01:10', 
                                             "2023-08-26 15:38:40",
                                             "2023-08-30 15:33:30", 
                                             "2023-08-31 18:37:10"
                                             ]), xlab="Time", ylab="")

plot_Control_Chart(window_coeff, C=3, file_name = "Arima_ConvertedPayloadTemp2_seasonalRemoved_5min120min_maxpq_09", T = "", reset=pd.DatetimeIndex(["2023-09-06 07:28:30", 
                                             "2023-09-07 21:55:40", 
                                             '2023-09-08 04:51:40', 
                                             "2023-09-11 14:42:50",
                                             "2023-09-13 17:41:00", 
                                             "2023-09-15 21:41:10",
                                             "2023-09-16 09:26:20",
                                             "2023-09-16 09:49:10",
                                             "2023-09-24 04:22:50",
                                             "2023-09-25 16:39:30",
                                             "2023-09-26 16:55:40",
                                             "2023-09-27 19:01:20"
                                             ]), xlab="Time", ylab="")


plot_Control_Chart(window_coeff, C=3, file_name = "Arima_Converted_3v3CurrMon_seasonalRemoved_5min120min_fault", T = "", reset=fault_times, xlab="Time", ylab="")

df = df_raw.diff()
window_size = pd.Timedelta(days=2)  
overlap_size = pd.Timedelta(days=.5)  
windows = create_overlapping_windows(df, window_size, overlap_size)
window_means = calculate_window_means(windows=windows)
Save_TimeSeries_plot(df=df_resample.to_frame())
plot_Control_Chart(df, reset=reset_times_dict["Unknown Reset"], C=3, file_name="Test")
type(window_means.to_frame())

#############################################
# check the distribution of sigma 2 

sigma_df = pd.DataFrame()
#variable = variables[1]
for variable in variables:

    df_raw=file[variable]
    resample_interval = "1min"
    resample = df_raw[~df_raw.index.duplicated(keep='first')].resample(resample_interval).mean().interpolate(method="linear")
    resample = filter_by_date(df=resample, fromDate="2023-08-01", toDate="2023-08-31")
    # Make windows
    data = resample
    window_size = pd.Timedelta(minutes=94)  
    overlap_size = pd.Timedelta(minutes=23)  
    windows = create_overlapping_windows(data, window_size, overlap_size)[:-1]
    order = (2,1,2)

    try:
        window_coeff = calculate_window_parameter_Arima(windows, order)
        #window_coeff, windows_SE = calculate_window_parameter_SE_Arima(windows, order=order)
        #window_coeff = calculate_window_parameter_autoArima_bic(windows)
    except np.linalg.LinAlgError:
        print(f"Skipping variable {variable} due to LU decomposition error.")
        continue
    
    #plot_Control_Chart(df=window_coeff, C=3, file_name = f"SPC\\test", T = "", reset=None, xlab="Time", ylab="")
    sigma2 = window_coeff["sigma2"]
    if sigma_df.empty:
        sigma_df = pd.DataFrame(index=window_coeff.index)
   
    sigma_df.loc[window_coeff.index, variable] = sigma2
    #plot_Control_Chart(df=window_coeff["sigma2"].to_frame(), C=3, file_name = f"SPC\\Aug\\{variable}_212_plo3", T = "", xlab="Time", ylab="")


plt.figure()
plt.hist(dfaic.iloc[:, 0], bins=50)
plt.show()