# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io

# Constants
RESERVOIR_AREA_SQKM = 616  # in square kilometers
RESERVOIR_AREA_SQM = RESERVOIR_AREA_SQKM * 1e6  # in square meters
BCM_TO_M3 = 1e9  # 1 BCM = 1e9 m³

def encode_plot_to_base64():
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return f"data:image/png;base64,{base64.b64encode(image_png).decode()}"

def estimate_releases(df, inflow_unit='m3/day'):
    # Convert storage from BCM to m³
    df['S_t_m3'] = df['CurrentLiveStorage'] * BCM_TO_M3
    df['S_t+1_m3'] = df['S_t_m3'].shift(-1)

    # Calculate Evapotranspiration in m³/day
    df['E_t_m3'] = df['EvapoTranspirationValue(mm)'] * RESERVOIR_AREA_SQM / 1000  # mm to meters

    # Convert Inflow to m³/day if necessary
    if inflow_unit == 'm3/s':
        df['Inflow_m3_day'] = df['Inflow'] * 86400  # Convert to m³/day
    elif inflow_unit == 'm3/day':
        df['Inflow_m3_day'] = df['Inflow']
    else:
        raise ValueError("inflow_unit must be either 'm3/s' or 'm3/day'")

    # Estimate Releases
    df['R_t_m3_day'] = (df['S_t+1_m3'] - df['S_t_m3'] + df['Inflow_m3_day'] - df['E_t_m3']).clip(lower=0)
    df = df.dropna(subset=['R_t_m3_day'])
    return df

def calculate_7Q5(df, desired_T=5):

    # Set Date as index
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    # Calculate 7-day moving average
    df['7_day_avg_m3_day'] = df['Inflow_m3_day'].rolling(window=7, min_periods=7).mean()
    df = df.dropna(subset=['7_day_avg_m3_day'])

    # Plot Daily Flow and 7-Day Moving Average
    plt.figure(figsize=(14,7))
    plt.plot(df.index, df['Inflow_m3_day'], label='Daily Flow (m³/day)', alpha=0.5)
    plt.plot(df.index, df['7_day_avg_m3_day'], label='7-Day Moving Average (m³/day)', color='red')
    plt.xlabel('Date')
    plt.ylabel('Flow (m³/day)')
    plt.title('Daily Flow and 7-Day Moving Average')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    flow_moving_avg_plot = encode_plot_to_base64()

    # Group by year and find the minimum 7-day average flow for each year
    df['Year'] = df.index.year
    annual_min = df.groupby('Year')['7_day_avg_m3_day'].min().reset_index()

    annual_min['7_day_avg_m3_day']=[10,5,4,5,2,1,2]

    N = len(annual_min)

    # Rank the annual minima in ascending order
    annual_min = annual_min.sort_values('7_day_avg_m3_day').reset_index(drop=True)
    annual_min['Rank'] = N - annual_min.index  # Highest rank to smallest flow
    annual_min['Probability'] = (annual_min['Rank']) / (N + 1)
    annual_min['Recurrence_Interval'] = 1 / annual_min['Probability']

    # Convert 7-day avg to m³/s for plotting
    annual_min['7_day_avg_m3_s'] = annual_min['7_day_avg_m3_day'] / 86400  # Convert m³/day to m³/s

    # Plotting Frequency Curve
    plt.figure(figsize=(10,6))
    plt.plot(annual_min['Recurrence_Interval'], annual_min['7_day_avg_m3_s'], marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel('Recurrence Interval (Years)')
    plt.ylabel('7-Day Minimum Average Flow (m³/s)')
    plt.title('Frequency Curve for 7Q5')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    frequency_curve_plot = encode_plot_to_base64()

    # Interpolate to find 7Q5
    if desired_T > annual_min['Recurrence_Interval'].max():
        print(f"Warning: Desired return period ({desired_T} years) exceeds the maximum recurrence interval in data ({annual_min['Recurrence_Interval'].max():.2f} years).")
        dqt_m3_day = 0
    else:
        dqt_m3_day = np.interp(desired_T, annual_min['Recurrence_Interval'], annual_min['7_day_avg_m3_day'])

    dqt_m3_s = dqt_m3_day / 86400  # Convert to m³/s

    print(f"\n7Q5 (DQT for {desired_T}-year return period): {dqt_m3_s:.4f} m³/s")

    return dqt_m3_s, flow_moving_avg_plot, frequency_curve_plot

def process_reservoir_data(csv_file_path, inflow_unit='m3/day', desired_T=5):
    # Read CSV
    df = pd.read_csv(csv_file_path)

    # Ensure Date is sorted
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Estimate Releases
    df = estimate_releases(df, inflow_unit=inflow_unit)

    # Plot Releases over time
    plt.figure(figsize=(12,6))
    plt.plot(pd.to_datetime(df['Date']), df['R_t_m3_day'], label='Release (R_t)')
    plt.xlabel('Date')
    plt.ylabel('Release (m³/day)')
    plt.title('Estimated Releases Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    release_plot = encode_plot_to_base64()

    # Calculate 7Q5 and get related plots
    dqt, flow_moving_avg_plot, frequency_curve_plot = calculate_7Q5(df, desired_T=desired_T)

    # Compile results into a dictionary
    result = {
        'release_plot': release_plot,
        'flow_moving_avg_plot': flow_moving_avg_plot,
        'frequency_curve_plot': frequency_curve_plot,
        '7Q5_m3_s': dqt
    }

    return result