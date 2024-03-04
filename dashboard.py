import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from babel.numbers import format_currency
sns.set(style='dark')

#Import Dataset
day = pd.read_csv('day.csv')    
hour = pd.read_csv('hour.csv')

#Helper Functions
def create_daily_rentals(df):
    daily_rentals_df = df.resample(rule='D', on='dteday').agg({
    "cnt": "sum"
    })
    daily_rentals_df.rename(columns={
        "cnt": "total_orders"
    }, inplace=True)

    return daily_rentals_df

def create_rental_hour(df):
    rental_hour_df = df.groupby('hr').agg({
        "cnt": "sum"
    }).reset_index()
    rental_hour_df.rename(columns={
        "cnt": "hour_total_orders"
    }, inplace=True)

    return rental_hour_df

def create_dominant_day(df):
    dominant_day_df = df.groupby('workingday')['cnt'].mean()
    return dominant_day_df

def create_dominant_weather(df):
    daily_agg = df.groupby(['mnth', 'season']).agg({'cnt': 'sum'}).reset_index()
    season_counts = daily_agg.groupby('season')['cnt'].sum()
    return season_counts


#Membuat Filter
day.sort_values(by="dteday", inplace=True)
day.reset_index(inplace=True)
day['dteday'] = pd.to_datetime(day['dteday'])

min_date = day["dteday"].min()
max_date = day["dteday"].max()

with st.sidebar:
    start_date, end_date = st.date_input(
        label='Rentang Waktu',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )
day_filtered = day[(day["dteday"] >= str(start_date)) & (day["dteday"] <= str(end_date))]
hour_filtered = hour[(hour["dteday"] >= str(start_date)) & (hour["dteday"] <= str(end_date))]

#Using Helper Funtions
daily_rentals_df = create_daily_rentals(day_filtered)
dominant_day_df = create_dominant_day(day_filtered)
rental_hour_df = create_rental_hour(hour_filtered)
dominant_weather_df = create_dominant_weather(day_filtered)

#View
st.header('Bike Rentals Dataset Dashboard :bike:')

col1, col2 = st.columns(2)
with col1:
    total_rentals = day_filtered.cnt.sum()
    total_casual = day_filtered.casual.sum()
    
    st.metric("Total Rentals of All Time", value=total_rentals)
    st.metric("Total Non-Members", value=total_casual)

with col2:
    total_regist = day_filtered.registered.sum()
    st.metric("Total Members Registered", value=total_regist)


#Plot 1
st.subheader('Total Rentals Charts')
window_size = 50
moving_avg = daily_rentals_df["total_orders"].rolling(window=window_size).mean()

# Plotting
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    day_filtered["dteday"],
    daily_rentals_df["total_orders"],
    color="cyan",
    linewidth=1,
    solid_capstyle='round'
)

# Menambahkan moving average ke dalam plot
ax.plot(day_filtered["dteday"], moving_avg, color='yellow', linestyle='-', linewidth=2, label=f'Moving Average ({window_size} Days)')

ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
ax.grid(axis='both', color='lightgrey', linestyle='--', linewidth=0.5)

ax.set_title('Total Bike Rental Visualization', fontsize=25, color='white')
ax.set_xlabel('Date', fontsize=20, color='white')
ax.set_ylabel('Total Bike Rentals', fontsize=20, color='white')

plt.tight_layout()
st.pyplot(fig)
st.caption("Peminjaman sepeda mengalami kondisi yang cukup fluktuatif, dapat dilihat pada grafik yang memiliki kecenderungan naik turun dan terjadi kenaikan penjualan pada bulan Oktober tahun 2012. Kemudian dilanjutkan dengan penurunan volume peminjaman setelahnya yang dapat dilihat pada grafik")

#Plot 2
st.subheader('Hourly Rentals')
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    rental_hour_df['hour_total_orders'],
    color="red",
    linewidth=3
)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)

ax.grid(axis='both', color='lightgrey', linestyle='--', linewidth=0.5)
ax.set_title('Total Rentals per Hour', fontsize=25, color='white')
ax.set_xlabel('Hour', fontsize=20, color='white')
ax.set_xticks(np.arange(0, 24, 1))

ax.set_ylabel('Total Bike Rentals', fontsize=20, color='white')

max_orders = rental_hour_df['hour_total_orders'].max()
ax.axhline(y=max_orders, color='white', linestyle='--', linewidth=1)
ax.text(0.5, max_orders + 50, f'Max Orders: {max_orders}', color='yellow', fontsize=12, ha='center')

x_values = np.arange(len(rental_hour_df))
z = np.polyfit(x_values, rental_hour_df['hour_total_orders'], 1)
p = np.poly1d(z)

ax.plot(x_values, p(x_values), color='yellow', linestyle='-', linewidth=1, label='Trend Line')
ax.legend()

plt.tight_layout()
st.pyplot(fig)
st.caption("Total peminjaman tertinggi berada pada angka 336860 ribu peminjaman pada jam 17.00 atau pada jam 5 sore, selain itu peminjaman tertinggi kedua berada pada jam 8 pagi yang berada pada angka sekitar 250 ribu peminjaman. Pada sisi lain, garis tren pada grafik juga menunjukkan adanya kenaikan volume peminjaman selama 2 tahun yang di mulai pada tahun 2011 hingga 2012")

# Plot 3
plt.figure(figsize=(8, 6.373))
dominant_day_df.plot(kind='bar', color=['lightgrey', 'red'])
plt.title('Average Bike Rentals: Working Day vs Non-Working Day')
plt.xlabel('Working Day (1: Yes, 0: No)')
plt.ylabel('Average Bike Rentals')
plt.xticks(ticks=[0, 1], labels=['Non-Working Day', 'Working Day'], rotation=0)
plt.grid(axis='y')
plt.tight_layout()

season_labels = ['Spring', 'Summer', 'Fall', 'Winter']
dominant_weather_df = create_dominant_weather(day_filtered)
bar_colors = ['lightgrey'] * len(season_labels)
max_season = dominant_weather_df.idxmax()
bar_colors[max_season - 1] = 'red'

col1, col2 = st.columns(2)
with col1:
    st.pyplot(plt)
    st.caption("Peminjaman terbanyak terjadi pada hari kerja dengan jumlah sebanyak 4584 peminjaman dibandingkan dengan peminjaman di luar hari kerja")

with col2:
    plt.figure(figsize=(8, 6))
    plt.bar(season_labels, dominant_weather_df, color=bar_colors)
    plt.title('Total Rental Counts by Season')
    plt.xlabel('Season')
    plt.ylabel('Total Counts')
    plt.show()
    st.pyplot(plt)
    st.caption("Musim yang paling dominan pada peminjaman sepeda terjadi pada saat musin gugur, dengan jumlah sebanyak 1,061,129 peminjaman. Kemudian dilanjutkan pada peminjama di musim panas sebagai tertinggi kedua, musim dingin sebagai tertinggi ketiga, dan yang terakhir ada pada musim semi.")