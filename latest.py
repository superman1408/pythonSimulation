import streamlit as st
import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.colors import to_rgba
import numpy as np
import time

# ------------------------------
# Simulation Function
# ------------------------------
def run_simulation(
    num_ships,
    fab_machines,
    assy_bays,
    erect_docks,
    outfit_berths,
    fab_time,
    assy_time,
    erect_time,
    outfit_time,
    sim_time,
    random_seed=42,
    progress_callback=None
):
    random.seed(random_seed)
    records = []
    completed_ships = {"count": 0}
    progress_log = []

    env = simpy.Environment()

    # Resources
    fab = simpy.Resource(env, capacity=fab_machines)
    assy = simpy.Resource(env, capacity=assy_bays)
    erect = simpy.Resource(env, capacity=erect_docks)
    outfit = simpy.Resource(env, capacity=outfit_berths)

    # Record events
    def record_event(ship, stage, start, end):
        records.append({
            "Ship": ship,
            "Stage": stage,
            "Start": start,
            "End": end,
            "Duration (weeks)": end - start
        })

    # Stage process
    def process_stage(ship, stage_name, resource, duration_range):
        with resource.request() as req:
            yield req
            start = env.now
            t = random.randint(*duration_range)
            yield env.timeout(t)
            end = env.now
            record_event(ship, stage_name, start, end)

    # Full ship process
    def ship_process(ship):
        yield env.process(process_stage(ship, "Fabrication", fab, fab_time))
        yield env.process(process_stage(ship, "Assembly", assy, assy_time))
        yield env.process(process_stage(ship, "Erection", erect, erect_time))
        yield env.process(process_stage(ship, "Outfitting", outfit, outfit_time))
        completed_ships["count"] += 1

    # Monitor progress
    def progress_monitor(env, interval=5):
        while True:
            percent = (completed_ships["count"] / num_ships) * 100
            progress_log.append((env.now, percent))
            if progress_callback:
                progress_callback(min(100, percent))
            yield env.timeout(interval)
            if env.now >= sim_time:
                break

    # Launch ships
    def shipyard(env):
        env.process(progress_monitor(env))
        for i in range(num_ships):
            env.process(ship_process(f"Ship-{i+1}"))
            yield env.timeout(random.randint(3, 8))  # stagger arrivals

    env.process(shipyard(env))
    env.run(until=sim_time)

    df_records = pd.DataFrame(records)
    df_progress = pd.DataFrame(progress_log, columns=["Week", "Completion (%)"])

    completed = completed_ships["count"]
    incomplete = num_ships - completed
    completion_percent = (completed / num_ships) * 100

    return df_records, df_progress, completed, incomplete, completion_percent

# ------------------------------
# Minimum Simulation Time Estimation
# ------------------------------
def estimate_min_time(num_ships, fab_machines, assy_bays, erect_docks, outfit_berths,
                      fab_time, assy_time, erect_time, outfit_time, trials=50):
    times = []
    for _ in range(trials):
        total_time = 0
        for _ in range(num_ships):
            fab = random.randint(*fab_time) / fab_machines
            assy = random.randint(*assy_time) / assy_bays
            erect = random.randint(*erect_time) / erect_docks
            outfit = random.randint(*outfit_time) / outfit_berths
            total_time += fab + assy + erect + outfit
        times.append(total_time)
    # Return average of multiple trials as min estimated time
    return int(np.ceil(np.mean(times)))

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="ASHKAM Shipyard Simulator", layout="wide")
st.title("🚢 Shipyard Simulation Dashboard (Weeks)")
st.caption("All durations and simulation time are measured in weeks.")

mode = st.radio("Select Mode", ["Default", "User Defined"], horizontal=True)

# Default parameters
if mode == "Default":
    num_ships = 5
    fab_machines = 3
    assy_bays = 2
    erect_docks = 1
    outfit_berths = 1
    fab_time = (8, 12)
    assy_time = (10, 15)
    erect_time = (12, 20)
    outfit_time = (15, 25)
    sim_time = 500
else:
    st.subheader("⚙️ User Inputs (Weeks)")
    num_ships = st.number_input("Number of Ships", min_value=1, value=5)
    fab_machines = st.number_input("Fabrication Machines", min_value=1, value=3)
    assy_bays = st.number_input("Assembly Bays", min_value=1, value=2)
    erect_docks = st.number_input("Erection Docks", min_value=1, value=1)
    outfit_berths = st.number_input("Outfitting Berths", min_value=1, value=1)
    sim_time = st.number_input("Simulation Time (weeks)", min_value=50, value=500)
    st.markdown("#### Stage Duration Ranges")
    col1, col2 = st.columns(2)
    with col1:
        fab_min = st.number_input("Fabrication Min", 1, 100, 8)
        assy_min = st.number_input("Assembly Min", 1, 100, 10)
        erect_min = st.number_input("Erection Min", 1, 100, 12)
        outfit_min = st.number_input("Outfitting Min", 1, 100, 15)
    with col2:
        fab_max = st.number_input("Fabrication Max", 1, 100, 12)
        assy_max = st.number_input("Assembly Max", 1, 100, 15)
        erect_max = st.number_input("Erection Max", 1, 100, 20)
        outfit_max = st.number_input("Outfitting Max", 1, 100, 25)
    fab_time = (fab_min, fab_max)
    assy_time = (assy_min, assy_max)
    erect_time = (erect_min, erect_max)
    outfit_time = (outfit_min, outfit_max)

# ------------------------------
# Run Simulation
# ------------------------------
if st.button("▶️ Run Simulation"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(percent):
        progress_bar.progress(int(percent))
        status_text.text(f"Simulation Progress: {percent:.1f}%")

    # Run simulation
    df_records, df_progress, completed, incomplete, completion_percent = run_simulation(
        num_ships, fab_machines, assy_bays, erect_docks, outfit_berths,
        fab_time, assy_time, erect_time, outfit_time, sim_time,
        progress_callback=update_progress
    )

    # Estimate minimum simulation time
    min_time = estimate_min_time(num_ships, fab_machines, assy_bays, erect_docks, outfit_berths,
                                 fab_time, assy_time, erect_time, outfit_time, trials=50)

    status_text.text("Simulation Completed ✅")
    progress_bar.empty()

    st.subheader("⏱️ Estimated Minimum Simulation Time (Weeks)")
    st.metric("Min Time to Complete All Ships", min_time)

    # The rest of the dashboard (overall summary, Gantt chart, S-curve, etc.)
    st.subheader("📊 Overall Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ships", num_ships)
    col2.metric("Completed Ships", completed)
    col3.metric("Incomplete Ships", incomplete)
    col4.metric("Completion (%)", f"{completion_percent:.1f}%")

    # Progress chart
    st.subheader("📈 Shipyard Completion Progress Over Time")
    fig_progress = px.line(df_progress, x="Week", y="Completion (%)", markers=True)
    st.plotly_chart(fig_progress, use_container_width=True)

    # Partial Stage Gantt chart
    st.subheader("🚀 Shipyard Gantt Chart with Partial Stage Completion")
    fig, ax = plt.subplots(figsize=(12, 6))
    stage_colors = {"Fabrication":"skyblue","Assembly":"orange","Erection":"lightgreen","Outfitting":"salmon"}
    df_records["Actual Duration"] = df_records.apply(lambda row: max(0, min(row["End"], sim_time) - row["Start"]), axis=1)
    df_records["Stage Completion (%)"] = (df_records["Actual Duration"] / df_records["Duration (weeks)"]) * 100

    for i, ship in enumerate(df_records["Ship"].unique()):
        ship_data = df_records[df_records["Ship"] == ship].sort_values(by="Start")
        for _, row in ship_data.iterrows():
            if row["Actual Duration"] <= 0:
                continue
            rgba_color = to_rgba(stage_colors[row["Stage"]], alpha=0.7)
            ax.barh(ship, row["Actual Duration"], left=row["Start"], color=rgba_color)
            ax.text(row["Start"] + row["Actual Duration"]/2, i,
                    f"{row['Stage']} ({row['Stage Completion (%)']:.0f}%)",
                    ha='center', va='center', fontsize=8, color='black')

    ax.set_xlabel("Simulation Time (weeks)")
    ax.set_ylabel("Ships")
    ax.set_title("Gantt Chart with Partial Stage Completion")
    plt.tight_layout()
    st.pyplot(fig)

    # Shipwise stage completion report
    st.subheader("📄 Shipwise Stage Completion Report")
    ship_completion = df_records.groupby("Ship")["Stage Completion (%)"].mean().reset_index()
    ship_completion.rename(columns={"Stage Completion (%)":"Ship Completion (%)"}, inplace=True)
    summary_report = df_records.merge(ship_completion, on="Ship")
    summary_report = summary_report[["Ship","Stage","Duration (weeks)","Actual Duration",
                                     "Stage Completion (%)","Ship Completion (%)"]].sort_values(["Ship","Stage"])
    st.dataframe(summary_report.style.format({
        "Duration (weeks)":"{:.1f}",
        "Actual Duration":"{:.1f}",
        "Stage Completion (%)":"{:.1f}%",
        "Ship Completion (%)":"{:.1f}%"
    }))

    # S-Curve with sigmoid
    st.subheader("📊 S-Curve: Actual vs Ideal Ship Completion (Sigmoid)")
    timeline = list(range(0, sim_time+1))
    cumulative_completion = []

    for week in timeline:
        completed_percentage = 0
        for _, row in df_records.iterrows():
            stage_start = row["Start"]
            stage_end = row["End"]
            stage_duration = row["Duration (weeks)"]
            if week < stage_start:
                frac = 0
            elif week >= stage_end:
                frac = 1
            else:
                frac = (week - stage_start) / stage_duration
            completed_percentage += frac
        total_stages = df_records.shape[0]
        cumulative_completion.append((week, (completed_percentage/total_stages)*100))

    s_curve_df = pd.DataFrame(cumulative_completion, columns=["Week","Completion (%)"])

    # Ideal sigmoid
    t = np.array(timeline)
    t0 = sim_time / 2
    k = 0.12
    ideal_completion = 100 / (1 + np.exp(-k*(t - t0)))
    ideal_df = pd.DataFrame({"Week": t, "Completion (%)": ideal_completion})

    fig3, ax3 = plt.subplots(figsize=(10,5))
    ax3.plot(s_curve_df["Week"], s_curve_df["Completion (%)"], label="Actual (Partial Progress)", color="blue")
    ax3.plot(ideal_df["Week"], ideal_df["Completion (%)"], label="Ideal S-Curve", color="orange", linestyle="--")
    ax3.set_xlabel("Simulation Week")
    ax3.set_ylabel("Cumulative Completion (%)")
    ax3.set_title("S-Curve: Shipyard Progress vs Ideal")
    ax3.legend()
    plt.tight_layout()
    st.pyplot(fig3)
