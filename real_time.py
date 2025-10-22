import streamlit as st
import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import time

# ------------------------------
# Simulation Function (Real-Time)
# ------------------------------
def run_simulation_real_time(
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
    progress_callback=None,
    stage_callback=None,
    random_seed=42
):
    random.seed(random_seed)
    records = []
    completed_ships = {"count": 0}

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

    # Stage process with smooth progress
    def process_stage(ship, stage_name, resource, duration_range):
        with resource.request() as req:
            yield req
            start = env.now
            t = random.randint(*duration_range)
            for w in range(t):
                if stage_callback:
                    stage_callback(ship, stage_name, w+1, t)
                yield env.timeout(1)  # simulate 1 week at a time
            end = env.now
            record_event(ship, stage_name, start, end)

    # Full ship process
    def ship_process(ship):
        yield env.process(process_stage(ship, "Fabrication", fab, fab_time))
        yield env.process(process_stage(ship, "Assembly", assy, assy_time))
        yield env.process(process_stage(ship, "Erection", erect, erect_time))
        yield env.process(process_stage(ship, "Outfitting", outfit, outfit_time))
        completed_ships["count"] += 1

    # Progress monitor (update every week)
    def progress_monitor(env):
        while True:
            percent = (completed_ships["count"] / num_ships) * 100
            if progress_callback:
                progress_callback(percent)
            yield env.timeout(1)
            if env.now >= sim_time:
                break

    # Launch ships
    def shipyard(env):
        env.process(progress_monitor(env))
        for i in range(num_ships):
            env.process(ship_process(f"Ship-{i+1}"))
            yield env.timeout(random.randint(1, 3))  # stagger arrivals

    env.process(shipyard(env))
    env.run(until=sim_time)

    df_records = pd.DataFrame(records)
    return df_records, completed_ships["count"]

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="ASHKAM Shipyard Simulator (Real-Time)", layout="wide")
st.title("🚢 Real-Time Shipyard Simulator (Weeks)")

# Inputs
num_ships = st.number_input("Number of Ships", 1, 20, 5)
fab_machines = st.number_input("Fabrication Machines", 1, 5, 3)
assy_bays = st.number_input("Assembly Bays", 1, 5, 2)
erect_docks = st.number_input("Erection Docks", 1, 5, 1)
outfit_berths = st.number_input("Outfitting Berths", 1, 5, 1)
sim_time = st.number_input("Simulation Time (weeks)", 50, 1000, 200)

st.markdown("#### Stage Duration Ranges (weeks)")
col1, col2 = st.columns(2)
with col1:
    fab_min = st.number_input("Fabrication Min", 1, 50, 8)
    assy_min = st.number_input("Assembly Min", 1, 50, 10)
    erect_min = st.number_input("Erection Min", 1, 50, 12)
    outfit_min = st.number_input("Outfitting Min", 1, 50, 15)
with col2:
    fab_max = st.number_input("Fabrication Max", 1, 50, 12)
    assy_max = st.number_input("Assembly Max", 1, 50, 15)
    erect_max = st.number_input("Erection Max", 1, 50, 20)
    outfit_max = st.number_input("Outfitting Max", 1, 50, 25)

fab_time = (fab_min, fab_max)
assy_time = (assy_min, assy_max)
erect_time = (erect_min, erect_max)
outfit_time = (outfit_min, outfit_max)

# Progress bar and per-stage display
progress_bar = st.progress(0)
status_text = st.empty()
stage_status = st.empty()

def update_progress(percent):
    progress_bar.progress(min(100, int(percent)))
    status_text.text(f"Simulation Progress: {percent:.1f}%")

def update_stage(ship, stage, done, total):
    stage_status.text(f"{ship} → {stage}: {done}/{total} weeks completed ({done/total*100:.1f}%)")
    time.sleep(0.05)  # smooth visible updates

# Run simulation button
if st.button("▶️ Run Real-Time Simulation"):
    df_records, completed = run_simulation_real_time(
        num_ships, fab_machines, assy_bays, erect_docks, outfit_berths,
        fab_time, assy_time, erect_time, outfit_time, sim_time,
        progress_callback=update_progress,
        stage_callback=update_stage
    )
    progress_bar.empty()
    status_text.text(f"Simulation Completed ✅ {completed}/{num_ships} Ships Completed")
    stage_status.empty()

    # Display Gantt chart
    st.subheader("🚀 Shipyard Gantt Chart")
    fig, ax = plt.subplots(figsize=(12, 6))
    stage_colors = {"Fabrication":"skyblue","Assembly":"orange","Erection":"lightgreen","Outfitting":"salmon"}

    for i, ship in enumerate(df_records["Ship"].unique()):
        ship_data = df_records[df_records["Ship"] == ship].sort_values(by="Start")
        for _, row in ship_data.iterrows():
            ax.barh(ship, row["Duration (weeks)"], left=row["Start"], color=stage_colors[row["Stage"]])
            ax.text(row["Start"] + row["Duration (weeks)"]/2, i,
                    f"{row['Stage']}", ha='center', va='center', fontsize=8, color='black')

    ax.set_xlabel("Simulation Time (weeks)")
    ax.set_ylabel("Ships")
    ax.set_title("Shipyard Gantt Chart")
    plt.tight_layout()
    st.pyplot(fig)
    # Display records