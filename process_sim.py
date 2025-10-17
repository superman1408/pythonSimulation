import streamlit as st
import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt

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
    random_seed=42
):
    random.seed(random_seed)
    records = []
    env = simpy.Environment()

    # Define resources
    fab = simpy.Resource(env, capacity=fab_machines)
    assy = simpy.Resource(env, capacity=assy_bays)
    erect = simpy.Resource(env, capacity=erect_docks)
    outfit = simpy.Resource(env, capacity=outfit_berths)

    def record_event(ship, stage, start, end):
        records.append({
            "Ship": ship,
            "Stage": stage,
            "Start": start,
            "End": end,
            "Duration": end - start
        })

    def process_stage(ship, stage_name, resource, duration_range):
        with resource.request() as req:
            yield req
            start = env.now
            t = random.randint(*duration_range)
            yield env.timeout(t)
            end = env.now
            record_event(ship, stage_name, start, end)

    def ship_process(ship):
        yield env.process(process_stage(ship, "Fabrication", fab, fab_time))
        yield env.process(process_stage(ship, "Assembly", assy, assy_time))
        yield env.process(process_stage(ship, "Erection", erect, erect_time))
        yield env.process(process_stage(ship, "Outfitting", outfit, outfit_time))

    def shipyard(env):
        for i in range(num_ships):
            env.process(ship_process(f"Ship-{i+1}"))
            yield env.timeout(random.randint(3, 8))  # stagger arrivals

    env.process(shipyard(env))
    env.run(until=sim_time)

    df = pd.DataFrame(records)
    return df

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🚢 Shipyard Simulation (Streamlit Version)")
st.write("Simulate ship fabrication, assembly, erection, and outfitting with customizable parameters.")

mode = st.radio("Select Mode", ["Default", "User Defined"], horizontal=True)

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
    st.subheader("⚙️ User Inputs")
    num_ships = st.number_input("Number of Ships", min_value=1, value=5)
    fab_machines = st.number_input("Fabrication Machines", min_value=1, value=3)
    assy_bays = st.number_input("Assembly Bays", min_value=1, value=2)
    erect_docks = st.number_input("Erection Docks", min_value=1, value=1)
    outfit_berths = st.number_input("Outfitting Berths", min_value=1, value=1)
    sim_time = st.number_input("Simulation Time", min_value=50, value=500)

    st.markdown("#### ⏱️ Time Ranges for Each Stage")
    col1, col2 = st.columns(2)
    with col1:
        fab_min = st.number_input("Fabrication Time - Min", min_value=1, value=8)
        assy_min = st.number_input("Assembly Time - Min", min_value=1, value=10)
        erect_min = st.number_input("Erection Time - Min", min_value=1, value=12)
        outfit_min = st.number_input("Outfitting Time - Min", min_value=1, value=15)
    with col2:
        fab_max = st.number_input("Fabrication Time - Max", min_value=1, value=12)
        assy_max = st.number_input("Assembly Time - Max", min_value=1, value=15)
        erect_max = st.number_input("Erection Time - Max", min_value=1, value=20)
        outfit_max = st.number_input("Outfitting Time - Max", min_value=1, value=25)

    fab_time = (fab_min, fab_max)
    assy_time = (assy_min, assy_max)
    erect_time = (erect_min, erect_max)
    outfit_time = (outfit_min, outfit_max)

# Run Simulation Button
if st.button("▶️ Run Simulation"):
    df = run_simulation(
        num_ships,
        fab_machines,
        assy_bays,
        erect_docks,
        outfit_berths,
        fab_time,
        assy_time,
        erect_time,
        outfit_time,
        sim_time
    )

    st.subheader("📊 Stage Timing Records")
    st.dataframe(df.sort_values(by=["Ship", "Start"]).reset_index(drop=True))

    # Compute completion times
    summary = df.groupby("Ship")["End"].max().reset_index().rename(columns={"End": "CompletionDay"})
    st.subheader("🚀 Ship Completion Summary")
    st.dataframe(summary)

    # Gantt Chart
    st.subheader("📈 Gantt Chart")
    stages = ["Fabrication", "Assembly", "Erection", "Outfitting"]
    colors = {
        "Fabrication": "skyblue",
        "Assembly": "orange",
        "Erection": "lightgreen",
        "Outfitting": "salmon"
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, ship in enumerate(df["Ship"].unique()):
        ship_data = df[df["Ship"] == ship]
        for _, row in ship_data.iterrows():
            ax.barh(ship, row["Duration"], left=row["Start"], color=colors[row["Stage"]])
            ax.text(row["Start"] + row["Duration"]/2, i, row["Stage"], ha='center', va='center', fontsize=8)

    ax.set_xlabel("Simulation Time (days)")
    ax.set_ylabel("Ships")
    ax.set_title("Shipyard Production Gantt Chart")
    plt.tight_layout()
    st.pyplot(fig)
