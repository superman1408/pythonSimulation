import streamlit as st
import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

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
    progress_log = []
    completed_ships = {"count": 0}

    env = simpy.Environment()

    # Define resources
    fab = simpy.Resource(env, capacity=fab_machines)
    assy = simpy.Resource(env, capacity=assy_bays)
    erect = simpy.Resource(env, capacity=erect_docks)
    outfit = simpy.Resource(env, capacity=outfit_berths)

    # Record each stage
    def record_event(ship, stage, start, end):
        records.append({
            "Ship": ship,
            "Stage": stage,
            "Start": start,
            "End": end,
            "Duration (weeks)": end - start
        })

    # Process a single stage
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
        completed_ships["count"] += 1  # ship finished

    # Monitor progress
    def progress_monitor(env, interval=10):
        while True:
            percent = (completed_ships["count"] / num_ships) * 100
            progress_log.append((env.now, percent))
            yield env.timeout(interval)
            if env.now >= sim_time:
                break

    # Launch ships and progress monitor
    def shipyard(env):
        env.process(progress_monitor(env))
        for i in range(num_ships):
            env.process(ship_process(f"Ship-{i+1}"))
            yield env.timeout(random.randint(3, 8))  # staggered arrivals

    env.process(shipyard(env))
    env.run(until=sim_time)

    df_records = pd.DataFrame(records)
    df_progress = pd.DataFrame(progress_log, columns=["Week", "Completion (%)"])

    # Final metrics
    completed = completed_ships["count"]
    incomplete = num_ships - completed
    completion_percent = (completed / num_ships) * 100

    return df_records, df_progress, completed, incomplete, completion_percent


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="ASHKAM Shipyard Simulator", layout="wide")
st.title("🚢 Shipyard Simulation Dashboard (Weekly Time Unit)")
st.write("Simulate ship construction and track cumulative progress over time.")
st.caption("⏱️ All durations and simulation time are measured in **weeks**.")

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

    st.markdown("#### ⏱️ Stage Duration Ranges (Weeks)")
    col1, col2 = st.columns(2)
    with col1:
        fab_min = st.number_input("Fabrication Min", min_value=1, value=8)
        assy_min = st.number_input("Assembly Min", min_value=1, value=10)
        erect_min = st.number_input("Erection Min", min_value=1, value=12)
        outfit_min = st.number_input("Outfitting Min", min_value=1, value=15)
    with col2:
        fab_max = st.number_input("Fabrication Max", min_value=1, value=12)
        assy_max = st.number_input("Assembly Max", min_value=1, value=15)
        erect_max = st.number_input("Erection Max", min_value=1, value=20)
        outfit_max = st.number_input("Outfitting Max", min_value=1, value=25)

    fab_time = (fab_min, fab_max)
    assy_time = (assy_min, assy_max)
    erect_time = (erect_min, erect_max)
    outfit_time = (outfit_min, outfit_max)


# ------------------------------
# Run Simulation Button
# ------------------------------
if st.button("▶️ Run Simulation"):
    df_records, df_progress, completed, incomplete, completion_percent = run_simulation(
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

    # ---- Summary Metrics ----
    st.subheader("📊 Simulation Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ships", num_ships)
    col2.metric("Completed Ships", completed)
    col3.metric("Incomplete Ships", incomplete)
    col4.metric("Completion (%)", f"{completion_percent:.1f}%")

    # ---- Progress Chart ----
    st.subheader("📈 Shipyard Completion Progress Over Time")
    fig_progress = px.line(
        df_progress,
        x="Week",
        y="Completion (%)",
        title="Ship Completion (%) vs Simulation Time (Weeks)",
        markers=True
    )
    st.plotly_chart(fig_progress, use_container_width=True)

    # ---- Gantt Chart ----
    # st.subheader("🚀 Shipyard Gantt Chart (Weeks)")
    # colors = {
    #     "Fabrication": "skyblue",
    #     "Assembly": "orange",
    #     "Erection": "lightgreen",
    #     "Outfitting": "salmon"
    # }

    # fig, ax = plt.subplots(figsize=(10, 5))
    # for i, ship in enumerate(df_records["Ship"].unique()):
    #     ship_data = df_records[df_records["Ship"] == ship]
    #     for _, row in ship_data.iterrows():
    #         ax.barh(ship, row["Duration (weeks)"], left=row["Start"], color=colors[row["Stage"]])
    #         ax.text(row["Start"] + row["Duration (weeks)"]/2, i, row["Stage"], ha='center', va='center', fontsize=8)

    # ax.set_xlabel("Simulation Time (weeks)")
    # ax.set_ylabel("Ships")
    # ax.set_title("Shipyard Production Gantt Chart")
    # plt.tight_layout()
    # st.pyplot(fig)
    # ---- Gantt Chart with Partial Stage Progress ----
    st.subheader("🚀 Shipyard Gantt Chart (Partial Stage Completion)")

    fig, ax = plt.subplots(figsize=(12, 6))

    stage_colors = {
        "Fabrication": "skyblue",
        "Assembly": "orange",
        "Erection": "lightgreen",
        "Outfitting": "salmon"
    }

    for i, ship in enumerate(df_records["Ship"].unique()):
        ship_data = df_records[df_records["Ship"] == ship].sort_values(by="Start")
        
        for _, row in ship_data.iterrows():
            # Calculate actual duration completed within simulation time
            actual_duration = max(0, min(row["End"], sim_time) - row["Start"])
            if actual_duration <= 0:
                continue  # stage not started yet in simulation

            # Compute % completion of the stage
            stage_percent = (actual_duration / row["Duration (weeks)"]) * 100

            # Adjust color intensity or hatch to show partial completion
            base_color = stage_colors[row["Stage"]]
            from matplotlib.colors import to_rgba
            rgba_color = to_rgba(base_color, alpha=0.7)  # keep color solid
            ax.barh(ship, actual_duration, left=row["Start"], color=rgba_color)
            ax.text(row["Start"] + actual_duration/2, i,
                    f"{row['Stage']} ({stage_percent:.0f}%)",
                    ha='center', va='center', fontsize=8, color='black')

    ax.set_xlabel("Simulation Time (weeks)")
    ax.set_ylabel("Ships")
    ax.set_title("Shipyard Gantt Chart with Partial Stage Completion")
    plt.tight_layout()
    st.pyplot(fig)


    # ---- Completion Summary per Ship ----
    summary = df_records.groupby("Ship")["End"].max().reset_index().rename(columns={"End": "Completion Week"})
    st.subheader("🏁 Ship Completion Summary")
    st.dataframe(summary)

    # ---- S-Curve Plot ----
    st.subheader("📊 S-Curve: Ship Completion Progress Over Simulation Time")
    completion_times = sorted(summary["Completion Week"].tolist())
    cumulative_ships = list(range(1, len(completion_times) + 1))

    # Actual progress
    actual_df = pd.DataFrame({
        "Simulation Week": completion_times,
        "Ships Completed (Actual)": cumulative_ships
    })

    # Ideal progress (linear)
    ideal_weeks = list(range(0, sim_time + 1, sim_time // num_ships if num_ships > 0 else 1))
    ideal_ships = list(range(0, num_ships + 1))
    ideal_df = pd.DataFrame({
        "Simulation Week": ideal_weeks,
        "Ships Completed (Ideal)": ideal_ships
    })

    # Plot
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(actual_df["Simulation Week"], actual_df["Ships Completed (Actual)"], marker="o", label="Actual Progress")
    ax2.plot(ideal_df["Simulation Week"], ideal_df["Ships Completed (Ideal)"], linestyle="--", color="gray", label="Ideal Progress")

    ax2.set_xlabel("Simulation Time (Weeks)")
    ax2.set_ylabel("Cumulative Ships Completed")
    ax2.set_title("S-Curve: Actual vs Ideal Ship Completion")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    st.pyplot(fig2)
