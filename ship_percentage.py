import simpy
import random
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------------
# 1️⃣ Streamlit UI Inputs
# -----------------------------------
st.title("⚓ Shipyard Production Simulation (Weekly Time Unit)")

num_ships = st.sidebar.slider("Number of Ships to Build", 1, 20, 10)
sim_time = st.sidebar.slider("Simulation Time (weeks)", 100, 1000, 500)

# Stage time ranges (in weeks)
fab_range = (6, 12)
assy_range = (8, 16)
erect_range = (10, 20)
outfit_range = (20, 40)

# Random seed for repeatability
random.seed(42)

# -----------------------------------
# 2️⃣ Global Variables for Tracking
# -----------------------------------
completed_ships = 0
progress_log = []  # For plotting completion % vs time


# -----------------------------------
# 3️⃣ Ship Process Definition
# -----------------------------------
def ship_process(env, ship_id, fabrication, assembly, erection, outfitting):
    global completed_ships

    start_time = env.now
    records = []

    # --- Fabrication ---
    with fabrication.request() as req:
        yield req
        fab_time = random.randint(*fab_range)
        yield env.timeout(fab_time)
        records.append((ship_id, "Fabrication", start_time, env.now))

    # --- Assembly ---
    with assembly.request() as req:
        yield req
        assy_time = random.randint(*assy_range)
        yield env.timeout(assy_time)
        records.append((ship_id, "Assembly", start_time + fab_time, env.now))

    # --- Erection ---
    with erection.request() as req:
        yield req
        erect_time = random.randint(*erect_range)
        yield env.timeout(erect_time)
        records.append((ship_id, "Erection", start_time + fab_time + assy_time, env.now))

    # --- Outfitting ---
    with outfitting.request() as req:
        yield req
        outfit_time = random.randint(*outfit_range)
        yield env.timeout(outfit_time)
        records.append((ship_id, "Outfitting", start_time + fab_time + assy_time + erect_time, env.now))

    # Ship completed ✅
    completed_ships += 1


# -----------------------------------
# 4️⃣ Progress Monitor Process
# -----------------------------------
def progress_monitor(env, total_ships, interval=10):
    """Logs completion % every 'interval' weeks."""
    while True:
        percent = (completed_ships / total_ships) * 100
        progress_log.append((env.now, percent))
        yield env.timeout(interval)
        if env.now >= sim_time:
            break


# -----------------------------------
# 5️⃣ Run Simulation
# -----------------------------------
def run_simulation():
    global completed_ships, progress_log
    completed_ships = 0
    progress_log = []

    env = simpy.Environment()

    # Define resources (machine/workshop capacity)
    fabrication = simpy.Resource(env, capacity=2)
    assembly = simpy.Resource(env, capacity=2)
    erection = simpy.Resource(env, capacity=1)
    outfitting = simpy.Resource(env, capacity=1)

    # Start progress monitor
    env.process(progress_monitor(env, num_ships))

    # Launch all ships
    for i in range(num_ships):
        env.process(ship_process(env, f"Ship-{i+1}", fabrication, assembly, erection, outfitting))

    # Run environment
    env.run(until=sim_time)

    # Create progress dataframe
    df_progress = pd.DataFrame(progress_log, columns=["Week", "Completion (%)"])

    return df_progress


# -----------------------------------
# 6️⃣ Run and Display Results
# -----------------------------------
if st.button("▶️ Run Simulation"):
    df_progress = run_simulation()

    total = num_ships
    done = completed_ships
    incomplete = total - done
    completion = (done / total) * 100

    # Summary
    st.subheader("📊 Simulation Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ships", total)
    col2.metric("Completed Ships", done)
    col3.metric("Incomplete Ships", incomplete)
    col4.metric("Completion (%)", f"{completion:.1f}%")

    # Line Chart for Progress
    st.subheader("📈 Shipyard Progress Over Time")
    fig = px.line(df_progress, x="Week", y="Completion (%)",
                  title="Ship Completion Percentage vs Simulation Time",
                  markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("⏱️ Time Unit: Weeks | Adjust simulation time to reach 100% completion.")

else:
    st.info("👆 Set your parameters and click **Run Simulation** to start.")
