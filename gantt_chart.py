# import pandas as pd
# import plotly.express as px

# # -----------------------------
# # 1️⃣ Sample project data
# # -----------------------------
# data = {
#     "Task": [
#         "Project Initiation",
#         "Requirement Gathering",
#         "Design Phase",
#         "Development Phase",
#         "Testing",
#         "Deployment",
#         "Project Closure"
#     ],
#     "Start": [
#         "2025-10-01",
#         "2025-10-03",
#         "2025-10-07",
#         "2025-10-15",
#         "2025-11-01",
#         "2025-11-10",
#         "2025-11-20"
#     ],
#     "Finish": [
#         "2025-10-02",
#         "2025-10-06",
#         "2025-10-14",
#         "2025-10-31",
#         "2025-11-09",
#         "2025-11-19",
#         "2025-11-22"
#     ],
#     "Resource": [
#         "Management",
#         "Business Team",
#         "Design Team",
#         "Dev Team",
#         "QA Team",
#         "Ops Team",
#         "Management"
#     ]
# }

# df = pd.DataFrame(data)

# # -----------------------------
# # 2️⃣ Create the Gantt chart
# # -----------------------------
# fig = px.timeline(
#     df,
#     x_start="Start",
#     x_end="Finish",
#     y="Task",
#     color="Resource",           # color by team/resource
#     title="📊 Project Gantt Chart",
#     labels={"Resource": "Team"}
# )

# # Reverse the Y-axis to show tasks top-down
# fig.update_yaxes(autorange="reversed")

# # Optional: improve layout
# fig.update_layout(
#     xaxis_title="Timeline",
#     yaxis_title="Project Tasks",
#     font=dict(size=12),
#     hoverlabel_align='left'
# )

# # -----------------------------
# # 3️⃣ Show the chart
# # -----------------------------
# fig.show()


import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Parameters
# ------------------------------
RANDOM_SEED = 42
NUM_SHIPS = 5
FAB_MACHINES = 3
ASSEMBLY_BAYS = 2
ERECTION_DOCKS = 1
OUTFITTING_BERTHS = 1
SIM_TIME = 500

FAB_TIME = (8, 12)
ASSY_TIME = (10, 15)
ERECT_TIME = (12, 20)
OUTFIT_TIME = (15, 25)

# To store results
records = []

# ------------------------------
# Process Definitions
# ------------------------------
def record_event(ship, stage, start, end):
    records.append({
        "Ship": ship,
        "Stage": stage,
        "Start": start,
        "End": end,
        "Duration": end - start
    })

def process_stage(env, ship, stage_name, resource, duration_range):
    with resource.request() as req:
        yield req
        start = env.now
        t = random.randint(*duration_range)
        yield env.timeout(t)
        end = env.now
        record_event(ship, stage_name, start, end)
        print(f"{ship} finished {stage_name} at day {end:.1f}")

def ship_process(env, ship, fab, assy, erect, outfit):
    print(f"{ship} enters yard at day {env.now:.1f}")
    yield env.process(process_stage(env, ship, "Fabrication", fab, FAB_TIME))
    yield env.process(process_stage(env, ship, "Assembly", assy, ASSY_TIME))
    yield env.process(process_stage(env, ship, "Erection", erect, ERECT_TIME))
    yield env.process(process_stage(env, ship, "Outfitting", outfit, OUTFIT_TIME))
    print(f"✅ {ship} completed total build at day {env.now:.1f}\n")

# ------------------------------
# Simulation Setup
# ------------------------------
def run_shipyard_sim():
    random.seed(RANDOM_SEED)
    env = simpy.Environment()

    # Define resources
    fab = simpy.Resource(env, capacity=FAB_MACHINES)
    assy = simpy.Resource(env, capacity=ASSEMBLY_BAYS)
    erect = simpy.Resource(env, capacity=ERECTION_DOCKS)
    outfit = simpy.Resource(env, capacity=OUTFITTING_BERTHS)

    # Launch ship processes
    for i in range(NUM_SHIPS):
        env.process(ship_process(env, f"Ship-{i+1}", fab, assy, erect, outfit))
        yield env.timeout(random.randint(3, 8))  # stagger arrivals

    env.run(until=SIM_TIME)
    print("Simulation complete.")

# ------------------------------
# Run Simulation
# ------------------------------
if __name__ == "__main__":
    env = simpy.Environment()
    env.process(run_shipyard_sim())
    env.run()

    # Convert records to DataFrame
    df = pd.DataFrame(records)
    print("\n--- Summary Data ---")
    print(df)

    # Compute completion times per ship
    summary = df.groupby("Ship")["End"].max().reset_index().rename(columns={"End": "CompletionDay"})
    print("\n--- Ship Completion Summary ---")
    print(summary)

    # --------------------------
    # Visualization: Gantt Chart
    # --------------------------
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
    plt.show()
