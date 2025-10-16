# # shipyard_sim_interruptions_fixed.py
# import simpy # pyright: ignore[reportMissingImports]
# import random
# import pandas as pd
# import numpy as np
# import streamlit as st # pyright: ignore[reportMissingImports]
# import plotly.express as px # pyright: ignore[reportMissingImports]
# import time

# # ------------------------------
# # 1️⃣ Streamlit UI Inputs
# # ------------------------------
# st.set_page_config(layout="wide", page_title="Shipyard Manufacturing Simulator")
# st.title("Step-by-Step Shipyard Manufacturing Simulation with Interruptions")

# num_jobs = st.sidebar.number_input("Number of Ship Blocks", min_value=1, value=5)
# step_minutes = st.sidebar.number_input("Simulation Step (minutes)", min_value=1, value=10)
# real_delay = st.sidebar.number_input("Real Delay per Step (seconds)", min_value=0.0, value=0.5)
# failure_prob = st.sidebar.slider("Failure Probability per Step", min_value=0.0, max_value=1.0, value=0.05)
# repair_time = st.sidebar.number_input("Repair Time (minutes)", min_value=1.0, value=3.0)

# # Define processes and resources
# processes = pd.DataFrame([
#     {"ProcessOrder": 1, "Process": "Cutting",    "Resource": "Cutter", "Count": 2, "MeanTime": 4.0, "StdDev": 1.0},
#     {"ProcessOrder": 2, "Process": "Welding",    "Resource": "Welder", "Count": 2, "MeanTime": 6.0, "StdDev": 1.5},
#     {"ProcessOrder": 3, "Process": "Painting",   "Resource": "Painter","Count": 1, "MeanTime": 5.0, "StdDev": 1.0},
#     {"ProcessOrder": 4, "Process": "Outfitting", "Resource": "Fitter", "Count": 1, "MeanTime": 7.0, "StdDev": 2.0}
# ])


# # ------------------------------
# # 1️⃣ Default Process Table
# # ------------------------------
# default_processes = pd.DataFrame([
#     {"ProcessOrder": 1, "Process": "Cutting",    "Resource": "Cutter", "Count": 2, "MeanTime": 4.0, "StdDev": 1.0},
#     {"ProcessOrder": 2, "Process": "Welding",    "Resource": "Welder", "Count": 2, "MeanTime": 6.0, "StdDev": 1.5},
#     {"ProcessOrder": 3, "Process": "Painting",   "Resource": "Painter","Count": 1, "MeanTime": 5.0, "StdDev": 1.0},
#     {"ProcessOrder": 4, "Process": "Outfitting", "Resource": "Fitter", "Count": 1, "MeanTime": 7.0, "StdDev": 2.0}
# ])


# # ------------------------------
# # 2️⃣ Editable Table in Sidebar
# # # ------------------------------
# st.sidebar.markdown("### Edit Process Table")
# edited_processes = st.sidebar.data_editor(default_processes, num_rows="dynamic")

# # ------------------------------
# # 3️⃣ Show Table
# # ------------------------------
# st.markdown("### Current Process Table")
# st.dataframe(edited_processes)

# # ------------------------------
# # 4️⃣ Example: Use in Simulation
# # ------------------------------
# if st.button("Start Simulation"):
#     st.write("Simulation would use this table:")
#     st.dataframe(edited_processes)
#     # Here you can pass `edited_processes` to your SimPy simulation instead of the fixed `processes`

# # ------------------------------
# # 2️⃣ Simulation Setup
# # ------------------------------
# random.seed(42)
# np.random.seed(42)
# env = simpy.Environment()

# resources = {row['Resource']: simpy.Resource(env, capacity=row['Count'])
#              for _, row in default_processes.iterrows()}

# event_log = []
# active_segments = []
# job_completion = {}

# # ------------------------------
# # 3️⃣ Helper Functions
# # ------------------------------
# def sample_time(mean, std):
#     t = random.normalvariate(mean, std)
#     return max(0.1, t)

# # ------------------------------
# # 4️⃣ Job Process with Interruptions
# # ------------------------------
# def job_process(env, job_id):
#     for _, proc in default_processes.sort_values("ProcessOrder").iterrows():
#         res_name = proc['Resource']
#         remaining_time = sample_time(proc['MeanTime'], proc['StdDev'])

#         while remaining_time > 0:
#             with resources[res_name].request() as req:
#                 yield req
#                 start_time = env.now
#                 segment = {
#                     "JobID": job_id,
#                     "Process": proc['Process'],
#                     "Resource": res_name,
#                     "Start": start_time
#                 }
#                 active_segments.append(segment)
#                 try:
#                     # Yield the timeout directly (do not wrap in env.process)
#                     yield env.timeout(remaining_time)
#                     segment['End'] = env.now
#                     segment['Duration'] = remaining_time
#                     event_log.append(segment)
#                     active_segments.remove(segment)
#                     remaining_time = 0
#                 except simpy.Interrupt:
#                     # Partial work logged
#                     done = env.now - start_time
#                     segment['End'] = env.now
#                     segment['Duration'] = done
#                     event_log.append(segment)
#                     active_segments.remove(segment)
#                     # Repair time
#                     yield env.timeout(repair_time)
#                     remaining_time -= done
#     job_completion[job_id] = env.now

# # ------------------------------
# # 5️⃣ Failure Generator
# # ------------------------------
# def failure_generator(env):
#     while True:
#         yield env.timeout(1)  # every simulated minute
#         for seg in list(active_segments):
#             if random.random() < failure_prob:
#                 # Interrupt the running job
#                 job_proc = seg.get('env_proc')
#                 if job_proc is not None:
#                     job_proc.interrupt()

# # ------------------------------
# # 6️⃣ Initialize Jobs
# # ------------------------------
# jobs = [f"Block_{i+1}" for i in range(num_jobs)]
# job_procs = []
# for job_id in jobs:
#     p = env.process(job_process(env, job_id))
#     job_procs.append(p)
#     # Store reference for interruption
#     for seg in active_segments:
#         seg['env_proc'] = p

# env.process(failure_generator(env))

# # ------------------------------
# # 7️⃣ Streamlit Placeholders
# # ------------------------------
# kpi_ph = st.empty()
# gantt_ph = st.empty()
# util_ph = st.empty()
# log_ph = st.empty()
# progress_ph = st.empty()

# # ------------------------------
# # 8️⃣ Step-by-Step Simulation Loop
# # ------------------------------
# sim_end = 2000
# step = step_minutes

# while env.now < sim_end:
#     env.run(until=min(env.now + step, sim_end))

#     # Event log snapshot
#     df_events = pd.DataFrame(event_log) if event_log else pd.DataFrame(
#         columns=["JobID","Process","Resource","Start","End","Duration"]
#     )

#     # KPIs
#     if job_completion:
#         makespan = max(job_completion.values())
#         avg_flow = np.mean(list(job_completion.values()))
#         throughput = len(job_completion)
#     else:
#         makespan = avg_flow = throughput = 0

#     kpi_ph.markdown(f"**Sim Time:** {env.now:.2f} min | **Makespan:** {makespan:.2f} | "
#                     f"**Avg Flow:** {avg_flow:.2f} | **Throughput:** {throughput}")
    
    
    

#     # Gantt chart
#     if not df_events.empty:
#         fig_gantt = px.timeline(df_events, x_start="Start", x_end="End", y="JobID", color="Process")
#         fig_gantt.update_yaxes(autorange="reversed")
#         gantt_ph.plotly_chart(fig_gantt, use_container_width=True, key=f"gantt_{env.now}")

#     # Resource utilization
#     util_data = []
#     for res_name, res_obj in resources.items():
#         busy = sum([seg['Duration'] for seg in event_log if seg['Resource']==res_name])
#         util = busy / max(env.now,1)
#         util_data.append({"Resource": res_name, "Utilization": min(util,1)})
#     if util_data:
#         df_util = pd.DataFrame(util_data)
#         fig_util = px.bar(df_util, x="Resource", y="Utilization", range_y=[0,1])
#         util_ph.plotly_chart(fig_util, use_container_width=True, key=f"util_{env.now}")

#     # Event log table
#     log_ph.dataframe(df_events)

#     # Progress bar
#     progress_ph.progress(min(env.now / sim_end, 1.0))

#     time.sleep(real_delay)

# st.success("Simulation Completed with Interruptions!")
# # Final event log



# # import streamlit as st
# # import pandas as pd

# # st.title("Dynamic Shipyard Process Table")

# # ------------------------------
# # 1️⃣ Default Process Table
# # ------------------------------
# # default_processes = pd.DataFrame([
# #     {"ProcessOrder": 1, "Process": "Cutting",    "Resource": "Cutter", "Count": 2, "MeanTime": 4.0, "StdDev": 1.0},
# #     {"ProcessOrder": 2, "Process": "Welding",    "Resource": "Welder", "Count": 2, "MeanTime": 6.0, "StdDev": 1.5},
# #     {"ProcessOrder": 3, "Process": "Painting",   "Resource": "Painter","Count": 1, "MeanTime": 5.0, "StdDev": 1.0},
# #     {"ProcessOrder": 4, "Process": "Outfitting", "Resource": "Fitter", "Count": 1, "MeanTime": 7.0, "StdDev": 2.0}
# # ])

# # ------------------------------
# # 2️⃣ Editable Table in Sidebar
# # # ------------------------------
# # st.sidebar.markdown("### Edit Process Table")
# # edited_processes = st.sidebar.data_editor(default_processes, num_rows="dynamic")

# # ------------------------------
# # 3️⃣ Show Table
# # ------------------------------
# # st.markdown("### Current Process Table")
# # st.dataframe(edited_processes)

# # ------------------------------
# # 4️⃣ Example: Use in Simulation
# # ------------------------------
# # if st.button("Start Simulation"):
# #     st.write("Simulation would use this table:")
# #     st.dataframe(edited_processes)
#     # Here you can pass `edited_processes` to your SimPy simulation instead of the fixed `processes`


# shipyard_sim_interruptions_fixed.py
import simpy  # pyright: ignore[reportMissingImports]
import random
import pandas as pd
import numpy as np
import streamlit as st  # pyright: ignore[reportMissingImports]
import plotly.express as px  # pyright: ignore[reportMissingImports]
import time

# -------------------------------------------------------
# 1️⃣ Streamlit Page Setup
# -------------------------------------------------------
st.set_page_config(layout="wide", page_title="Shipyard Manufacturing Simulator")
st.title("🚢 Shipyard Manufacturing Simulation with Interruptions")

# -------------------------------------------------------
# 2️⃣ Sidebar Inputs
# -------------------------------------------------------
num_jobs = st.sidebar.number_input("Number of Ship Blocks", min_value=1, value=5)
step_minutes = st.sidebar.number_input("Simulation Step (minutes)", min_value=1, value=10)
real_delay = st.sidebar.number_input("Real Delay per Step (seconds)", min_value=0.0, value=0.5)
failure_prob = st.sidebar.slider("Failure Probability per Step", min_value=0.0, max_value=1.0, value=0.05)
repair_time = st.sidebar.number_input("Repair Time (minutes)", min_value=1.0, value=3.0)

# -------------------------------------------------------
# 3️⃣ Default Process Table
# -------------------------------------------------------
default_processes = pd.DataFrame([
    {"ProcessOrder": 1, "Process": "Cutting",    "Resource": "Cutter", "Count": 2, "MeanTime": 4.0, "StdDev": 1.0},
    {"ProcessOrder": 2, "Process": "Welding",    "Resource": "Welder", "Count": 2, "MeanTime": 6.0, "StdDev": 1.5},
    {"ProcessOrder": 3, "Process": "Painting",   "Resource": "Painter","Count": 1, "MeanTime": 5.0, "StdDev": 1.0},
    {"ProcessOrder": 4, "Process": "Outfitting", "Resource": "Fitter", "Count": 1, "MeanTime": 7.0, "StdDev": 2.0}
])

# -------------------------------------------------------
# 4️⃣ Editable Table in Sidebar
# -------------------------------------------------------
st.sidebar.markdown("### 🛠 Edit Process Table")
edited_processes = st.sidebar.data_editor(default_processes, num_rows="dynamic")

# -------------------------------------------------------
# 5️⃣ Show Current Process Table
# -------------------------------------------------------
st.markdown("### 📋 Current Process Table")
st.dataframe(edited_processes)

# Start Simulation Button
start_sim = st.button("▶ Start Simulation")

if not start_sim:
    st.stop()

# -------------------------------------------------------
# 6️⃣ Simulation Setup
# -------------------------------------------------------
random.seed(42)
np.random.seed(42)
env = simpy.Environment()

resources = {row['Resource']: simpy.Resource(env, capacity=int(row['Count']))
             for _, row in edited_processes.iterrows()}

event_log = []
active_segments = []
job_completion = {}
job_process_refs = {}  # For interruption tracking

# -------------------------------------------------------
# 7️⃣ Helper Function
# -------------------------------------------------------
def sample_time(mean, std):
    t = random.normalvariate(mean, std)
    return max(0.1, t)

# -------------------------------------------------------
# 8️⃣ Job Process with Interruptions
# -------------------------------------------------------
def job_process(env, job_id):
    for _, proc in edited_processes.sort_values("ProcessOrder").iterrows():
        res_name = proc['Resource']
        remaining_time = sample_time(proc['MeanTime'], proc['StdDev'])

        while remaining_time > 0:
            with resources[res_name].request() as req:
                yield req
                start_time = env.now
                segment = {
                    "JobID": job_id,
                    "Process": proc['Process'],
                    "Resource": res_name,
                    "Start": start_time
                }
                active_segments.append(segment)
                job_process_refs[job_id] = env.active_process
                try:
                    yield env.timeout(remaining_time)
                    # Finished without interruption
                    segment['End'] = env.now
                    segment['Duration'] = remaining_time
                    event_log.append(segment)
                    active_segments.remove(segment)
                    remaining_time = 0
                except simpy.Interrupt:
                    # Interrupted mid-way
                    done = env.now - start_time
                    segment['End'] = env.now
                    segment['Duration'] = done
                    event_log.append(segment)
                    active_segments.remove(segment)
                    yield env.timeout(repair_time)
                    remaining_time -= done
    job_completion[job_id] = env.now

# -------------------------------------------------------
# 9️⃣ Failure Generator
# -------------------------------------------------------
def failure_generator(env):
    while True:
        yield env.timeout(1)  # check every simulated minute
        for seg in list(active_segments):
            if random.random() < failure_prob:
                job_id = seg['JobID']
                proc = job_process_refs.get(job_id)
                if proc:
                    proc.interrupt()

# -------------------------------------------------------
# 🔟 Initialize Jobs
# -------------------------------------------------------
jobs = [f"Block_{i+1}" for i in range(num_jobs)]
for job_id in jobs:
    env.process(job_process(env, job_id))

env.process(failure_generator(env))

# -------------------------------------------------------
# 11️⃣ Streamlit Placeholders
# -------------------------------------------------------
kpi_ph = st.empty()
gantt_ph = st.empty()
util_ph = st.empty()
log_ph = st.empty()
progress_ph = st.empty()

# -------------------------------------------------------
# 12️⃣ Step-by-Step Simulation Loop
# -------------------------------------------------------
sim_end = 2000
step = step_minutes

while env.now < sim_end:
    env.run(until=min(env.now + step, sim_end))

    # Event log snapshot
    df_events = pd.DataFrame(event_log) if event_log else pd.DataFrame(
        columns=["JobID", "Process", "Resource", "Start", "End", "Duration"]
    )

    # KPIs
    if job_completion:
        makespan = max(job_completion.values())
        avg_flow = np.mean(list(job_completion.values()))
        throughput = len(job_completion)
    else:
        makespan = avg_flow = throughput = 0

    kpi_ph.markdown(f"""
    **Sim Time:** {env.now:.1f} min | 
    **Makespan:** {makespan:.1f} min | 
    **Avg Flow Time:** {avg_flow:.1f} min | 
    **Throughput:** {throughput}
    """)

    # Gantt Chart
    if not df_events.empty:
        fig_gantt = px.timeline(df_events, x_start="Start", x_end="End", y="JobID", color="Process")
        fig_gantt.update_yaxes(autorange="reversed")
        fig_gantt.update_layout(title="Gantt Chart", height=500)
        gantt_ph.plotly_chart(fig_gantt, use_container_width=True, key=f"gantt_{env.now}")

    # Resource Utilization
    util_data = []
    for res_name, res_obj in resources.items():
        busy = sum([seg['Duration'] for seg in event_log if seg['Resource'] == res_name])
        util = busy / max(env.now, 1)
        util_data.append({"Resource": res_name, "Utilization": min(util, 1)})
    if util_data:
        df_util = pd.DataFrame(util_data)
        fig_util = px.bar(df_util, x="Resource", y="Utilization", range_y=[0, 1])
        fig_util.update_layout(title="Resource Utilization")
        util_ph.plotly_chart(fig_util, use_container_width=True, key=f"util_{env.now}")

    # Event Log Table
    log_ph.dataframe(df_events)

    # Progress Bar
    progress_ph.progress(min(env.now / sim_end, 1.0))

    time.sleep(real_delay)

st.success("✅ Simulation Completed with Interruptions!")
