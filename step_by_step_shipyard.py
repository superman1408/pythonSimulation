# shipyard_sim.py
import simpy
import random
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import time

# ------------------------------
# 1️⃣ Streamlit UI Inputs
# ------------------------------
st.title("Step-by-Step Shipyard Simulation")

num_jobs = st.sidebar.number_input("Number of Ship Blocks", min_value=1, value=5)
step_minutes = st.sidebar.number_input("Simulation Step (minutes)", min_value=1, value=10)
real_delay = st.sidebar.number_input("Real Delay per Step (seconds)", min_value=0.0, value=0.5)
failure_prob = st.sidebar.slider("Failure Probability per Step", min_value=0.0, max_value=1.0, value=0.05)

# Define processes and resources
processes = pd.DataFrame([
    {"ProcessOrder": 1, "Process": "Cutting",    "Resource": "Cutter", "Count": 2, "MeanTime": 4.0, "StdDev": 1.0},
    {"ProcessOrder": 2, "Process": "Welding",    "Resource": "Welder", "Count": 2, "MeanTime": 6.0, "StdDev": 1.5},
    {"ProcessOrder": 3, "Process": "Painting",   "Resource": "Painter","Count": 1, "MeanTime": 5.0, "StdDev": 1.0},
    {"ProcessOrder": 4, "Process": "Outfitting", "Resource": "Fitter", "Count": 1, "MeanTime": 7.0, "StdDev": 2.0}
])

# ------------------------------
# 2️⃣ Simulation Setup
# ------------------------------
random.seed(42)
np.random.seed(42)

env = simpy.Environment()

# Resources
resources = {row['Resource']: simpy.Resource(env, capacity=row['Count'])
             for _, row in processes.iterrows()}

# Event logging
event_log = []
active_segments = []
job_completion = {}

# ------------------------------
# 3️⃣ Helper Functions
# ------------------------------
def sample_time(mean, std):
    """Sample positive duration from normal distribution."""
    t = random.normalvariate(mean, std)
    return max(0.1, t)

def job_process(env, job_id):
    """SimPy process for a single job."""
    current_time = 0.0
    for _, proc in processes.sort_values("ProcessOrder").iterrows():
        res_name = proc['Resource']
        duration = sample_time(proc['MeanTime'], proc['StdDev'])
        with resources[res_name].request() as req:
            yield req
            start_time = env.now
            # Add to active segments for live Gantt
            segment = {"JobID": job_id, "Process": proc['Process'], "Resource": res_name, "Start": start_time}
            active_segments.append(segment)
            try:
                # Work may be interrupted randomly
                yield env.timeout(duration)
            except simpy.Interrupt:
                # Partial work logging
                segment['End'] = env.now
                segment['Duration'] = env.now - start_time
                event_log.append(segment)
                active_segments.remove(segment)
                # Resume work
                yield env.timeout(duration - (env.now - start_time))
            # Finished normally
            segment['End'] = env.now
            segment['Duration'] = duration
            event_log.append(segment)
            active_segments.remove(segment)
            current_time = env.now
    job_completion[job_id] = current_time

# Failure generator (randomly interrupts running segments)
def failure_generator(env):
    while True:
        yield env.timeout(1)  # check every simulated minute
        for seg in list(active_segments):
            if random.random() < failure_prob:
                seg_proc = seg.get('process')
                if seg_proc:
                    seg_proc.interrupt()

# ------------------------------
# 4️⃣ Initialize Jobs
# ------------------------------
jobs = [f"Block_{i+1}" for i in range(num_jobs)]
for job_id in jobs:
    p = env.process(job_process(env, job_id))

# env.process(failure_generator(env))  # Optional failures (could be skipped for simplicity)

# ------------------------------
# 5️⃣ Streamlit Placeholders
# ------------------------------
kpi_ph = st.empty()
gantt_ph = st.empty()
util_ph = st.empty()
log_ph = st.empty()
progress_ph = st.empty()

# ------------------------------
# 6️⃣ Step-by-Step Simulation Loop
# ------------------------------
sim_end = 2000  # total simulation time in minutes
step = step_minutes

while env.now < sim_end:
    env.run(until=min(env.now + step, sim_end))

    # Event log snapshot
    if event_log:
        df_events = pd.DataFrame(event_log)
    else:
        df_events = pd.DataFrame(columns=["JobID","Process","Resource","Start","End","Duration"])

    # KPIs
    if job_completion:
        makespan = max(job_completion.values())
        avg_flow = np.mean(list(job_completion.values()))
        throughput = len(job_completion)
    else:
        makespan = avg_flow = throughput = 0

    kpi_ph.markdown(f"**Simulation Time:** {env.now:.2f} min | **Makespan:** {makespan:.2f} | **Avg Flow:** {avg_flow:.2f} | **Throughput:** {throughput}")

    # Gantt chart
    if not df_events.empty:
        fig_gantt = px.timeline(df_events, x_start="Start", x_end="End", y="JobID", color="Process")
        fig_gantt.update_yaxes(autorange="reversed")
        gantt_ph.plotly_chart(fig_gantt, use_container_width=True, key=f"gantt_{env.now}")

    # Resource utilization
    util_data = []
    for res_name, res_obj in resources.items():
        busy = sum([seg['Duration'] for seg in event_log if seg['Resource']==res_name])
        util = busy / (len(res_obj.queue) + res_obj.count) / max(env.now,1)
        util_data.append({"Resource": res_name, "Utilization": util})
    if util_data:
        df_util = pd.DataFrame(util_data)
        fig_util = px.bar(df_util, x="Resource", y="Utilization", range_y=[0,1])
        util_ph.plotly_chart(fig_util, use_container_width=True, key=f"util_{env.now}")

    # Event log table
    log_ph.dataframe(df_events)

    # Progress bar
    progress_ph.progress(min(env.now / sim_end, 1.0))

    # Real delay for visualization
    time.sleep(real_delay)

st.success("Simulation Completed!")
