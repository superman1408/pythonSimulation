import pandas as pd
import numpy as np

# ------------------------------
# STEP 1: Define Processes and Parameters
# ------------------------------

processes = pd.DataFrame([
    {"ProcessOrder": 1, "Process": "Cutting",    "Resource": "Cutters", "ResourceCount": 2, "MeanTime": 4.0, "StdDev": 1.0, "Dist": "Normal"},
    {"ProcessOrder": 2, "Process": "Welding",    "Resource": "Welders", "ResourceCount": 3, "MeanTime": 8.0, "StdDev": 2.0, "Dist": "Normal"},
    {"ProcessOrder": 3, "Process": "Painting",   "Resource": "Painters","ResourceCount": 1, "MeanTime": 6.0, "StdDev": 1.5, "Dist": "Normal"},
    {"ProcessOrder": 4, "Process": "Outfitting", "Resource": "Fitters", "ResourceCount": 2, "MeanTime": 10.0,"StdDev": 3.0, "Dist": "Normal"}
])

resources = processes[["Resource", "ResourceCount"]].drop_duplicates().reset_index(drop=True)

# ------------------------------
# STEP 2: Define Jobs (Ship Blocks)
# ------------------------------
num_jobs = 10
jobs = pd.DataFrame({
    "JobID": [f"Block_{i+1}" for i in range(num_jobs)],
    "ReleaseTime": [0] * num_jobs
})

# ------------------------------
# STEP 3: Define a Sampling Function for Process Times
# ------------------------------
def sample_process_time(mean, std, dist="Normal"):
    """Sample processing time for a task."""
    if dist == "Normal":
        return max(0.1, np.random.normal(mean, std))
    else:
        return mean

# ------------------------------
# STEP 4: Initialize Resources and Simulation Variables
# ------------------------------
np.random.seed(42)
resource_servers = {res: [0.0] * count for res, count in zip(resources["Resource"], resources["ResourceCount"])}

event_log = []
job_completion_times = {}

# ------------------------------
# STEP 5: Run the Discrete Event Simulation
# ------------------------------
for _, job in jobs.iterrows():
    job_id = job["JobID"]
    current_time = job["ReleaseTime"]

    for _, proc in processes.sort_values("ProcessOrder").iterrows():
        res_name = proc["Resource"]
        # Find earliest available server
        next_free_times = resource_servers[res_name]
        server_index = int(np.argmin(next_free_times))
        server_available = next_free_times[server_index]

        # Determine start and end times
        start_time = max(current_time, server_available)
        duration = sample_process_time(proc["MeanTime"], proc["StdDev"], proc["Dist"])
        end_time = start_time + duration

        # Update resource availability
        resource_servers[res_name][server_index] = end_time

        # Log event
        event_log.append({
            "JobID": job_id,
            "Process": proc["Process"],
            "Resource": res_name,
            "Server": server_index + 1,
            "Start": round(start_time, 3),
            "Duration": round(duration, 3),
            "End": round(end_time, 3)
        })

        current_time = end_time

    job_completion_times[job_id] = current_time

# ------------------------------
# STEP 6: Generate Event Log and Performance Metrics
# ------------------------------
event_df = pd.DataFrame(event_log)

# Compute make_span (overall completion time)
make_span = max(job_completion_times.values())

# Compute average job flow time
avg_flow_time = np.mean(list(job_completion_times.values()))

# Compute utilization for each resource
utilization_data = []
for res_name, servers in resource_servers.items():
    busy_time = sum(servers)
    utilization = busy_time / (len(servers) * make_span)
    utilization_data.append({
        "Resource": res_name,
        "Servers": len(servers),
        "Utilization": round(utilization, 3)
    })
util_df = pd.DataFrame(utilization_data)

# Summary results
results = pd.DataFrame([
    {"Metric": "Make_span (max completion time)", "Value": round(make_span, 3)},
    {"Metric": "Average Flow Time", "Value": round(avg_flow_time, 3)},
    {"Metric": "Throughput (jobs completed)", "Value": len(job_completion_times)}
])

# ------------------------------
# STEP 7: Display Simulation Results
# ------------------------------
print("===== EVENT LOG (first 10 events) =====")
print(event_df.head(10), "\n")

print("===== PERFORMANCE METRICS =====")
print(results, "\n")

print("===== RESOURCE UTILIZATION =====")
print(util_df, "\n")

# Optionally, export to Excel
event_df.to_excel("Shipyard_DES_EventLog.xlsx", index=False)
results.to_excel("Shipyard_DES_Results.xlsx", index=False)
util_df.to_excel("Shipyard_DES_Utilization.xlsx", index=False)
