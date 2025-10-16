import simpy
import pandas as pd
import numpy as np
import random

# ------------------------------
# STEP 1: Define Process Data
# ------------------------------
processes = pd.DataFrame([
    {"Order": 1, "Name": "Cutting",    "Resource": "Cutter",   "Count": 2, "Mean": 4.0,  "Std": 1.0},
    {"Order": 2, "Name": "Welding",    "Resource": "Welder",   "Count": 3, "Mean": 8.0,  "Std": 2.0},
    {"Order": 3, "Name": "Painting",   "Resource": "Painter",  "Count": 1, "Mean": 6.0,  "Std": 1.5},
    {"Order": 4, "Name": "Outfitting", "Resource": "Fitter",   "Count": 2, "Mean": 10.0, "Std": 3.0}
])

# Machine breakdown settings (example for Welding only)
WELDER_MEAN_TIME_TO_FAILURE = 100.0   # minutes
WELDER_REPAIR_TIME = 30.0             # minutes

# Jobs
NUM_JOBS = 10
jobs = [f"Block_{i+1}" for i in range(NUM_JOBS)]
RELEASE_TIME = 0

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ------------------------------
# STEP 2: Sampling Function
# ------------------------------
def sample_time(mean, std):
    """Sample process duration from normal distribution with truncation at 0.1."""
    t = np.random.normal(mean, std)
    return max(0.1, t)

# ------------------------------
# STEP 3: Shipyard Simulation Environment
# ------------------------------
class ShipyardSimulation:
    def __init__(self, env):
        self.env = env
        # Create SimPy resources
        self.resources = {
            row.Resource: simpy.PreemptiveResource(env, capacity=row.Count)
            for _, row in processes.iterrows()
        }
        # Event log
        self.event_log = []
        # Job completion times
        self.job_completion_times = {}
        # Breakdown process references
        self.welder_broken = False
        self.welder_processes = []

        # Start breakdown generator for welders
        env.process(self.break_welder_machines())

    def job_process(self, job_id, release_time=0):
        """Process representing one ship block moving through all stages."""
        yield self.env.timeout(release_time)  # Wait until release

        start_job = self.env.now
        for _, proc in processes.sort_values("Order").iterrows():
            resource_name = proc.Resource
            resource = self.resources[resource_name]

            # Request the resource
            with resource.request(priority=1) as req:
                start_wait = self.env.now
                yield req  # Wait for availability

                start_time = self.env.now
                duration = sample_time(proc.Mean, proc.Std)

                try:
                    yield self.env.timeout(duration)
                    end_time = self.env.now
                except simpy.Interrupt:
                    # Handle interruption: machine breakdown mid-process
                    remaining = duration - (self.env.now - start_time)
                    # Wait for repair to finish (breakdown handler handles it)
                    yield self.env.timeout(remaining)
                    end_time = self.env.now

                # Log the event
                self.event_log.append({
                    "JobID": job_id,
                    "Process": proc.Name,
                    "Resource": resource_name,
                    "Start": round(start_time, 3),
                    "End": round(end_time, 3),
                    "Duration": round(end_time - start_time, 3)
                })

        # Mark job completion
        self.job_completion_times[job_id] = self.env.now - start_job

    def break_welder_machines(self):
        """Random breakdown generator for welders, interrupting ongoing jobs."""
        while True:
            time_to_failure = random.expovariate(1.0 / WELDER_MEAN_TIME_TO_FAILURE)
            yield self.env.timeout(time_to_failure)
            self.welder_broken = True

            # Interrupt all welding processes currently active
            for proc in list(self.welder_processes):
                if not proc.triggered:
                    proc.interrupt()

            # Repair time
            yield self.env.timeout(WELDER_REPAIR_TIME)
            self.welder_broken = False

# ------------------------------
# STEP 4: Run the Simulation
# ------------------------------
env = simpy.Environment()
shipyard = ShipyardSimulation(env)

# Launch all job processes
for job in jobs:
    env.process(shipyard.job_process(job, RELEASE_TIME))

# Run simulation
env.run()

# ------------------------------
# STEP 5: Analyze Results
# ------------------------------
event_df = pd.DataFrame(shipyard.event_log)
makespan = event_df["End"].max()
avg_flow_time = np.mean(list(shipyard.job_completion_times.values()))

# Compute simple resource utilization
utilization_data = []
for resource_name, resource in shipyard.resources.items():
    # Sum all durations for that resource
    busy_time = event_df[event_df["Resource"] == resource_name]["Duration"].sum()
    utilization = busy_time / (resource.capacity * makespan)
    utilization_data.append({
        "Resource": resource_name,
        "Servers": resource.capacity,
        "Utilization": round(utilization, 3)
    })
util_df = pd.DataFrame(utilization_data)

summary_df = pd.DataFrame([
    {"Metric": "Makespan", "Value": round(makespan, 2)},
    {"Metric": "Average Flow Time", "Value": round(avg_flow_time, 2)},
    {"Metric": "Jobs Completed", "Value": len(jobs)}
])

# ------------------------------
# STEP 6: Print Summary
# ------------------------------
print("===== EVENT LOG (first 10) =====")
print(event_df.head(10), "\n")

print("===== SUMMARY METRICS =====")
print(summary_df, "\n")

print("===== RESOURCE UTILIZATION =====")
print(util_df, "\n")

# Optional: export to Excel
event_df.to_excel("Shipyard_SimPy_EventLog.xlsx", index=False)
summary_df.to_excel("Shipyard_SimPy_Summary.xlsx", index=False)
util_df.to_excel("Shipyard_SimPy_Utilization.xlsx", index=False)
