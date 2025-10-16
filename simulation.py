# shipyard_sim_app.py
import simpy # pyright: ignore[reportMissingImports]
import random
import numpy as np
import pandas as pd
import streamlit as st # pyright: ignore[reportMissingImports]
import plotly.express as px # pyright: ignore[reportMissingImports]
from math import isfinite

# ------------------------------
# Utility / Sampling
# ------------------------------
def sample_normal_trunc(mean, std):
    t = np.random.normal(mean, std)
    return max(0.01, float(t))

# ------------------------------
# SimPy Shipyard with per-process failures
# ------------------------------
class ShipyardSim:
    def __init__(self, env, processes_df, num_jobs, release_time=0,
                 seed=None, failure_params=None):
        """
        processes_df: DataFrame with columns: Order, Name, Resource, Count, Mean, Std
        failure_params: dict keyed by Resource name -> {"mttf":..., "repair": ...}
        """
        self.env = env
        self.processes = processes_df.sort_values("Order").reset_index(drop=True)
        self.num_jobs = num_jobs
        self.release_time = release_time
        self.event_log = []  # list of dicts: JobID, Process, Resource, Start, End, Duration
        self.job_completion = {}
        self.failure_params = failure_params or {}
        self.rng = random.Random()
        self.active_by_resource = {r: [] for r in self.processes["Resource"].unique()}
        # One resource object per resource type (capacity = Count)
        self.resources = {
            row.Resource: simpy.Resource(env, capacity=int(row.Count))
            for _, row in self.processes.drop_duplicates("Resource").iterrows()
        }
        # Start failure generators for each resource that has failure params
        for res, params in self.failure_params.items():
            if res in self.resources:
                env.process(self._failure_generator(res, params["mttf"], params["repair_time"]))

    def _register_active(self, resource, proc):
        self.active_by_resource[resource].append(proc)

    def _unregister_active(self, resource, proc):
        if proc in self.active_by_resource[resource]:
            self.active_by_resource[resource].remove(proc)

    def _failure_generator(self, resource_name, mttf, repair_time):
        """Randomly generate failures for a resource. When failure occurs,
           an active process on that resource (if any) gets interrupted.
           repair_time is the downtime duration for that failure.
        """
        while True:
            # time to next failure (exponential)
            time_to_fail = self.rng.expovariate(1.0 / mttf)
            yield self.env.timeout(time_to_fail)
            # choose one active process (if any) to interrupt
            active = list(self.active_by_resource[resource_name])
            if not active:
                # nothing to interrupt; downtime will still consume time but not interrupt anyone
                # simulate downtime (machines unavailable) by waiting repair_time
                yield self.env.timeout(repair_time)
                continue

            # interrupt one randomly
            target_proc = self.rng.choice(active)
            try:
                target_proc.interrupt(resource_name)  # pass resource_name in interrupt value
            except RuntimeError:
                # process may have finished between choice and interrupt; ignore
                pass

            # resource is down for repair_time (during which other processes wanting that resource
            # will still be blocked on the resource capacity; we do not explicitly decrease capacity
            # here but the interrupted process will wait for repair_time before resuming)
            yield self.env.timeout(repair_time)

    def job_process(self, job_id, release_time=0):
        """SimPy process for a single job going through all operations sequentially."""
        yield self.env.timeout(release_time)
        job_start = self.env.now

        for _, proc in self.processes.iterrows():
            resource_name = proc["Resource"]
            resource = self.resources[resource_name]
            mean = float(proc["Mean"])
            std = float(proc["Std"])
            duration = sample_normal_trunc(mean, std)

            # Request resource
            with resource.request() as req:
                yield req  # wait for a free server/slot
                # When allocated, register this process as active for potential interruption
                this_proc = self.env.active_process  # generator object (SimPy internals)
                # NOTE: env.active_process exists in SimPy - it's the currently executing event
                # store a reference to the generator-like process so failures can call interrupt()
                # But some SimPy versions might require storing the process returned by env.process(...)
                # As a robust alternative, we wrap the inner timeout in a sub-process below.

                start = self.env.now
                remaining = duration
                interrupted = False
                try:
                    # Register before starting the timed work
                    # We'll create a sub-process to represent the actual work so we can interrupt it.
                    work_proc = self.env.process(self._do_work(resource_name, remaining))
                    self._register_active(resource_name, work_proc)
                    yield work_proc  # this can be interrupted by failure_generator via work_proc.interrupt()
                except simpy.Interrupt as interrupt_info:
                    # A failure occurred and interrupted this work_proc. The interrupt value may contain resource name
                    interrupted = True
                    # work_proc already removed from active_by_resource in _do_work on interrupt
                    # compute how much work remains: the _do_work sets ._remaining attribute for us
                    remaining = getattr(work_proc, "_remaining", 0.0)
                    # wait repair_time for this resource (failure generator already advanced time for repair,
                    # but the interrupted job should be blocked until resource is repaired)
                    # We will try to get the repair_time from failure_params
                    repair_time = self.failure_params.get(resource_name, {}).get("repair_time", 0.0)
                    if repair_time > 0:
                        # simulate waiting for repair to complete
                        yield self.env.timeout(repair_time)
                    # After repair, resume remaining
                    # create another sub-process to finish the remaining work
                    work_proc2 = self.env.process(self._do_work(resource_name, remaining))
                    self._register_active(resource_name, work_proc2)
                    yield work_proc2
                finally:
                    end = self.env.now
                    # Ensure removal from active list if still present
                    # (some code paths remove in _do_work already)
                    # Add event
                    self.event_log.append({
                        "JobID": job_id,
                        "Process": proc["Name"],
                        "Resource": resource_name,
                        "Start": round(start, 4),
                        "End": round(end, 4),
                        "Duration": round(end - start, 4)
                    })
                    # Cleanup any lingering registrations
                    # remove any work_proc / work_proc2 if still present
                    # (list removal is safe inside _unregister_active)
                    # nothing else needed

        # job completed
        self.job_completion[job_id] = round(self.env.now - job_start, 4)

    def _do_work(self, resource_name, duration):
        """Subprocess that performs work for `duration` and supports interruption.
           Stores remaining time on interruption so caller can resume.
        """
        try:
            start = self.env.now
            yield self.env.timeout(duration)
            # finished normally
            return
        except simpy.Interrupt:
            # compute remaining time
            elapsed = self.env.now - start
            remaining = max(0.0, duration - elapsed)
            # store remaining on this process object for caller to read
            self.env.active_process._remaining = remaining  # best-effort (SimPy internals)
            # unregister this proc from active list
            # Note: the interrupting generator didn't pass resource name here so we can't easily
            # know which resource; but caller passed in resource_name and registered the work_proc in active_by_resource
            # We'll attempt to unregister by scanning lists
            for r, lst in self.active_by_resource.items():
                if self.env.active_process in lst:
                    try:
                        lst.remove(self.env.active_process)
                    except ValueError:
                        pass
            # propagate the interrupt to the caller
            raise

    def run(self, until=None):
        # launch jobs
        for i in range(self.num_jobs):
            job_id = f"Block_{i+1}"
            self.env.process(self.job_process(job_id, self.release_time))
        # run env
        self.env.run(until=until)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(layout="wide", page_title="Shipyard Manufacturing Simulator")

st.title("Shipyard Manufacturing Simulation with Resource Failures")

# Sidebar controls
st.sidebar.header("Simulation parameters")
num_jobs = st.sidebar.number_input("Number of jobs (blocks)", min_value=1, max_value=200, value=10, step=1)
seed = st.sidebar.number_input("Random seed", value=42, step=1)
weeks = st.sidebar.number_input("Simulation hours (duration to run)", min_value=1.0, value=500.0, step=1.0)
run_button = st.sidebar.button("Run simulation")

# Default processes table
default_processes = pd.DataFrame([
    {"Order": 1, "Name": "Cutting",    "Resource": "Cutter", "Count": 2, "Mean": 4.0,  "Std": 1.0},
    {"Order": 2, "Name": "Welding",    "Resource": "Welder", "Count": 3, "Mean": 8.0,  "Std": 2.0},
    {"Order": 3, "Name": "Painting",   "Resource": "Painter","Count": 1, "Mean": 6.0,  "Std": 1.5},
    {"Order": 4, "Name": "Outfitting", "Resource": "Fitter", "Count": 2, "Mean": 10.0, "Std": 3.0}
])

st.sidebar.markdown("### Process list (editable)")
# processes_df = st.sidebar.experimental_data_editor(default_processes, num_rows="dynamic")
processes_df = st.sidebar.data_editor(default_processes, num_rows="dynamic")


st.sidebar.markdown("### Failure settings (per resource)")
# Build editable failure settings
failure_defaults = {}
for res in processes_df["Resource"].unique():
    col = st.sidebar.columns([1,1,1])
    mttf = st.sidebar.number_input(f"{res} - MTTF (min)", min_value=1.0, value=100.0, key=f"{res}_mttf")
    repair = st.sidebar.number_input(f"{res} - Repair time (min)", min_value=0.0, value=30.0, key=f"{res}_repair")
    enable = st.sidebar.checkbox(f"Enable failures for {res}", value=(res=="Welder"), key=f"{res}_enable")
    if enable:
        failure_defaults[res] = {"mttf": float(mttf), "repair_time": float(repair)}

# Main area placeholders
placeholder_metrics = st.empty()
placeholder_event = st.empty()
placeholder_util = st.empty()
placeholder_gantt = st.empty()

if run_button:
    # Run the simulation
    st.sidebar.info("Running simulation...")
    np.random.seed(seed)
    random.seed(seed)

    env = simpy.Environment()
    sim = ShipyardSim(env, processes_df, num_jobs=int(num_jobs), release_time=0,
                      seed=seed, failure_params=failure_defaults)
    # run for specified time (or until all jobs done); we'll run until 'weeks' minutes
    sim.run(until=float(weeks))

    # Collect event log DataFrame
    event_df = pd.DataFrame(sim.event_log)
    if event_df.empty:
        st.error("No events were generated. Increase simulation duration or jobs.")
    else:
        # summary metrics
        makespan = event_df["End"].max()
        avg_flow = np.mean(list(sim.job_completion.values())) if sim.job_completion else 0.0
        throughput = len(sim.job_completion)

        metrics_md = f"""
        **Makespan:** {makespan:.2f} minutes  
        **Average flow time:** {avg_flow:.2f} minutes  
        **Throughput (jobs completed):** {throughput}  
        **Total events:** {len(event_df)}
        """
        placeholder_metrics.markdown("### Summary metrics")
        placeholder_metrics.markdown(metrics_md)

        # Event log
        placeholder_event.markdown("### Event log (first 200 rows)")
        placeholder_event.dataframe(event_df.sort_values(["JobID","Start"]).reset_index(drop=True).head(200))

        # Resource utilization: sum durations / (capacity * makespan)
        util_rows = []
        for resname, resource in sim.resources.items():
            busy = event_df[event_df["Resource"]==resname]["Duration"].sum()
            cap = resource.capacity
            util = busy / (cap * makespan) if makespan>0 else 0.0
            util_rows.append({"Resource": resname, "Servers": cap, "BusyTime": round(busy,3), "Utilization": round(util,3)})
        util_df = pd.DataFrame(util_rows)
        placeholder_util.markdown("### Resource utilization")
        placeholder_util.dataframe(util_df)

        # Utilization chart
        fig_util = px.bar(util_df, x="Resource", y="Utilization", text="Utilization", title="Resource Utilization")
        placeholder_util.plotly_chart(fig_util, use_container_width=True)

        # Gantt / timeline: transform event_df to timeline per job-process
        # create a column 'Task' as JobID + Process
        timeline_df = event_df.copy()
        timeline_df["Task"] = timeline_df["JobID"] + " - " + timeline_df["Process"]
        # For a proper Gantt, use start/end, color by Process
        fig_gantt = px.timeline(timeline_df, x_start="Start", x_end="End", y="JobID",
                                color="Process", hover_data=["Resource","Duration"],
                                title="Job timeline (Gantt-like)")
        fig_gantt.update_yaxes(autorange="reversed")  # jobs top-down
        placeholder_gantt.plotly_chart(fig_gantt, use_container_width=True)

        # Allow export
        st.download_button("Download event log (CSV)", event_df.to_csv(index=False), file_name="shipyard_event_log.csv")
        st.download_button("Download utilization (CSV)", util_df.to_csv(index=False), file_name="shipyard_utilization.csv")
        st.success("Simulation finished.")
