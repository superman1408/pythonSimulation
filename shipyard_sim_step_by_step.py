# shipyard_sim_step_by_step.py
import simpy
import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import time

st.set_page_config(layout="wide", page_title="Shipyard Step-by-Step Simulator")

# -----------------------
# Utility sampling
# -----------------------
def sample_normal_trunc(mean, std, rng=np.random):
    t = rng.normal(mean, std)
    return max(0.01, float(t))

# -----------------------
# Default process table
# -----------------------
DEFAULT_PROCESSES = pd.DataFrame([
    {"Order": 1, "Name": "Cutting",    "Resource": "Cutter",  "Count": 2, "Mean": 4.0,  "Std": 1.0},
    {"Order": 2, "Name": "Welding",    "Resource": "Welder",  "Count": 3, "Mean": 8.0,  "Std": 2.0},
    {"Order": 3, "Name": "Painting",   "Resource": "Painter", "Count": 1, "Mean": 6.0,  "Std": 1.5},
    {"Order": 4, "Name": "Outfitting", "Resource": "Fitter",  "Count": 2, "Mean": 10.0, "Std": 3.0}
])

# -----------------------
# Simulation class
# -----------------------
class ShipyardSim:
    """
    SimPy shipyard simulation.
    - Logs start and end times of each process step per job.
    - Supports "resource-level" failures that interrupt active work (chosen randomly).
    - For step-by-step visualization we expose event_log and current active segments.
    """
    def __init__(self, env, processes_df, num_jobs=10, release_time=0, seed=42, failure_params=None):
        self.env = env
        self.processes = processes_df.sort_values("Order").reset_index(drop=True)
        self.num_jobs = int(num_jobs)
        self.release_time = release_time
        self.event_log = []      # finished segments: dicts with JobID, Process, Resource, Start, End, Duration
        self.active_segments = []  # running segments (job segments currently in progress)
        self.job_completion = {}
        self.rng = random.Random(seed)
        self.nprng = np.random.RandomState(seed)
        self.failure_params = failure_params or {}  # {resource: {"mttf":.., "repair_time":.., "enabled":True}}

        # Create SimPy resources (capacity = Count)
        # We use Resource (capacity = count) and track running segments ourselves to support interruption
        self.resources = {}
        for _, row in self.processes.drop_duplicates("Resource").iterrows():
            self.resources[row["Resource"]] = simpy.Resource(self.env, capacity=int(row["Count"]))

        # Start failure generators for each resource where enabled
        for res, params in self.failure_params.items():
            if params.get("enabled", False) and res in self.resources:
                env.process(self._failure_generator(res, params["mttf"], params["repair_time"]))

        # Launch job processes
        for i in range(self.num_jobs):
            job_id = f"Block_{i+1}"
            env.process(self.job_process(job_id, self.release_time))

    # -----------------------
    # Worker subprocess that performs the work and supports interruption
    # -----------------------
    def _perform_work(self, duration):
        """
        This subprocess yields a timeout for `duration`. If interrupted,
        it stores the remaining time on the process object as attribute '_remaining'
        and propagates the interrupt to the caller.
        """
        start = self.env.now
        try:
            yield self.env.timeout(duration)
            return 0.0  # no remaining
        except simpy.Interrupt:
            elapsed = self.env.now - start
            remaining = max(0.0, duration - elapsed)
            # store remaining on the active process object for the caller to inspect
            # env.active_process refers to the Process object currently running this generator
            try:
                self.env.active_process._remaining = remaining
            except Exception:
                # best-effort; if not available, caller will handle
                pass
            raise

    # -----------------------
    # Failure generator for a resource (interrupts one active segment randomly when failure occurs)
    # -----------------------
    def _failure_generator(self, resource_name, mttf, repair_time):
        """Randomly generate failures for resource_name. On failure, interrupt one active segment if any."""
        while True:
            # time to next failure (exponential)
            ttf = self.rng.expovariate(1.0 / float(mttf))
            yield self.env.timeout(ttf)
            # Choose a currently active segment on that resource (if any)
            active_on_res = [seg for seg in self.active_segments if seg["Resource"] == resource_name]
            if active_on_res:
                target = self.rng.choice(active_on_res)
                # interrupt its process
                try:
                    target["process"].interrupt()  # process is the SimPy Process returned by env.process(...)
                except Exception:
                    pass
            # We model repair by advancing env time (repairs occupy simulated time)
            # However we don't block other resources here; we simply wait repair_time
            yield self.env.timeout(repair_time)

    # -----------------------
    # Job process moving sequentially through operations
    # -----------------------
    def job_process(self, job_id, release_time=0):
        yield self.env.timeout(release_time)
        job_start = self.env.now

        for _, proc in self.processes.iterrows():
            resource_name = proc["Resource"]
            resource = self.resources[resource_name]
            mean = float(proc["Mean"])
            std = float(proc["Std"])
            duration = sample_normal_trunc(mean, std, rng=self.nprng)

            # Acquire resource (queue if needed)
            with resource.request() as req:
                yield req
                start_ts = self.env.now

                # Create the work subprocess so it can be interrupted
                work_proc = self.env.process(self._perform_work(duration))
                # Register active segment (so failure_generator can find it)
                seg = {
                    "JobID": job_id,
                    "Process": proc["Name"],
                    "Resource": resource_name,
                    "Start": start_ts,
                    "process": work_proc
                }
                self.active_segments.append(seg)

                try:
                    # Wait until work_proc completes (or is interrupted)
                    yield work_proc
                    # Completed normally: record finished segment
                    end_ts = self.env.now
                    self.event_log.append({
                        "JobID": job_id,
                        "Process": proc["Name"],
                        "Resource": resource_name,
                        "Start": round(start_ts, 4),
                        "End": round(end_ts, 4),
                        "Duration": round(end_ts - start_ts, 4)
                    })
                    # remove from active segments
                    try:
                        self.active_segments.remove(seg)
                    except ValueError:
                        pass

                except simpy.Interrupt:
                    # Work was interrupted by a failure
                    # remaining time stored on work_proc (if available), else infer 0
                    remaining = getattr(work_proc, "_remaining", None)
                    if remaining is None:
                        remaining = max(0.0, duration - (self.env.now - start_ts))

                    # Record the partial segment that ran until interruption
                    end_ts = self.env.now
                    self.event_log.append({
                        "JobID": job_id,
                        "Process": proc["Name"] + " (partial)",
                        "Resource": resource_name,
                        "Start": round(start_ts, 4),
                        "End": round(end_ts, 4),
                        "Duration": round(end_ts - start_ts, 4)
                    })
                    try:
                        # Remove segment from active list
                        self.active_segments.remove(seg)
                    except ValueError:
                        pass

                    # Synchronize with repair time if configured for this resource
                    repair_time = self.failure_params.get(resource_name, {}).get("repair_time", 0.0)
                    # Wait repair_time (this simulates machine being repaired for this job)
                    if repair_time > 0:
                        yield self.env.timeout(repair_time)

                    # After repair, resume remaining work as a new sub-segment
                    resume_start = self.env.now
                    work_proc2 = self.env.process(self._perform_work(remaining))
                    seg2 = {
                        "JobID": job_id,
                        "Process": proc["Name"],
                        "Resource": resource_name,
                        "Start": resume_start,
                        "process": work_proc2
                    }
                    self.active_segments.append(seg2)
                    try:
                        yield work_proc2
                        resume_end = self.env.now
                        self.event_log.append({
                            "JobID": job_id,
                            "Process": proc["Name"] + " (resumed)",
                            "Resource": resource_name,
                            "Start": round(resume_start, 4),
                            "End": round(resume_end, 4),
                            "Duration": round(resume_end - resume_start, 4)
                        })
                    except simpy.Interrupt:
                        # If interrupted again, we let the loop handle further interruptions
                        # We re-raise here to allow the outer exception handling to capture it in next iteration
                        raise
                    finally:
                        try:
                            self.active_segments.remove(seg2)
                        except Exception:
                            pass

        # job completed
        self.job_completion[job_id] = round(self.env.now - job_start, 4)

# -----------------------
# Streamlit UI layout
# -----------------------
st.title("Shipyard Simulation — Step-by-step (SimPy)")

# Sidebar controls
st.sidebar.header("Simulation controls")
num_jobs = st.sidebar.number_input("Number of jobs (blocks)", min_value=1, max_value=200, value=10, step=1)
seed = st.sidebar.number_input("Random seed", value=42, step=1)
sim_minutes = st.sidebar.number_input("Total simulated minutes", min_value=10.0, max_value=1e6, value=500.0, step=10.0)
step_size = st.sidebar.number_input("Sim minutes per UI update (step)", min_value=1.0, max_value=240.0, value=5.0, step=1.0)
delay = st.sidebar.number_input("Real seconds delay per update", min_value=0.0, max_value=5.0, value=0.35, step=0.05)

st.sidebar.markdown("### Processes (edit and press Run)")
# Use data_editor if available, otherwise show dataframe and let user edit via simpler inputs
try:
    processes_df = st.sidebar.data_editor(DEFAULT_PROCESSES, num_rows="dynamic")
except Exception:
    # fallback to simple editing: show and use defaults
    st.sidebar.write("data_editor not available — using defaults")
    processes_df = DEFAULT_PROCESSES.copy()

# Failure settings per resource
st.sidebar.markdown("### Failure settings per resource")
failure_params = {}
for res in processes_df["Resource"].unique():
    col1, col2, col3 = st.sidebar.columns([1,1,1])
    mttf = col1.number_input(f"{res} MTTF (min)", min_value=1.0, value=100.0, key=f"{res}_mttf")
    repair = col2.number_input(f"{res} Repair (min)", min_value=0.0, value=30.0, key=f"{res}_repair")
    enabled = col3.checkbox(f"Enable", value=(res=="Welder"), key=f"{res}_enabled")
    failure_params[res] = {"mttf": float(mttf), "repair_time": float(repair), "enabled": bool(enabled)}

# Run button
run_button = st.sidebar.button("Run step-by-step")

# Placeholders for dynamic UI
kpi_ph = st.empty()
gantt_ph = st.empty()
util_ph = st.empty()
log_ph = st.empty()
progress_ph = st.empty()

if run_button:
    # Seed RNGs
    np.random.seed(int(seed))
    random.seed(int(seed))

    # Create environment and simulation
    env = simpy.Environment()
    sim = ShipyardSim(env, processes_df, num_jobs=int(num_jobs), release_time=0, seed=int(seed),
                      failure_params=failure_params)

    # Stepwise run loop
    sim_end = float(sim_minutes)
    step = float(step_size)
    real_delay = float(delay)

    # We'll keep updating the UI each time we advance the sim by `step` minutes
    last_update_time = -1.0

    # Run until sim_end or until all jobs complete (whichever comes first)
    while env.now < sim_end:
        target = min(env.now + step, sim_end)
        # Run until target sim time
        env.run(until=target)

        # Build event dataframe (completed segments)
        event_df = pd.DataFrame(sim.event_log)
        # Build active segments snapshot (for visualization we show them as running up to env.now)
        active_snapshot = []
        for seg in sim.active_segments:
            # seg contains 'Start' and 'process'; add a visible partial segment from start->now
            active_snapshot.append({
                "JobID": seg["JobID"],
                "Process": seg["Process"],
                "Resource": seg["Resource"],
                "Start": round(seg["Start"], 4),
                "End": round(env.now, 4),
                "Duration": round(env.now - seg["Start"], 4),
                "Status": "Running"
            })

        # Combine finished segments and active snapshot for Gantt view
        gantt_df = pd.DataFrame(event_df) if not event_df.empty else pd.DataFrame(columns=["JobID","Process","Resource","Start","End","Duration"])
        if not gantt_df.empty:
            gantt_df = gantt_df[["JobID","Process","Resource","Start","End","Duration"]].copy()
            gantt_df["Status"] = "Completed"
        else:
            gantt_df = pd.DataFrame(columns=["JobID","Process","Resource","Start","End","Duration","Status"])

        if active_snapshot:
            gantt_df = pd.concat([gantt_df, pd.DataFrame(active_snapshot)], ignore_index=True, sort=False)

        # KPIs
        makespan = None
        if not event_df.empty:
            makespan = event_df["End"].max()
        else:
            makespan = env.now

        avg_flow = None
        if sim.job_completion:
            avg_flow = np.mean(list(sim.job_completion.values()))
        else:
            avg_flow = np.nan

        throughput = len(sim.job_completion)

        # Resource utilization: sum finished durations + running durations divided by (capacity * horizon)
        util_rows = []
        for resname in sim.resources.keys():
            finished_busy = 0.0
            if not event_df.empty:
                finished_busy = event_df[event_df["Resource"] == resname]["Duration"].sum()
            running_busy = sum([s["Duration"] for s in active_snapshot if s["Resource"] == resname])
            busy = finished_busy + running_busy
            cap = sim.resources[resname].capacity
            horizon = max(1.0, makespan)  # avoid div by zero
            util = busy / (cap * horizon)
            util_rows.append({"Resource": resname, "Servers": cap, "BusyTime": round(busy,3), "Utilization": round(util,3)})
        util_df = pd.DataFrame(util_rows)

        # Render KPIs
        kpi_ph.markdown("### Summary (live)")
        kpi_ph.write({
            "Sim time (now)": round(env.now, 3),
            "Makespan (so far)": round(makespan,3),
            "Average flow time (completed jobs)": round(avg_flow,3) if np.isfinite(avg_flow) else "n/a",
            "Throughput (completed jobs)": throughput,
            "Total recorded segments": len(event_df)
        })

        # Render utilization
        util_ph.markdown("### Resource utilization (live)")
        util_ph.dataframe(util_df)
        fig_util = px.bar(util_df, x="Resource", y="Utilization", text="Utilization", title="Resource Utilization (live)")
        # util_ph.plotly_chart(fig_util, use_container_width=True)
        util_ph.plotly_chart(fig_util, use_container_width=True, key=f"util_{env.now}")
        
        # Before (causes duplicate IDs)
# util_ph.plotly_chart(fig_util, use_container_width=True)
# gantt_ph.plotly_chart(fig_gantt, use_container_width=True)

# After (add unique keys per iteration)
# util_ph.plotly_chart(fig_util, use_container_width=True, key=f"util_{env.now}")
# gantt_ph.plotly_chart(fig_gantt, use_container_width=True, key=f"gantt_{env.now}")


        # Render Gantt-like timeline
        if not gantt_df.empty:
            gantt_ph.markdown("### Gantt-like timeline (completed + running)")
            fig_gantt = px.timeline(gantt_df, x_start="Start", x_end="End", y="JobID", color="Process",
                                    hover_data=["Resource","Duration","Status"], title="Job timeline (live)")
            fig_gantt.update_yaxes(autorange="reversed")
            # gantt_ph.plotly_chart(fig_gantt, use_container_width=True)
            gantt_ph.plotly_chart(fig_gantt, use_container_width=True, key=f"gantt_{env.now}")

        # Render event log (completed segments)
        log_ph.markdown("### Event log (completed segments)")
        if not event_df.empty:
            log_ph.dataframe(event_df.sort_values(["JobID","Start"]).reset_index(drop=True).tail(500))
        else:
            log_ph.write("No completed segments yet.")

        # Progress bar
        frac = env.now / sim_end
        if frac > 1.0:
            frac = 1.0
        progress_ph.progress(min(1.0, frac))

        # small sleep to give the UI time to update and to create smooth animation
        if real_delay > 0:
            time.sleep(real_delay)

        # if all jobs finished, we can break early
        if throughput >= int(num_jobs):
            st.success("All jobs completed — simulation finished early.")
            break

    # Final message and outputs
    st.balloons()
    st.success(f"Simulation run complete. Sim time reached: {round(env.now,3)} minutes. Completed jobs: {len(sim.job_completion)}")

    # Final exports
    final_event_df = pd.DataFrame(sim.event_log)
    if not final_event_df.empty:
        st.download_button("Download final event log (CSV)", final_event_df.to_csv(index=False), file_name="shipyard_event_log.csv")
    st.download_button("Download utilization snapshot (CSV)", util_df.to_csv(index=False), file_name="shipyard_utilization.csv")


# Before (causes duplicate IDs)
# util_ph.plotly_chart(fig_util, use_container_width=True)
# gantt_ph.plotly_chart(fig_gantt, use_container_width=True)

# After (add unique keys per iteration)
# util_ph.plotly_chart(fig_util, use_container_width=True, key=f"util_{env.now}")
# gantt_ph.plotly_chart(fig_gantt, use_container_width=True, key=f"gantt_{env.now}")
