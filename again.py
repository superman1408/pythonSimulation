# ashkham_real_time_full.py
import streamlit as st
import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import time
from matplotlib.colors import to_rgba

# ------------------------------
# Real-time Simulation with Callbacks + Mode
# ------------------------------
def run_simulation_real_time(
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
    progress_cb=None,    # percent (0-100)
    stage_cb=None,       # (ship, stage, done_weeks, total_weeks, env_now)
    snapshot_cb=None,    # (records_so_far, in_progress_dict, env_now)
    random_seed=42,
    slow_down=0.04       # seconds to sleep on each weekly update (for UX)
):
    """
    Runs the SimPy simulation and calls callbacks to update UI in realtime.
    - snapshot_cb receives current finished records and in-progress stages to draw live plots.
    - stage_cb receives per-stage weekly progress updates.
    - progress_cb receives overall percent complete (based on completed ships).
    """
    random.seed(random_seed)
    records = []  # finished stage records (append at stage end)
    completed_ships = {"count": 0}
    # in-progress: dict keyed by (ship, stage) -> (start_week, done_weeks, total_weeks)
    in_progress = {}

    env = simpy.Environment()

    # Resources
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
            "Duration (weeks)": end - start
        })

    # Process one stage but step 1 week at a time to provide progress updates
    def process_stage(ship, stage_name, resource, duration_range):
        with resource.request() as req:
            yield req
            start = env.now
            t = random.randint(*duration_range)
            # register in_progress
            in_progress[(ship, stage_name)] = [start, 0, t]  # start_week, done_weeks, total
            for w in range(t):
                # update done_weeks
                in_progress[(ship, stage_name)][1] = w + 1
                # call stage-level callback
                if stage_cb:
                    stage_cb(ship, stage_name, w + 1, t, env.now)
                # call snapshot callback to update live plots occasionally
                if snapshot_cb:
                    snapshot_cb(list(records), dict(in_progress), env.now)
                # call progress callback (percent) will be handled by monitor mainly, but we can also call it here
                if progress_cb:
                    percent = (completed_ships["count"] / num_ships) * 100
                    progress_cb(percent)
                yield env.timeout(1)  # 1 simulated week
                # slow down for UX so the user sees updates (real wall-time)
                if slow_down and slow_down > 0:
                    time.sleep(slow_down)
            # stage finished -> record and remove from in_progress
            end = env.now
            record_event(ship, stage_name, start, end)
            in_progress.pop((ship, stage_name), None)
            # snapshot after stage finishes
            if snapshot_cb:
                snapshot_cb(list(records), dict(in_progress), env.now)

    def ship_process(ship):
        yield env.process(process_stage(ship, "Fabrication", fab, fab_time))
        yield env.process(process_stage(ship, "Assembly", assy, assy_time))
        yield env.process(process_stage(ship, "Erection", erect, erect_time))
        yield env.process(process_stage(ship, "Outfitting", outfit, outfit_time))
        completed_ships["count"] += 1
        # immediate progress update when ship completes
        if progress_cb:
            percent = (completed_ships["count"] / num_ships) * 100
            progress_cb(percent)
        if snapshot_cb:
            snapshot_cb(list(records), dict(in_progress), env.now)

    # monitor progress each week (ensures progress bar updates even if no stage_cb called)
    def progress_monitor(env):
        while True:
            percent = (completed_ships["count"] / num_ships) * 100
            if progress_cb:
                progress_cb(percent)
            if snapshot_cb:
                snapshot_cb(list(records), dict(in_progress), env.now)
            yield env.timeout(1)
            if env.now >= sim_time:
                break

    # launch everything
    def shipyard(env):
        env.process(progress_monitor(env))
        for i in range(num_ships):
            env.process(ship_process(f"Ship-{i+1}"))
            # stagger arrivals by 1-3 weeks for faster realtime demo
            yield env.timeout(random.randint(1, 3))

    env.process(shipyard(env))
    env.run(until=sim_time)

    df_records = pd.DataFrame(records)
    return df_records, completed_ships["count"]

# ------------------------------
# UI: Mode + Inputs
# ------------------------------
st.set_page_config(page_title="ASHKAM - RealTime Simulator", layout="wide")
st.title("🚢 ASHKAM Shipyard Simulator — Real-Time Dashboard")
st.caption("Simulation time unit = weeks. Use Default or User-Defined mode and press ▶️ Run to simulate.")

mode = st.radio("Select Mode", ["Default", "User Defined"], horizontal=True)

if mode == "Default":
    num_ships = 6
    fab_machines = 3
    assy_bays = 2
    erect_docks = 1
    outfit_berths = 1
    fab_time = (8, 12)
    assy_time = (10, 15)
    erect_time = (12, 20)
    outfit_time = (15, 25)
    sim_time = 300
    slow_down = 0.03
else:
    st.subheader("⚙️ User Inputs (Weeks)")
    num_ships = st.number_input("Number of Ships", min_value=1, value=6)
    fab_machines = st.number_input("Fabrication Machines", min_value=1, value=3)
    assy_bays = st.number_input("Assembly Bays", min_value=1, value=2)
    erect_docks = st.number_input("Erection Docks", min_value=1, value=1)
    outfit_berths = st.number_input("Outfitting Berths", min_value=1, value=1)
    sim_time = st.number_input("Simulation Time (weeks)", min_value=50, value=300)
    slow_down = st.slider("UI slow-down (seconds per simulated week)", min_value=0.0, max_value=0.2, value=0.03, step=0.01)
    st.markdown("#### Stage Duration Ranges (weeks)")
    col1, col2 = st.columns(2)
    with col1:
        fab_min = st.number_input("Fabrication Min", 1, 100, 8)
        assy_min = st.number_input("Assembly Min", 1, 100, 10)
        erect_min = st.number_input("Erection Min", 1, 100, 12)
        outfit_min = st.number_input("Outfitting Min", 1, 100, 15)
    with col2:
        fab_max = st.number_input("Fabrication Max", 1, 100, 12)
        assy_max = st.number_input("Assembly Max", 1, 100, 15)
        erect_max = st.number_input("Erection Max", 1, 100, 20)
        outfit_max = st.number_input("Outfitting Max", 1, 100, 25)
    fab_time = (fab_min, fab_max)
    assy_time = (assy_min, assy_max)
    erect_time = (erect_min, erect_max)
    outfit_time = (outfit_min, outfit_max)

# Placeholders for live UI
progress_bar = st.progress(0)
status_text = st.empty()
stage_text = st.empty()
gantt_placeholder = st.empty()
scurve_placeholder = st.empty()
summary_placeholder = st.empty()
report_placeholder = st.empty()

# Helper: build Gantt (partial completion) plot from finished records + in-progress dict + sim_time
def plot_gantt_live(finished_records, in_progress, sim_time):
    # Build a DataFrame combining finished stages and in-progress partial segments
    recs = finished_records.copy()
    # finished_records is list of dicts -> convert to df
    if len(recs) == 0:
        df_finished = pd.DataFrame(columns=["Ship","Stage","Start","End","Duration (weeks)"])
    else:
        df_finished = pd.DataFrame(recs)

    # Build df of in-progress partial segments clipped to sim_time
    inprog_rows = []
    for (ship, stage), (start_week, done_weeks, total_weeks) in in_progress.items():
        # actual duration completed so far = done_weeks
        actual = done_weeks
        inprog_rows.append({
            "Ship": ship,
            "Stage": stage,
            "Start": start_week,
            "End": start_week + actual,
            "Duration (weeks)": actual,
            "InProgress": True,
            "StageTotal": total_weeks
        })
    if len(inprog_rows) > 0:
        df_inprog = pd.DataFrame(inprog_rows)
    else:
        df_inprog = pd.DataFrame(columns=["Ship","Stage","Start","End","Duration (weeks)","InProgress","StageTotal"])

    # mark finished as InProgress=False
    if not df_finished.empty:
        df_finished = df_finished.assign(InProgress=False, StageTotal=df_finished["Duration (weeks)"])
        df_all = pd.concat([df_finished, df_inprog], ignore_index=True, sort=False)
    else:
        df_all = df_inprog.copy()

    # if nothing to plot, show message
    if df_all.empty:
        fig, ax = plt.subplots(figsize=(10,3))
        ax.text(0.5, 0.5, "No stages started yet", ha='center', va='center')
        ax.axis('off')
        return fig

    # Sort ships so order consistent
    ships = sorted(df_all["Ship"].unique(), key=lambda s: int(s.split("-")[1]))
    fig, ax = plt.subplots(figsize=(12, max(3, 0.6*len(ships))))
    stage_colors = {"Fabrication":"skyblue","Assembly":"orange","Erection":"lightgreen","Outfitting":"salmon"}

    for i, ship in enumerate(ships):
        ship_data = df_all[df_all["Ship"] == ship].sort_values(by="Start")
        for _, row in ship_data.iterrows():
            left = row["Start"]
            width = row["Duration (weeks)"]
            color = stage_colors.get(row["Stage"], "gray")
            alpha = 0.9 if not row.get("InProgress", False) else 0.6
            ax.barh(ship, width, left=left, color=to_rgba(color, alpha=alpha), edgecolor="k", height=0.5)
            # label shows stage and % of stage completed if in-progress
            if row.get("InProgress", False):
                pct = (row["Duration (weeks)"] / row["StageTotal"]) * 100 if row["StageTotal"] > 0 else 0
                txt = f"{row['Stage']} ({pct:.0f}%)"
            else:
                txt = f"{row['Stage']} (100%)"
            ax.text(left + width/2, i, txt, ha='center', va='center', fontsize=8, color='black')

    ax.set_xlabel("Simulation Time (weeks)")
    ax.set_ylabel("Ships")
    ax.set_title("Live Gantt (finished segments + in-progress partials)")
    ax.set_xlim(0, sim_time)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

# Helper: build live S-curve (actual partial completion %) and ideal sigmoid
def build_scurve_live(finished_records, in_progress, sim_time):
    # finished_records: list of dicts with Start, End, Duration
    recs = finished_records.copy()
    timeline = list(range(0, sim_time + 1))
    total_stage_count = 0
    # prepare list of all stages (finished and in-progress totals)
    stage_list = []
    for r in recs:
        stage_list.append((r["Start"], r["End"], r["Duration (weeks)"]))
    for (ship, stage), (start_week, done_weeks, total_weeks) in in_progress.items():
        # treat in-progress as having currently done_weeks out of total_weeks
        stage_list.append((start_week, start_week + done_weeks, total_weeks))
    total_stage_count = len(stage_list)
    if total_stage_count == 0:
        # nothing yet
        return pd.DataFrame({"Week": timeline, "Completion (%)": [0]*len(timeline)}), None

    completion_over_time = []
    for week in timeline:
        completed_frac_sum = 0.0
        for (s, e, dur) in stage_list:
            if week < s:
                frac = 0.0
            elif week >= e:
                frac = 1.0
            else:
                # fraction progress at this week for that stage
                frac = (week - s) / dur if dur > 0 else 0
            # clamp 0..1
            frac = max(0.0, min(1.0, frac))
            completed_frac_sum += frac
        pct = (completed_frac_sum / total_stage_count) * 100.0
        completion_over_time.append(pct)
    s_curve_df = pd.DataFrame({"Week": timeline, "Completion (%)": completion_over_time})

    # Ideal sigmoid
    t = np.array(timeline)
    t0 = sim_time / 2.0
    k = 0.12  # steepness; you can expose this to UI if needed
    ideal_completion = 100.0 / (1.0 + np.exp(-k * (t - t0)))
    ideal_df = pd.DataFrame({"Week": timeline, "Completion (%)": ideal_completion})

    return s_curve_df, ideal_df

# Callbacks to update UI
# We'll capture finished records and in_progress via snapshot callback for live plotting
finished_records_global = []  # will be lists of dicts
in_progress_global = {}       # dict from (ship,stage) -> [start, done, total]
env_now_global = 0

def progress_callback(percent):
    progress_bar.progress(min(100, int(percent)))
    status_text.text(f"Overall completion: {percent:.1f}%")

def stage_callback(ship, stage, done_weeks, total_weeks, env_now):
    # update stage text
    stage_text.text(f"{ship} → {stage}: {done_weeks}/{total_weeks} weeks ({done_weeks/total_weeks*100:.1f}%)")
    # small sleep for visible smoothness is handled inside simulation
    # nothing else here; snapshot_cb will redraw plots

def snapshot_callback(finished_records, in_progress, env_now):
    # update globals
    global finished_records_global, in_progress_global, env_now_global
    finished_records_global = finished_records  # list of dicts
    in_progress_global = in_progress
    env_now_global = env_now

    # Live Gantt (draw and show)
    fig = plot_gantt_live(finished_records_global, in_progress_global, sim_time)
    gantt_placeholder.pyplot(fig)

    # live S-curve
    s_df, ideal_df = build_scurve_live(finished_records_global, in_progress_global, sim_time)
    if s_df is not None:
        fig_s, ax = plt.subplots(figsize=(10,4))
        ax.plot(s_df["Week"], s_df["Completion (%)"], label="Actual (partial)", color="tab:blue")
        if ideal_df is not None:
            ax.plot(ideal_df["Week"], ideal_df["Completion (%)"], label="Ideal (sigmoid)", color="orange", linestyle="--")
        ax.set_xlabel("Week")
        ax.set_ylabel("Cumulative Completion (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Live S-Curve (Actual vs Ideal)")
        ax.legend()
        ax.grid(True)
        scurve_placeholder.pyplot(fig_s)

    # live summary snapshot
    if len(finished_records_global) > 0:
        df_tmp = pd.DataFrame(finished_records_global)
        ship_latest = df_tmp.groupby("Ship")["End"].max().reset_index().rename(columns={"End":"Latest End"})
        summary_placeholder.dataframe(ship_latest)

    # live stage report (small snapshot)
    if len(finished_records_global) > 0 or len(in_progress_global) > 0:
        # build a temporary report showing for all ships their stage completion % so far
        rec_list = []
        for r in finished_records_global:
            rec_list.append({
                "Ship": r["Ship"],
                "Stage": r["Stage"],
                "Duration (weeks)": r["Duration (weeks)"],
                "Actual Duration": r["Duration (weeks)"],
                "Stage Completion (%)": 100.0
            })
        for (ship, stage), (start, done, total) in in_progress_global.items():
            rec_list.append({
                "Ship": ship,
                "Stage": stage,
                "Duration (weeks)": total,
                "Actual Duration": done,
                "Stage Completion (%)": (done/total*100.0) if total>0 else 0.0
            })
        df_report = pd.DataFrame(rec_list)
        if not df_report.empty:
            # compute ship completion avg over stages present
            ship_comp = df_report.groupby("Ship")["Stage Completion (%)"].mean().reset_index().rename(columns={"Stage Completion (%)":"Ship Completion (%)"})
            df_report = df_report.merge(ship_comp, on="Ship", how="left")
            report_placeholder.dataframe(df_report.sort_values(["Ship","Stage"]))

# ------------------------------
# Run button
# ------------------------------
if st.button("▶️ Run Real-Time Simulation"):
    # reset placeholders
    gantt_placeholder.empty()
    scurve_placeholder.empty()
    summary_placeholder.empty()
    report_placeholder.empty()
    stage_text.empty()
    status_text.empty()
    progress_bar.progress(0)

    # run sim (blocks the thread but updates UI via callbacks)
    df_records, completed_count = run_simulation_real_time(
        num_ships, fab_machines, assy_bays, erect_docks, outfit_berths,
        fab_time, assy_time, erect_time, outfit_time, sim_time,
        progress_cb=progress_callback,
        stage_cb=stage_callback,
        snapshot_cb=snapshot_callback,
        random_seed=42,
        slow_down=slow_down
    )

    # finalize UI
    progress_bar.empty()
    status_text.text(f"Simulation completed — {completed_count}/{num_ships} ships finished (within {sim_time} weeks)")

    # Final full reports (table + gantt + scurve)
    # Final df_records (may be empty if no stage finished)
    if df_records.empty:
        st.warning("No stages finished during simulation time.")
    else:
        # compute actual durations (clip to sim_time) and stage completion %
        df_records["Actual Duration"] = df_records.apply(lambda row: max(0, min(row["End"], sim_time) - row["Start"]), axis=1)
        df_records["Stage Completion (%)"] = (df_records["Actual Duration"] / df_records["Duration (weeks)"]) * 100
        # Ship completion
        ship_completion = df_records.groupby("Ship")["Stage Completion (%)"].mean().reset_index().rename(columns={"Stage Completion (%)":"Ship Completion (%)"})
        summary_report = df_records.merge(ship_completion, on="Ship", how="left")
        summary_report = summary_report[["Ship","Stage","Duration (weeks)","Actual Duration","Stage Completion (%)","Ship Completion (%)"]]
        st.subheader("📄 Final Shipwise Stage Completion Report")
        st.dataframe(summary_report.style.format({
            "Duration (weeks)":"{:.1f}",
            "Actual Duration":"{:.1f}",
            "Stage Completion (%)":"{:.1f}%",
            "Ship Completion (%)":"{:.1f}%"
        }))

        # final gantt
        final_fig = plot_gantt_live(df_records.to_dict("records"), {}, sim_time)
        st.subheader("🚀 Final Gantt (finished stages)")
        st.pyplot(final_fig)

        # final S-curve
        s_df, ideal_df = build_scurve_live(df_records.to_dict("records"), {}, sim_time)
        fig_final, axf = plt.subplots(figsize=(10,4))
        axf.plot(s_df["Week"], s_df["Completion (%)"], label="Actual (partial)", color="tab:blue")
        if ideal_df is not None:
            axf.plot(ideal_df["Week"], ideal_df["Completion (%)"], label="Ideal (sigmoid)", color="orange", linestyle="--")
        axf.set_xlabel("Week")
        axf.set_ylabel("Cumulative Completion (%)")
        axf.set_ylim(0, 100)
        axf.set_title("Final S-Curve")
        axf.legend()
        axf.grid(True)
        st.pyplot(fig_final)
