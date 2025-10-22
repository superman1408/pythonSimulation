# ashkham_realtime_full_with_per_ship.py
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
# Minimum simulation time estimator (simple Monte Carlo)
# ------------------------------
def estimate_min_time(num_ships, fab_machines, assy_bays, erect_docks, outfit_berths,
                      fab_time, assy_time, erect_time, outfit_time, trials=200):
    samples = []
    for _ in range(trials):
        # approximate pipeline: assume each stage can be parallelized by its capacity
        total = 0.0
        for _ in range(num_ships):
            # sample stage durations
            f = random.randint(*fab_time)
            a = random.randint(*assy_time)
            e = random.randint(*erect_time)
            o = random.randint(*outfit_time)
            # divide by capacity (rough parallelism estimate)
            total += f / max(1, fab_machines) + a / max(1, assy_bays) + e / max(1, erect_docks) + o / max(1, outfit_berths)
        samples.append(total)
    # return median to be robust
    return int(np.ceil(np.median(samples)))

# ------------------------------
# Real-time simulation function (with callbacks)
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
    progress_cb=None,    # (percent)
    stage_cb=None,       # (ship, stage, done_weeks, total_weeks, env_now)
    snapshot_cb=None,    # (finished_records, in_progress, env_now)
    per_ship_progress_cb=None,  # (ship, percent)
    random_seed=42,
    slow_down=0.03
):
    random.seed(random_seed)
    records = []
    completed_ships = {"count": 0}
    in_progress = {}  # (ship,stage) -> [start_week, done_weeks, total_weeks]
    ship_stage_totals = {}  # ship -> total weeks of all stages (for per-ship %)

    env = simpy.Environment()

    # resources
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

    # compute per-ship total expected weeks (sampled once per ship for progress denominator)
    # We'll sample stage durations now for totals (this doesn't affect actual sim durations,
    # it's only used to compute per-ship total weeks for progress %). To be consistent we compute totals
    # per ship on the fly when the ship starts its first stage.
    def ensure_ship_totals(ship):
        if ship not in ship_stage_totals:
            # sample one possible total for this ship
            s = random.randint(*fab_time) + random.randint(*assy_time) + random.randint(*erect_time) + random.randint(*outfit_time)
            ship_stage_totals[ship] = max(1, s)

    def update_ship_progress(ship):
        # compute percent completion for ship as sum of completed + in-progress weeks / ship_stage_totals[ship]
        total_done = 0.0
        # finished
        for r in records:
            if r["Ship"] == ship:
                # clip to sim_time
                done = max(0, min(r["End"], sim_time) - r["Start"])
                total_done += done
        # in-progress
        for (s, stg), (start, done_weeks, total_weeks) in in_progress.items():
            if s == ship:
                total_done += done_weeks
        denom = ship_stage_totals.get(ship, 1)
        percent = min(100.0, (total_done / denom) * 100.0)
        if per_ship_progress_cb:
            per_ship_progress_cb(ship, percent)
        return percent

    # process stage stepping 1 week at a time
    def process_stage(ship, stage_name, resource, duration_range):
        with resource.request() as req:
            yield req
            ensure_ship_totals(ship)
            start = env.now
            t = random.randint(*duration_range)
            in_progress[(ship, stage_name)] = [start, 0, t]
            for w in range(t):
                # update done weeks
                in_progress[(ship, stage_name)][1] = w + 1
                # callbacks
                if stage_cb:
                    stage_cb(ship, stage_name, w + 1, t, env.now)
                # update per-ship progress
                update_ship_progress(ship)
                if snapshot_cb:
                    snapshot_cb(list(records), dict(in_progress), env.now)
                if progress_cb:
                    progress_cb((completed_ships["count"] / num_ships) * 100.0)
                # advance 1 week
                yield env.timeout(1)
                # wall-clock slow down to make updates visible
                if slow_down and slow_down > 0:
                    time.sleep(slow_down)
            # finished this stage
            end = env.now
            record_event(ship, stage_name, start, end)
            in_progress.pop((ship, stage_name), None)
            # update ship progress after finishing stage
            update_ship_progress(ship)
            if snapshot_cb:
                snapshot_cb(list(records), dict(in_progress), env.now)

    def ship_process(ship):
        yield env.process(process_stage(ship, "Fabrication", fab, fab_time))
        yield env.process(process_stage(ship, "Assembly", assy, assy_time))
        yield env.process(process_stage(ship, "Erection", erect, erect_time))
        yield env.process(process_stage(ship, "Outfitting", outfit, outfit_time))
        completed_ships["count"] += 1
        # final update
        if progress_cb:
            progress_cb((completed_ships["count"] / num_ships) * 100.0)
        # update all ship progress bars after ship completes
        for s in ship_stage_totals.keys():
            update_ship_progress(s)
        if snapshot_cb:
            snapshot_cb(list(records), dict(in_progress), env.now)

    # monitor and snapshot weekly
    def progress_monitor(env):
        while True:
            if progress_cb:
                progress_cb((completed_ships["count"] / num_ships) * 100.0)
            if snapshot_cb:
                snapshot_cb(list(records), dict(in_progress), env.now)
            yield env.timeout(1)
            if env.now >= sim_time:
                break

    # launch
    def shipyard(env):
        env.process(progress_monitor(env))
        for i in range(num_ships):
            env.process(ship_process(f"Ship-{i+1}"))
            yield env.timeout(random.randint(1, 3))

    env.process(shipyard(env))
    env.run(until=sim_time)

    df_records = pd.DataFrame(records)
    # final update of per-ship progress
    for s in ship_stage_totals.keys():
        update_ship_progress(s)
    return df_records, completed_ships["count"]

# ------------------------------
# UI: Mode + Inputs + Estimator
# ------------------------------
st.set_page_config(page_title="ASHKAM - RealTime Simulator (Per-Ship Bars)", layout="wide")
st.title("🚢 ASHKAM Shipyard Simulator — Real-Time Dashboard")
st.caption("Simulation time unit = weeks. Default/User-defined mode available.")

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

# Show estimated minimum time before running
est_min = estimate_min_time(num_ships, fab_machines, assy_bays, erect_docks, outfit_berths,
                            fab_time, assy_time, erect_time, outfit_time, trials=200)
st.info(f"Estimated minimum time to complete all {num_ships} ships (approx): **{est_min} weeks** (median of samples)")

# placeholders & widgets
progress_bar = st.progress(0)
status_text = st.empty()
stage_text = st.empty()
gantt_placeholder = st.empty()
scurve_placeholder = st.empty()
summary_placeholder = st.empty()
report_placeholder = st.empty()

# prepare per-ship progress UI in both sidebar and main
sidebar_placeholders = {}
main_placeholders = {}

st.sidebar.header("Per-Ship Progress (Sidebar)")
for i in range(1, num_ships + 1):
    ph = st.sidebar.empty()
    sidebar_placeholders[f"Ship-{i}"] = ph
st.header("Per-Ship Progress (Main Area)")
for i in range(1, num_ships + 1):
    ph = st.empty()
    main_placeholders[f"Ship-{i}"] = ph

# helper functions to render per-ship bars
def update_per_ship_bar(placeholders_dict, ship, percent):
    # show a simple text + st.progress for each ship
    ph = placeholders_dict.get(ship)
    if ph:
        # use a small layout: text + progress
        ph.markdown(f"**{ship}** — {percent:.1f}%")
        ph.progress(int(percent))

# snapshot callback used by sim to update live visuals
finished_records_global = []
in_progress_global = {}
env_now_global = 0

def snapshot_callback(finished_records, in_progress, env_now):
    global finished_records_global, in_progress_global, env_now_global
    finished_records_global = finished_records
    in_progress_global = in_progress
    env_now_global = env_now

    # update gantt and scurve snapshots
    fig = plot_gantt_live(finished_records_global, in_progress_global, sim_time)
    gantt_placeholder.pyplot(fig)

    s_df, ideal_df = build_scurve_live(finished_records_global, in_progress_global, sim_time)
    if s_df is not None:
        fig_s, ax = plt.subplots(figsize=(10,4))
        ax.plot(s_df["Week"], s_df["Completion (%)"], label="Actual (partial)", color="tab:blue")
        if ideal_df is not None:
            ax.plot(ideal_df["Week"], ideal_df["Completion (%)"], label="Ideal (sigmoid)", color="orange", linestyle="--")
        ax.set_xlabel("Week"); ax.set_ylabel("Cumulative Completion (%)"); ax.set_ylim(0,100)
        ax.set_title("Live S-Curve (Actual vs Ideal)"); ax.legend(); ax.grid(True)
        scurve_placeholder.pyplot(fig_s)

    # small summary snapshot
    if len(finished_records_global) > 0:
        df_tmp = pd.DataFrame(finished_records_global)
        ship_latest = df_tmp.groupby("Ship")["End"].max().reset_index().rename(columns={"End":"Latest End"})
        summary_placeholder.dataframe(ship_latest)

    # stage report snapshot
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
        ship_comp = df_report.groupby("Ship")["Stage Completion (%)"].mean().reset_index().rename(columns={"Stage Completion (%)":"Ship Completion (%)"})
        df_report = df_report.merge(ship_comp, on="Ship", how="left")
        report_placeholder.dataframe(df_report.sort_values(["Ship","Stage"]))

# progress callback
def progress_callback(percent):
    progress_bar.progress(min(100, int(percent)))
    status_text.text(f"Overall completion: {percent:.1f}%")

# stage callback updates per-ship bars (both sidebar and main)
def stage_callback(ship, stage, done_weeks, total_weeks, env_now):
    # compute ship percent roughly: using finished_records_global and in_progress_global we can compute
    # but simpler: compute percent as (sum done weeks for that ship) / (sum sampled totals) - handled in main sim func via per-ship cb
    # here display stage status text
    stage_text.text(f"{ship} → {stage}: {done_weeks}/{total_weeks} weeks ({done_weeks/total_weeks*100:.1f}%)")
    # also update per-ship bars using snapshot data when available
    # compute percent from finished_records_global + in_progress_global (same logic as in earlier helper)
    # Build per-ship percent map:
    per_ship_done = {}
    per_ship_total = {}
    for r in finished_records_global:
        s = r["Ship"]
        per_ship_done[s] = per_ship_done.get(s, 0) + max(0, min(r["End"], sim_time) - r["Start"])
        per_ship_total[s] = per_ship_total.get(s, 0) + r["Duration (weeks)"]
    for (s, stg), (start, done, total) in in_progress_global.items():
        per_ship_done[s] = per_ship_done.get(s, 0) + done
        per_ship_total[s] = per_ship_total.get(s, 0) + total
    # update UI bars
    for i in range(1, num_ships + 1):
        ship_name = f"Ship-{i}"
        done = per_ship_done.get(ship_name, 0)
        total = per_ship_total.get(ship_name, None)
        if total is None or total == 0:
            pct = 0.0
        else:
            pct = min(100.0, (done / total) * 100.0)
        update_per_ship_bar(sidebar_placeholders, ship_name, pct)
        update_per_ship_bar(main_placeholders, ship_name, pct)

# helpers reused from earlier code: plot_gantt_live and build_scurve_live
def plot_gantt_live(finished_records, in_progress, sim_time):
    recs = finished_records.copy()
    if len(recs) == 0:
        df_finished = pd.DataFrame(columns=["Ship","Stage","Start","End","Duration (weeks)"])
    else:
        df_finished = pd.DataFrame(recs)
    inprog_rows = []
    for (ship, stage), (start_week, done_weeks, total_weeks) in in_progress.items():
        inprog_rows.append({
            "Ship": ship,
            "Stage": stage,
            "Start": start_week,
            "End": start_week + done_weeks,
            "Duration (weeks)": done_weeks,
            "InProgress": True,
            "StageTotal": total_weeks
        })
    if len(inprog_rows) > 0:
        df_inprog = pd.DataFrame(inprog_rows)
    else:
        df_inprog = pd.DataFrame(columns=["Ship","Stage","Start","End","Duration (weeks)","InProgress","StageTotal"])
    if not df_finished.empty:
        df_finished = df_finished.assign(InProgress=False, StageTotal=df_finished["Duration (weeks)"])
        df_all = pd.concat([df_finished, df_inprog], ignore_index=True, sort=False)
    else:
        df_all = df_inprog.copy()
    if df_all.empty:
        fig, ax = plt.subplots(figsize=(10,3))
        ax.text(0.5, 0.5, "No stages started yet", ha='center', va='center')
        ax.axis('off')
        return fig
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

def build_scurve_live(finished_records, in_progress, sim_time):
    recs = finished_records.copy()
    timeline = list(range(0, sim_time + 1))
    stage_list = []
    for r in recs:
        stage_list.append((r["Start"], r["End"], r["Duration (weeks)"]))
    for (ship, stage), (start_week, done_weeks, total_weeks) in in_progress.items():
        stage_list.append((start_week, start_week + done_weeks, total_weeks))
    total_stage_count = len(stage_list)
    if total_stage_count == 0:
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
                frac = (week - s) / dur if dur > 0 else 0
            frac = max(0.0, min(1.0, frac))
            completed_frac_sum += frac
        pct = (completed_frac_sum / total_stage_count) * 100.0
        completion_over_time.append(pct)
    s_curve_df = pd.DataFrame({"Week": timeline, "Completion (%)": completion_over_time})
    t = np.array(timeline); t0 = sim_time / 2.0; k = 0.12
    ideal_completion = 100.0 / (1.0 + np.exp(-k * (t - t0)))
    ideal_df = pd.DataFrame({"Week": timeline, "Completion (%)": ideal_completion})
    return s_curve_df, ideal_df

# Run button
if st.button("▶️ Run Real-Time Simulation"):
    # clear placeholders
    gantt_placeholder.empty(); scurve_placeholder.empty(); summary_placeholder.empty(); report_placeholder.empty()
    stage_text.empty(); status_text.empty(); progress_bar.progress(0)

    # run simulation (blocks but updates UI via callbacks)
    df_records, completed_count = run_simulation_real_time(
        num_ships, fab_machines, assy_bays, erect_docks, outfit_berths,
        fab_time, assy_time, erect_time, outfit_time, sim_time,
        progress_cb=progress_callback,
        stage_cb=stage_callback,
        snapshot_cb=snapshot_callback,
        per_ship_progress_cb=None,  # per-ship updates done inside stage_callback/snapshot
        random_seed=42,
        slow_down=slow_down
    )

    # finalize
    progress_bar.empty()
    status_text.text(f"Simulation completed — {completed_count}/{num_ships} ships finished (within {sim_time} weeks)")

    # Final summary and outputs (as previous)
    if df_records.empty:
        st.warning("No stages finished during simulation time.")
    else:
        df_records["Actual Duration"] = df_records.apply(lambda row: max(0, min(row["End"], sim_time) - row["Start"]), axis=1)
        df_records["Stage Completion (%)"] = (df_records["Actual Duration"] / df_records["Duration (weeks)"]) * 100
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

        final_fig = plot_gantt_live(df_records.to_dict("records"), {}, sim_time)
        st.subheader("🚀 Final Gantt (finished stages)")
        st.pyplot(final_fig)

        s_df, ideal_df = build_scurve_live(df_records.to_dict("records"), {}, sim_time)
        fig_final, axf = plt.subplots(figsize=(10,4))
        axf.plot(s_df["Week"], s_df["Completion (%)"], label="Actual (partial)", color="tab:blue")
        if ideal_df is not None:
            axf.plot(ideal_df["Week"], ideal_df["Completion (%)"], label="Ideal (sigmoid)", color="orange", linestyle="--")
        axf.set_xlabel("Week"); axf.set_ylabel("Cumulative Completion (%)"); axf.set_ylim(0,100)
        axf.set_title("Final S-Curve"); axf.legend(); axf.grid(True)
        st.pyplot(fig_final)

    # final per-ship bars update (ensure all 100% where finished)
    if not df_records.empty:
        per_ship_done = {}
        per_ship_total = {}
        for r in df_records.to_dict("records"):
            per_ship_done[r["Ship"]] = per_ship_done.get(r["Ship"], 0) + max(0, min(r["End"], sim_time) - r["Start"])
            per_ship_total[r["Ship"]] = per_ship_total.get(r["Ship"], 0) + r["Duration (weeks)"]
        for i in range(1, num_ships + 1):
            ship_name = f"Ship-{i}"
            done = per_ship_done.get(ship_name, 0)
            total = per_ship_total.get(ship_name, None)
            pct = min(100.0, (done / total) * 100.0) if total and total > 0 else 0.0
            update_per_ship_bar(sidebar_placeholders, ship_name, pct)
            update_per_ship_bar(main_placeholders, ship_name, pct)
