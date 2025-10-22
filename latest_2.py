# ashkam_realtime_final.py
import streamlit as st
import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.colors import to_rgba
import plotly.express as px

# ------------------------------
# Utility: Minimum time estimator (Monte Carlo, median)
# ------------------------------
def estimate_min_time(num_ships, fab_machines, assy_bays, erect_docks, outfit_berths,
                      fab_time, assy_time, erect_time, outfit_time, trials=200):
    samples = []
    for _ in range(trials):
        total = 0.0
        for _ in range(num_ships):
            f = random.randint(*fab_time)
            a = random.randint(*assy_time)
            e = random.randint(*erect_time)
            o = random.randint(*outfit_time)
            total += f / max(1, fab_machines) + a / max(1, assy_bays) + e / max(1, erect_docks) + o / max(1, outfit_berths)
        samples.append(total)
    return int(np.ceil(np.median(samples)))

# ------------------------------
# Helpers: Gantt + S-curve builders (used both live and final)
# ------------------------------
def plot_gantt_live(finished_records, in_progress, sim_time):
    # finished_records: list of dicts
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
    df_inprog = pd.DataFrame(inprog_rows) if len(inprog_rows)>0 else pd.DataFrame(columns=["Ship","Stage","Start","End","Duration (weeks)","InProgress","StageTotal"])

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
    stage_colors = {"Fabricate":"#87CEEB","Assembly":"#FFA500","Erection":"#90EE90","Outfitting":"#FA8072"}

    for i, ship in enumerate(ships):
        ship_data = df_all[df_all["Ship"] == ship].sort_values(by="Start")
        for _, row in ship_data.iterrows():
            left = row["Start"]
            width = row["Duration (weeks)"]
            color = stage_colors.get(row["Stage"], "gray")
            alpha = 0.95 if not row.get("InProgress", False) else 0.6
            ax.barh(ship, width, left=left, color=to_rgba(color, alpha=alpha), edgecolor="k", height=0.5)
            if row.get("InProgress", False):
                pct = (row["Duration (weeks)"] / row["StageTotal"]) * 100 if row["StageTotal"] > 0 else 0
                txt = f"{row['Stage']} ({pct:.0f}%)"
            else:
                txt = f"{row['Stage']} (100%)"
            ax.text(left + width/2, i, txt, ha='center', va='center', fontsize=8, color='black')

    ax.set_xlabel("Simulation Time (weeks)")
    ax.set_ylabel("Ships")
    ax.set_title("Live Gantt (finished + in-progress)")
    ax.set_xlim(0, sim_time)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def build_scurve_live(finished_records, in_progress, sim_time):
    timeline = list(range(0, sim_time+1))
    stage_list = []
    for r in finished_records:
        stage_list.append((r["Start"], r["End"], r["Duration (weeks)"]))
    for (ship, stage), (start_week, done_weeks, total_weeks) in in_progress.items():
        stage_list.append((start_week, start_week + done_weeks, total_weeks))
    total_stage_count = len(stage_list)
    if total_stage_count == 0:
        s_curve_df = pd.DataFrame({"Week": timeline, "Completion (%)": [0]*len(timeline)})
        return s_curve_df, None
    completion_over_time = []
    for week in timeline:
        completed_frac_sum = 0.0
        for (s,e,dur) in stage_list:
            if week < s:
                frac = 0.0
            elif week >= e:
                frac = 1.0
            else:
                frac = (week - s) / dur if dur>0 else 0.0
            frac = max(0.0, min(1.0, frac))
            completed_frac_sum += frac
        pct = (completed_frac_sum / total_stage_count) * 100.0
        completion_over_time.append(pct)
    s_curve_df = pd.DataFrame({"Week": timeline, "Completion (%)": completion_over_time})
    t = np.array(timeline); t0 = sim_time/2.0; k = 0.12
    ideal_completion = 100.0 / (1.0 + np.exp(-k * (t - t0)))
    ideal_df = pd.DataFrame({"Week": timeline, "Completion (%)": ideal_completion})
    return s_curve_df, ideal_df

# ------------------------------
# Core: Real-time Sim with callbacks
# ------------------------------
def run_simulation_real_time(
    num_ships, fab_machines, assy_bays, erect_docks, outfit_berths,
    fab_time, assy_time, erect_time, outfit_time, sim_time,
    progress_cb=None, stage_cb=None, snapshot_cb=None, per_ship_cb=None,
    random_seed=42, slow_down=0.03
):
    random.seed(random_seed)
    records = []
    completed_ships = {"count": 0}
    in_progress = {}          # (ship,stage) -> [start_week, done_weeks, total_weeks]
    ship_stage_totals = {}    # sampled totals for each ship (denominator for per-ship %)

    env = simpy.Environment()
    fab = simpy.Resource(env, capacity=fab_machines)
    assy = simpy.Resource(env, capacity=assy_bays)
    erect = simpy.Resource(env, capacity=erect_docks)
    outfit = simpy.Resource(env, capacity=outfit_berths)

    def record_event(ship, stage, start, end):
        records.append({"Ship": ship, "Stage": stage, "Start": start, "End": end, "Duration (weeks)": end - start})

    def ensure_ship_totals(ship):
        if ship not in ship_stage_totals:
            ship_stage_totals[ship] = max(1, random.randint(*fab_time) + random.randint(*assy_time) + random.randint(*erect_time) + random.randint(*outfit_time))

    def compute_and_emit_ship_percent(ship):
        total_done = 0.0
        denom = ship_stage_totals.get(ship, 1)
        for r in records:
            if r["Ship"] == ship:
                total_done += max(0, min(r["End"], sim_time) - r["Start"])
        for (s, stg), (start, done, tot) in in_progress.items():
            if s == ship:
                total_done += done
        pct = min(100.0, (total_done / denom) * 100.0)
        if per_ship_cb:
            per_ship_cb(ship, pct)
        return pct

    def process_stage(ship, stage_name, resource, duration_range):
        with resource.request() as req:
            yield req
            ensure_ship_totals(ship)
            start = env.now
            t = random.randint(*duration_range)
            in_progress[(ship, stage_name)] = [start, 0, t]
            for w in range(t):
                in_progress[(ship, stage_name)][1] = w + 1
                if stage_cb:
                    stage_cb(ship, stage_name, w+1, t, env.now)
                # update per-ship percent and callbacks
                compute_and_emit_ship_percent(ship)
                if snapshot_cb:
                    snapshot_cb(list(records), dict(in_progress), env.now)
                if progress_cb:
                    progress_cb((completed_ships["count"] / num_ships) * 100.0)
                yield env.timeout(1)
                if slow_down and slow_down > 0:
                    time.sleep(slow_down)
            end = env.now
            record_event(ship, stage_name, start, end)
            in_progress.pop((ship, stage_name), None)
            # emit updates after stage finishes
            compute_and_emit_ship_percent(ship)
            if snapshot_cb:
                snapshot_cb(list(records), dict(in_progress), env.now)

    def ship_process(ship):
        yield env.process(process_stage(ship, "Fabricate", fab, fab_time))
        yield env.process(process_stage(ship, "Assembly", assy, assy_time))
        yield env.process(process_stage(ship, "Erection", erect, erect_time))
        yield env.process(process_stage(ship, "Outfitting", outfit, outfit_time))
        completed_ships["count"] += 1
        if progress_cb:
            progress_cb((completed_ships["count"] / num_ships) * 100.0)
        # emit all ships' percents
        for s in list(ship_stage_totals.keys()):
            compute_and_emit_ship_percent(s)
        if snapshot_cb:
            snapshot_cb(list(records), dict(in_progress), env.now)

    def progress_monitor(env):
        while True:
            if progress_cb:
                progress_cb((completed_ships["count"] / num_ships) * 100.0)
            if snapshot_cb:
                snapshot_cb(list(records), dict(in_progress), env.now)
            yield env.timeout(1)
            if env.now >= sim_time:
                break

    def shipyard(env):
        env.process(progress_monitor(env))
        for i in range(num_ships):
            env.process(ship_process(f"Ship-{i+1}"))
            yield env.timeout(random.randint(1, 3))

    env.process(shipyard(env))
    env.run(until=sim_time)
    df_records = pd.DataFrame(records)
    # final emit per-ship percents
    for s in list(ship_stage_totals.keys()):
        compute_and_emit_ship_percent(s)
    return df_records, completed_ships["count"]

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="ASHKAM Real-Time Simulator", layout="wide")
st.title("🚢 ASHKAM Shipyard Simulator — Real-Time")
st.markdown("**Simulation unit:** weeks")

# Top overall progress bar
overall_progress = st.progress(0)
overall_status = st.empty()

# Mode and display controls (sidebar)
with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["Default", "User Defined"])
    display_mode = st.radio("Per-Ship Progress Display", ["Sidebar", "Main Body"])
    speed = st.selectbox("Simulation Speed", ["Slow", "Medium", "Fast"], index=1)
    if speed == "Slow":
        slow_down = 0.08
    elif speed == "Medium":
        slow_down = 0.03
    else:
        slow_down = 0.0

    st.markdown("---")
    st.subheader("Resources & Ships")
    if mode == "Default":
        num_ships = st.number_input("Number of Ships", min_value=1, value=6)
        fab_machines = st.number_input("Fabrication Machines", min_value=1, value=3)
        assy_bays = st.number_input("Assembly Bays", min_value=1, value=2)
        erect_docks = st.number_input("Erection Docks", min_value=1, value=1)
        outfit_berths = st.number_input("Outfitting Berths", min_value=1, value=1)
        sim_time = st.number_input("Simulation Time (weeks)", min_value=20, value=300)
        fab_time = (8,12); assy_time=(10,15); erect_time=(12,20); outfit_time=(15,25)
    else:
        num_ships = st.number_input("Number of Ships", min_value=1, value=6)
        fab_machines = st.number_input("Fabrication Machines", min_value=1, value=3)
        assy_bays = st.number_input("Assembly Bays", min_value=1, value=2)
        erect_docks = st.number_input("Erection Docks", min_value=1, value=1)
        outfit_berths = st.number_input("Outfitting Berths", min_value=1, value=1)
        sim_time = st.number_input("Simulation Time (weeks)", min_value=20, value=300)
        st.markdown("Stage Duration Ranges (weeks)")
        fab_min = st.number_input("Fabrication Min", 1, 100, 8)
        fab_max = st.number_input("Fabrication Max", 1, 100, 12)
        assy_min = st.number_input("Assembly Min", 1, 100, 10)
        assy_max = st.number_input("Assembly Max", 1, 100, 15)
        erect_min = st.number_input("Erection Min", 1, 100, 12)
        erect_max = st.number_input("Erection Max", 1, 100, 20)
        outfit_min = st.number_input("Outfitting Min", 1, 100, 15)
        outfit_max = st.number_input("Outfitting Max", 1, 100, 25)
        fab_time = (fab_min, fab_max); assy_time=(assy_min,assy_max); erect_time=(erect_min,erect_max); outfit_time=(outfit_min,outfit_max)

    st.markdown("---")
    est_min = estimate_min_time(num_ships, fab_machines, assy_bays, erect_docks, outfit_berths,
                                fab_time, assy_time, erect_time, outfit_time, trials=200)
    st.info(f"Estimated minimum time to complete all {num_ships} ships (approx): **{est_min} weeks** (median)")

# placeholders and UI areas
st.markdown("### Real-time Controls & Status")
status_box = st.empty()
stage_box = st.empty()

st.markdown("### Per-Ship Progress")
# prepare placeholders for per-ship bars in both positions
main_placeholders = {}
sidebar_placeholders = {}
for i in range(1, num_ships+1):
    main_placeholders[f"Ship-{i}"] = st.empty()
with st.sidebar:
    st.markdown("### Per-Ship Progress (Sidebar)")
    for i in range(1, num_ships+1):
        sidebar_placeholders[f"Ship-{i}"] = st.empty()

# lower area for visuals
gantt_placeholder = st.empty()
scurve_placeholder = st.empty()
report_placeholder = st.empty()

# global snapshot holders used by callbacks
finished_records_global = []
in_progress_global = {}
env_now_global = 0

# callback implementations
def progress_callback(percent):
    overall_progress.progress(min(100, int(percent)))
    overall_status.text(f"Overall completion: {percent:.1f}%")

def stage_callback(ship, stage, done_weeks, total_weeks, env_now):
    # label above bar: Ship — Stage — %
    pct = (done_weeks / total_weeks) * 100 if total_weeks>0 else 0.0
    label = f"**{ship} — {stage} — {pct:.1f}%**"
    status_box.markdown(label)
    # update whichever display mode chosen
    # compute per-ship percent from snapshot globals
    per_done = {}
    per_total = {}
    for r in finished_records_global:
        s = r["Ship"]
        per_done[s] = per_done.get(s, 0) + max(0, min(r["End"], sim_time) - r["Start"])
        per_total[s] = per_total.get(s, 0) + r["Duration (weeks)"]
    for (s, stg), (start, done, tot) in in_progress_global.items():
        per_done[s] = per_done.get(s, 0) + done
        per_total[s] = per_total.get(s, 0) + tot
    for i in range(1, num_ships+1):
        ship_name = f"Ship-{i}"
        done_val = per_done.get(ship_name, 0)
        total_val = per_total.get(ship_name, None)
        if total_val is None or total_val == 0:
            pct_ship = 0.0
        else:
            pct_ship = min(100.0, (done_val / total_val) * 100.0)
        # render to chosen place and also keep both updated (for user trying both)
        if display_mode == "Main Body":
            # main area placeholders show above-bar label then progress
            main_placeholders[ship_name].markdown(f"**{ship_name} — {pct_ship:.1f}%**")
            main_placeholders[ship_name].write(f"**{ship_name} — {pct_ship:.1f}%**")
            main_placeholders[ship_name].progress(int(pct_ship))
            # keep sidebar also updated but less prominent
            # sidebar_placeholders[ship_name].markdown(f"{ship_name}: {pct_ship:.1f}%")
            # sidebar_placeholders[ship_name].progress(int(pct_ship))
        else:
            sidebar_placeholders[ship_name].markdown(f"**{ship_name} — {pct_ship:.1f}%**")
            sidebar_placeholders[ship_name].write(f"**{ship_name} — {pct_ship:.1f}%**")
            sidebar_placeholders[ship_name].progress(int(pct_ship))
            # main_placeholders[ship_name].markdown(f"{ship_name}: {pct_ship:.1f}%")
            # main_placeholders[ship_name].progress(int(pct_ship))

def snapshot_callback(finished_records, in_progress, env_now):
    global finished_records_global, in_progress_global, env_now_global
    finished_records_global = finished_records
    in_progress_global = in_progress
    env_now_global = env_now
    # update live Gantt & S-curve
    fig = plot_gantt_live(finished_records_global, in_progress_global, sim_time)
    gantt_placeholder.pyplot(fig)
    s_df, ideal_df = build_scurve_live(finished_records_global, in_progress_global, sim_time)
    fig_s, ax = plt.subplots(figsize=(10,4))
    ax.plot(s_df["Week"], s_df["Completion (%)"], label="Actual (partial)", color="tab:blue")
    if ideal_df is not None:
        ax.plot(ideal_df["Week"], ideal_df["Completion (%)"], label="Ideal (sigmoid)", color="orange", linestyle="--")
    ax.set_xlabel("Week"); ax.set_ylabel("Completion (%)"); ax.set_ylim(0,100)
    ax.set_title("Live S-Curve (Actual vs Ideal)"); ax.legend(); ax.grid(True)
    scurve_placeholder.pyplot(fig_s)
    # small report snapshot
    rec_list = []
    for r in finished_records_global:
        rec_list.append({"Ship": r["Ship"], "Stage": r["Stage"], "Duration (weeks)": r["Duration (weeks)"], "Actual Duration": r["Duration (weeks)"], "Stage Completion (%)": 100.0})
    for (ship, stage), (start, done, tot) in in_progress_global.items():
        rec_list.append({"Ship": ship, "Stage": stage, "Duration (weeks)": tot, "Actual Duration": done, "Stage Completion (%)": (done/tot*100.0) if tot>0 else 0.0})
    df_report = pd.DataFrame(rec_list)
    if not df_report.empty:
        ship_comp = df_report.groupby("Ship")["Stage Completion (%)"].mean().reset_index().rename(columns={"Stage Completion (%)":"Ship Completion (%)"})
        df_report = df_report.merge(ship_comp, on="Ship", how="left")
        report_placeholder.dataframe(df_report.sort_values(["Ship","Stage"]))

# Run button
if st.button("▶️ Run Real-Time Simulation"):
    # clear visuals
    gantt_placeholder.empty(); scurve_placeholder.empty(); report_placeholder.empty()
    overall_progress.progress(0); overall_status.text("Simulation starting...")
    status_box.empty(); stage_box.empty()

    df_records, completed_count = run_simulation_real_time(
        num_ships, fab_machines, assy_bays, erect_docks, outfit_berths,
        fab_time, assy_time, erect_time, outfit_time, sim_time,
        progress_cb=progress_callback,
        stage_cb=stage_callback,
        snapshot_cb=snapshot_callback,
        per_ship_cb=None,
        random_seed=42,
        slow_down=slow_down
    )

    # finalize UI
    overall_progress.empty()
    overall_status.text(f"Simulation completed — {completed_count}/{num_ships} ships finished (within {sim_time} weeks)")

    if df_records.empty:
        st.warning("No stages finished during simulation time.")
    else:
        # compute partials and ship completion
        df_records["Actual Duration"] = df_records.apply(lambda row: max(0, min(row["End"], sim_time) - row["Start"]), axis=1)
        df_records["Stage Completion (%)"] = (df_records["Actual Duration"] / df_records["Duration (weeks)"]) * 100
        ship_completion = df_records.groupby("Ship")["Stage Completion (%)"].mean().reset_index().rename(columns={"Stage Completion (%)":"Ship Completion (%)"})
        summary_report = df_records.merge(ship_completion, on="Ship", how="left")
        summary_report = summary_report[["Ship","Stage","Duration (weeks)","Actual Duration","Stage Completion (%)","Ship Completion (%)"]]
        st.subheader("📄 Final Shipwise Stage Completion Report")
        st.dataframe(summary_report.style.format({"Duration (weeks)":"{:.1f}","Actual Duration":"{:.1f}","Stage Completion (%)":"{:.1f}%","Ship Completion (%)":"{:.1f}%"}))

        # final gantt
        final_fig = plot_gantt_live(df_records.to_dict("records"), {}, sim_time)
        st.subheader("🚀 Final Gantt (finished stages)")
        st.pyplot(final_fig)

        # final scurve
        s_df, ideal_df = build_scurve_live(df_records.to_dict("records"), {}, sim_time)
        fig_final, axf = plt.subplots(figsize=(10,4))
        axf.plot(s_df["Week"], s_df["Completion (%)"], label="Actual (partial)", color="tab:blue")
        if ideal_df is not None:
            axf.plot(ideal_df["Week"], ideal_df["Completion (%)"], label="Ideal (sigmoid)", color="orange", linestyle="--")
        axf.set_xlabel("Week"); axf.set_ylabel("Completion (%)"); axf.set_ylim(0,100)
        axf.set_title("Final S-Curve"); axf.legend(); axf.grid(True)
        st.pyplot(fig_final)

    # ensure per-ship bars show final percents
    per_done = {}
    per_total = {}
    for r in df_records.to_dict("records"):
        per_done[r["Ship"]] = per_done.get(r["Ship"], 0) + max(0, min(r["End"], sim_time) - r["Start"])
        per_total[r["Ship"]] = per_total.get(r["Ship"], 0) + r["Duration (weeks)"]
    for i in range(1, num_ships+1):
        ship_name = f"Ship-{i}"
        done = per_done.get(ship_name, 0)
        total = per_total.get(ship_name, None)
        pct = min(100.0, (done/total)*100.0) if total and total>0 else 0.0
        # display final in both places for clarity
        main_placeholders[ship_name].markdown(f"**{ship_name} — {pct:.1f}%**")
        # main_placeholders[ship_name].write(f"{ship_name} — {pct:.1f}%")
        # main_placeholders[ship_name].progress(int(pct))
        # sidebar_placeholders[ship_name].markdown(f"**{ship_name} — {pct:.1f}%**")
        sidebar_placeholders[ship_name].write(f"**{ship_name} — {pct:.1f}%**")
        # sidebar_placeholders[ship_name].progress(int(pct))
        # main_placeholders[ship_name].write(f"{ship_name} — {pct:.1f}%")
        # main_placeholders[ship_name].progress(int(pct))

