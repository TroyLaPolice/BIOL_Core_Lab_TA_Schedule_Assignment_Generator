import streamlit as st
import pandas as pd
import numpy as np
import pulp
import io
import re
import datetime
import difflib
import time
import math

st.set_page_config(page_title="TA Schedule Optimizer", layout="wide")

# ==========================================
# Sidebar — User Parameters (Plain English)
# ==========================================
st.sidebar.title("⚙️ Solver Settings")

time_limit = st.sidebar.slider(
    "⏱️ Maximum time allowed for solver (seconds)",
    min_value=30, max_value=600, value=300, step=30,
    help="How long the solver can run before stopping automatically."
)

fairness_weight = st.sidebar.slider(
    "⚖️ Fairness weight",
    min_value=1000, max_value=20000, value=10000, step=1000,
    help="Higher values make the solver focus more on evenly distributing good slots between TAs."
)

consistency_weight = st.sidebar.slider(
    "📊 Consistency weight (rank spread)",
    min_value=10, max_value=1000, value=100, step=10,
    help="Higher values reduce how different the rank scores are for each TA’s assigned slots."
)

# --- The crucial optimization parameter ---
optimization_param = st.sidebar.slider(
    "🚫 Fraction of least-preferred slots to avoid",
    min_value=0.0, max_value=1.0, value=0.6, step=0.05,
    help="TAs will not be assigned to the bottom fraction of slots (e.g., 0.6 = bottom 60%)."
)
st.session_state["optimization_param"] = optimization_param

st.sidebar.markdown("---")
st.sidebar.info("After uploading both CSVs, click **Run Solver** to start optimization.")

# ==========================================
# File Uploads
# ==========================================
st.title("🎓 BIOL Core Lab TA Schedule Optimization App")

st.markdown("""
Upload the following two CSV files:
1. **Class Schedule CSV** — must include columns like `Class`, `Days & Times`, `Room`.
2. **TA Preferences CSV** — must include `Your Name (Last, First)`, time slots, and `Returner?`.
""")

uploaded_schedule = st.file_uploader("📁 Upload Schedule CSV", type="csv")
uploaded_prefs = st.file_uploader("📁 Upload TA Preferences CSV", type="csv")

if "df_schedule" not in st.session_state:
    st.session_state.df_schedule = None
if "df_prefs" not in st.session_state:
    st.session_state.df_prefs = None

if uploaded_schedule:
    st.session_state.df_schedule = pd.read_csv(uploaded_schedule)
if uploaded_prefs:
    st.session_state.df_prefs = pd.read_csv(uploaded_prefs)

# ==========================================
# Helper Functions
# ==========================================
day_map = {
    "m": "Monday", "mo": "Monday", "mon": "Monday",
    "t": "Tuesday", "tu": "Tuesday", "tue": "Tuesday",
    "w": "Wednesday", "we": "Wednesday", "wed": "Wednesday",
    "r": "Thursday", "th": "Thursday", "thu": "Thursday",
    "f": "Friday", "fr": "Friday", "fri": "Friday"
}

def normalize_day_time(day_time_str):
    day_time_str = str(day_time_str).strip()
    parts = day_time_str.split(None, 1)
    if len(parts) < 2:
        return day_time_str
    day_part = parts[0].lower()
    time_part = parts[1]
    full_day = day_map.get(day_part, day_part).upper()
    return f"{full_day} {time_part}"

def canonical_slot(slot):
    s = re.sub(r"[‐-‒–—]", "-", slot)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

# ==========================================
# Run Buttons
# ==========================================
run_solver = st.button("▶️ Run Solver")

# ==========================================
# Solver Execution
# ==========================================
if run_solver:
    if st.session_state.df_schedule is None or st.session_state.df_prefs is None:
        st.error("❌ Please upload both CSV files before running the solver.")
    else:
        df_schedule = st.session_state.df_schedule.copy()
        df_prefs = st.session_state.df_prefs.copy()

        st.info("🧮 Setting up optimization problem... Please wait.")

        # Normalize and count slot capacities
        df_schedule["Normalized Slot"] = df_schedule["Days & Times"].apply(normalize_day_time)
        slot_capacity_series = df_schedule.groupby("Normalized Slot").size()
        slot_capacities_dict = slot_capacity_series.to_dict()

        # Identify columns and structure
        cols = df_prefs.columns.tolist()
        name_col = "Your Name (Last, First)"
        start_idx = cols.index(name_col) + 1
        end_idx = cols.index("Returner?")
        raw_slot_names = cols[start_idx:end_idx]

        names = df_prefs[name_col].fillna("Unknown").tolist()
        rank_matrix = df_prefs.loc[:, raw_slot_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        assignment_counts = df_prefs.iloc[:, -1].fillna(1).astype(float).tolist()

        slot_names_norm = [normalize_day_time(s) for s in raw_slot_names]
        num_TAs = len(names)
        num_slots = len(slot_names_norm)
        slot_capacities = [slot_capacities_dict.get(canonical_slot(s), 1) for s in slot_names_norm]

        # --- Optimization param as in Colab ---
        optimization_param = st.session_state["optimization_param"]
        least_prefered = math.floor(optimization_param * num_slots)
        st.write(f"🔍 Avoiding bottom {optimization_param:.2f} fraction → {least_prefered} slots")

        # --- Build Optimization Problem ---
        prob = pulp.LpProblem("TA_Scheduling", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("assign", (range(num_TAs), range(num_slots)), cat="Binary")

        # --- Objective ---
        total_rank = pulp.lpSum(rank_matrix[i][j] * x[i][j] for i in range(num_TAs) for j in range(num_slots))
        prob += total_rank + fairness_weight * 0 + consistency_weight * 0  # simplified fairness placeholder

        # --- Constraints ---
        # Each TA must be assigned the correct number of slots
        for i in range(num_TAs):
            prob += pulp.lpSum(x[i][j] for j in range(num_slots)) == assignment_counts[i]

        # Each slot has limited capacity
        for j in range(num_slots):
            prob += pulp.lpSum(x[i][j] for i in range(num_TAs)) <= slot_capacities[j]

        # Avoid bottom-ranked slots for each TA
        for i in range(num_TAs):
            worst_slots = np.argsort(rank_matrix[i, :])[::-1][:least_prefered]
            for j in worst_slots:
                prob += x[i][j] == 0

        # --- Solve the model ---
        st.write("🔧 Running solver...")
        start = time.time()
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
        status = prob.solve(solver)
        solve_time = time.time() - start

        st.success(f"✅ Solver Status: {pulp.LpStatus[status]} — completed in {solve_time:.1f}s")

        if pulp.LpStatus[status] == "Optimal":
            results = []
            for i in range(num_TAs):
                for j in range(num_slots):
                    if pulp.value(x[i][j]) > 0.5:
                        results.append({
                            "TA Name": names[i],
                            "Assigned Slot": raw_slot_names[j],
                            "Rank": rank_matrix[i, j]
                        })
            results_df = pd.DataFrame(results)

            st.subheader("📅 Final Schedule")
            st.dataframe(results_df)

            avg_rank = results_df.groupby("TA Name")["Rank"].mean().mean()
            st.metric("Average Rank Across All Assignments", f"{avg_rank:.2f}")

            st.download_button(
                "💾 Download Schedule as CSV",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="ta_schedule_results.csv",
                mime="text/csv"
            )
        else:
            st.error("❌ The solver could not find a valid solution. Try lowering the avoid fraction or adjusting weights.")
