import pandas as pd

SRC = "student_semester_summary.csv"
SCORE = {"FCD": 3, "FC": 2, "SC": 1, "PASS": 1, "FAIL": 0}   # result class -> number (Pass treated like SC)
NOT_READY = [(2025, 3), (2025, 4)]   # (batch, semester): unfinished or unverified results, left out

d = pd.read_csv(SRC, encoding="utf-8-sig")
print("start:", len(d), "rows")

# 1. work out each student's REAL batch from the USN (file name is not reliable)
#    4MH22CD001 -> 2022. Lateral entry students have numbers 400+ and belong to the year before.
d["sem_no"] = d["semester"].str.extract(r"(\d)")[0].astype(int)
d["own_batch"] = 2000 + d["USN"].str[3:5].astype(int) - (d["USN"].str[7:].astype(int) >= 400).astype(int)
d["file_batch"] = d["batch"].str[2:].astype(int)
d = d.drop(columns=["batch"])          # the file-name label is not needed any more

# 2. keep a student's rows only from their own batch's file (removes copied 6th/7th sem sheets)
d = d[d["own_batch"] == d["file_batch"]]
print("after keeping own-batch rows:", len(d))

# 3. drop semesters that are not finished or not verified
for b, s in NOT_READY:
    d = d[~((d["own_batch"] == b) & (d["sem_no"] == s))]
print("after dropping not-ready semesters:", len(d))

# 4. drop rows with no result or no marks
d["result"] = d["result"].astype("string").str.strip().str.upper()
d = d.dropna(subset=["result", "percentage", "failed_subjects"])
d = d[d["percentage"] > 0]
print("after dropping missing/zero-mark rows:", len(d))

# 5. numbers for the model
d["result_score"] = d["result"].map(SCORE)
assert d["result_score"].notna().all(), "unknown result label"
d["failed_subjects"] = d["failed_subjects"].astype(int)
d["at_risk"] = (d["result"] == "FAIL").astype(int)

# 6. anonymise: student_id like 2022_001 (running number inside the batch)
d = d.sort_values(["own_batch", "USN", "sem_no"]).reset_index(drop=True)
running = d.groupby("own_batch")["USN"].rank(method="dense").astype(int)
d["student_id"] = d["own_batch"].astype(str) + "_" + running.astype(str).str.zfill(3)

sem_table = d.rename(columns={"own_batch": "batch", "sem_no": "semester_no"})[
    ["student_id", "batch", "semester_no", "total_marks", "percentage",
     "result", "result_score", "failed_subjects", "at_risk"]]
sem_table.to_csv("semester_results_clean.csv", index=False)

# 7. model table: one row per student per semester to predict, using ONLY earlier semesters
rows = []
for sid, g in sem_table.groupby("student_id"):
    g = g.sort_values("semester_no").reset_index(drop=True)
    for i in range(1, len(g)):
        cur, prev = g.iloc[i], g.iloc[:i]
        if cur["semester_no"] != prev.iloc[-1]["semester_no"] + 1:
            continue                                  # need the semester right before
        rows.append({
            "student_id": sid,
            "batch": cur["batch"],
            "target_sem": cur["semester_no"],
            "n_prev_sems": len(prev),
            "prev_pct": prev.iloc[-1]["percentage"],
            "prev_score": prev.iloc[-1]["result_score"],
            "prev_failed_subjects": prev.iloc[-1]["failed_subjects"],
            "avg_prev_pct": prev["percentage"].mean(),
            "min_prev_pct": prev["percentage"].min(),
            "prev_fail_count": int(prev["at_risk"].sum()),
            "at_risk": cur["at_risk"],
        })
model = pd.DataFrame(rows)
model.to_csv("model_data.csv", index=False)

print("\nsemester table:", sem_table.shape, "| students:", sem_table["student_id"].nunique())
print("model table:", model.shape, "| fails:", int(model["at_risk"].sum()))
print(model.groupby("batch")["at_risk"].agg(["count", "sum"]))
