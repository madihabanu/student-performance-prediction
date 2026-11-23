import streamlit as st
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "dataset"

@st.cache_data
def load_semester_data():
    return pd.read_csv(DATA_DIR / "semester_results_clean.csv")

semester_df = load_semester_data()

SCORE = {"FCD": 3, "FC": 2, "SC": 1, "Fail": 0}


def get_group(scores):
    if min(scores) == 0:
        return "Needs support"
    elif min(scores) >= 2:
        return "Doing well"
    else:
        return "Needs attention"


st.title("Student Performance Prediction")

page = st.sidebar.radio("Go to", ["Overview", "EDA", "Model Results", "Predict"])

if page == "Overview":
    st.header("Overview")

    st.write(
        "An overview of student academic performance "
        "based on previous semester results."
    )

    # Use all semester records
    academic_df = semester_df.copy()

    # Count unique students
    total_students = academic_df["student_id"].nunique()

    # Count semester records
    total_records = len(academic_df)

    # Identify students with at least one failed semester
    student_risk = (
        academic_df.groupby("student_id")["at_risk"]
        .max()
    )

    students_at_risk = int(student_risk.sum())

    risk_percentage = (
        students_at_risk / total_students * 100
    )

    # Display metrics
    c1, c2, c3 = st.columns(3)

    c1.metric("Total Students", total_students)
    c2.metric("Students At-Risk", students_at_risk)
    c3.metric("At-Risk Percentage", f"{risk_percentage:.1f}%")

    result_counts = (
    academic_df["result"]
    .value_counts()
    .rename_axis("Result")
    .reset_index(name="Count")
)

    st.bar_chart(
        result_counts,
        x="Result",
        y="Count",
        x_label="Result Category",
        y_label="Number of Records"
 )

    st.subheader("Student Academic Records")

    with st.expander("View All Records"):

        filtered_df = academic_df.copy()

        # Student ID search
        search_id = st.text_input(
            "Search by Student ID",
            placeholder="Enter Student ID"
        )

        if search_id:
            filtered_df = filtered_df[
                filtered_df["student_id"]
                .astype(str)
                .str.contains(search_id, case=False, na=False)
            ]

        # Semester filter
        semesters = sorted(academic_df["semester_no"].dropna().unique())

        selected_semester = st.selectbox(
            "Filter by Semester",
            ["All Semesters"] + semesters
        )

        if selected_semester != "All Semesters":
            filtered_df = filtered_df[
                filtered_df["semester_no"] == selected_semester
            ]

        # Result filter
        results = sorted(academic_df["result"].dropna().unique())

        selected_result = st.selectbox(
            "Filter by Result",
            ["All Results"] + results
        )

        if selected_result != "All Results":
            filtered_df = filtered_df[
                filtered_df["result"] == selected_result
            ]

        # Display filtered records
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"Showing {len(filtered_df)} of {len(academic_df)} records"
        )

        # Download filtered records
        csv = filtered_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Filtered Records",
            data=csv,
            file_name="filtered_student_records.csv",
            mime="text/csv"
        )
        
elif page == "EDA":
    st.header("EDA")
    st.write("Charts will go here.")
elif page == "Model Results":
    st.header("Model Results")
    st.write("Model scores will go here.")
else:
    st.header("Predict")
    st.caption("This is an estimate, not a guarantee.")

    target = st.selectbox("Which semester do you want to predict?", [2, 3, 4, 5, 6, 7, 8])
    st.write(f"Enter the results for semesters 1 to {target - 1}:")

    with st.form("predict_form"):
        history = []
        for i in range(1, target):
            choice = st.selectbox(f"Semester {i} result", ["FCD", "FC", "SC", "Fail"], key=f"sem_{i}")
            history.append(choice)
        submitted = st.form_submit_button("Predict")

    if submitted:
        scores = [SCORE[h] for h in history]
        average = sum(scores) / len(scores)
        risk = (3 - average) / 3
        group = get_group(scores)

        st.subheader(f"This student belongs to Group {GROUP_LETTER[group]}: {group}")
        st.metric(f"Estimated risk of failing semester {target}", f"{risk:.0%}")
        st.progress(risk)

        if group == "Needs support":
            st.error("This student may need early help, such as mentoring or extra classes.")
        elif group == "Needs attention":
            st.warning("This student is worth keeping an eye on. Extra practice may help.")
        else:
            st.success("This student is doing well. No special action needed.")

        st.caption("Groups and risk are estimates to help teachers give support. They are not a judgment of the student.")