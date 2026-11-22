import streamlit as st
import pandas as pd

GROUP_LETTER = {"Doing well": "A", "Needs attention": "B", "Needs support": "C"}

SAMPLE = pd.DataFrame({
    "student_id": ["S01", "S02", "S03", "S04", "S05", "S06"],
    "sem1": ["FCD", "FC", "Fail", "FC", "FCD", "SC"],
    "sem2": ["FCD", "FC", "Fail", "FCD", "FC", "FC"],
    "sem3": ["FCD", "FCD", "FC", "FCD", "FCD", "SC"],
})

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
    st.write("This app estimates which students may be at risk of failing a semester, using their earlier semester results.")
    st.info("Sample data for layout. Real data will be connected later.")

    sem_cols = ["sem1", "sem2", "sem3"]
    has_fail = (SAMPLE[sem_cols] == "Fail").any(axis=1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Students", len(SAMPLE))
    c2.metric("With at least one Fail", int(has_fail.sum()))
    c3.metric("Share with a Fail", f"{has_fail.mean():.0%}")

    st.subheader("Sample records")
    st.dataframe(SAMPLE)

    st.caption("All student data is anonymized. No names or USNs are stored or shown.")
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