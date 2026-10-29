import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Cloud ECG Analyzer", page_icon="❤️", layout="wide")
st.title("❤️ Cloud ECG Analyzer")
st.caption("Educational project: ECG waveform analysis + optional heartbeat classification")
st.warning("Research/education only. Not for diagnosis or medical decisions.")

def make_demo(fs=250, seconds=10):
    t=np.arange(0,seconds,1/fs)
    s=0.025*np.sin(2*np.pi*.33*t)+0.015*np.sin(2*np.pi*.1*t)
    for bt in np.arange(.7,seconds,60/72):
        s += .9*np.exp(-.5*((t-bt)/.018)**2)
        s -= .12*np.exp(-.5*((t-(bt-.035))/.012)**2)
        s -= .16*np.exp(-.5*((t-(bt+.04))/.016)**2)
        s += .12*np.exp(-.5*((t-(bt+.22))/.055)**2)
    s += np.random.default_rng(7).normal(0,.008,len(t))
    return t,s

def parse_csv(file):
    # First attempt normal headered CSV. The common ECG5000-style dataset is headerless,
    # with 140 numeric sample features followed by one target label.
    raw_bytes=file.getvalue()
    from io import BytesIO
    try:
        df=pd.read_csv(BytesIO(raw_bytes), header=None)
        numeric=df.apply(pd.to_numeric, errors="coerce")
        numeric=numeric.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if numeric.shape[1] >= 2 and numeric.shape[0] > 20:
            return numeric, "headerless"
    except Exception:
        pass
    df=pd.read_csv(BytesIO(raw_bytes))
    return df, "headered"

with st.sidebar:
    st.header("Settings")
    fs_setting=st.number_input("Sampling rate (Hz)",min_value=25,max_value=5000,value=250,step=25)
    do_filter=st.checkbox("Apply band-pass filter (0.5–40 Hz)",value=True)
    st.caption("If using a labelled heartbeat dataset, upload the original CSV to the classifier section.")

uploaded=st.file_uploader("Upload ECG CSV (optional)", type=["csv"])
tab1,tab2=st.tabs(["Waveform analysis","Heartbeat classifier"])

with tab1:
    try:
        if uploaded:
            df,mode=parse_csv(uploaded)
            # For a wide labelled heartbeat dataset, use its first row's waveform only for visualization.
            if df.shape[1] >= 20 and df.shape[0] > 20:
                signal=df.iloc[0,:-1].to_numpy(dtype=float) if df.shape[1] >= 100 else df.select_dtypes(include=[np.number]).iloc[:,0].to_numpy(dtype=float)
                raw=np.asarray(signal,dtype=float)
                fs=float(fs_setting); t=np.arange(len(raw))/fs
                source=f"Uploaded dataset preview — first row, {len(raw)} samples"
            else:
                numeric=df.select_dtypes(include=[np.number])
                if numeric.empty:
                    st.error("CSV must contain numeric values.")
                    st.stop()
                cols={str(c).lower():c for c in numeric.columns}
                sigcol=next((cols[k] for k in ["ecg","signal","voltage","amplitude","value"] if k in cols),None)
                if sigcol is None:
                    sigcol=numeric.columns[0]
                raw=numeric[sigcol].dropna().to_numpy(float)
                fs=float(fs_setting); t=np.arange(len(raw))/fs
                source=f"Uploaded file — signal column: {sigcol}"
        else:
            fs=float(fs_setting); t,raw=make_demo(fs); source="Built-in synthetic demo signal"
        if len(raw)<10:
            st.error("Not enough numeric samples to plot.")
            st.stop()
        raw=np.nan_to_num(raw)
        clean=raw.copy()
        if do_filter and len(raw)>int(fs*2):
            nyq=fs/2; lo=.5/nyq; hi=min(40/nyq,.95)
            if 0<lo<hi<1:
                b,a=butter(2,[lo,hi],btype="band")
                clean=filtfilt(b,a,raw)
        peaks,_=find_peaks(clean,distance=max(1,int(fs*.3)),prominence=max(float(np.std(clean)*.7),1e-6))
        duration=(len(clean)-1)/fs
        rate=(len(peaks)/duration*60) if duration>0 else 0
        st.info(source)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Samples",f"{len(raw):,}")
        c2.metric("Duration",f"{duration:.2f} s")
        c3.metric("Candidate peaks",str(len(peaks)))
        c4.metric("Peak rate estimate",f"{rate:.1f}/min")
        fig,ax=plt.subplots(figsize=(12,4))
        ax.plot(t,raw,alpha=.55,lw=.8,label="Original")
        ax.plot(t,clean,lw=1,label="Processed")
        if len(peaks): ax.scatter(t[peaks],clean[peaks],s=22,label="Candidate peaks")
        ax.set_xlabel("Time (seconds)"); ax.set_ylabel("Amplitude (dataset units)")
        ax.grid(True,alpha=.25); ax.legend()
        st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.write({"Mean":round(float(np.mean(raw)),5),"Standard deviation":round(float(np.std(raw)),5),
                  "Minimum":round(float(np.min(raw)),5),"Maximum":round(float(np.max(raw)),5)})
    except Exception as e:
        st.error(f"Could not analyze this CSV: {e}")

with tab2:
    st.subheader("Train a baseline heartbeat classifier")
    st.write("For a labelled dataset with many numeric columns, this section treats the last column as the class label and earlier columns as sample features.")
    if uploaded:
        try:
            df,mode=parse_csv(uploaded)
            df=df.apply(pd.to_numeric,errors="coerce").dropna(axis=1,how="all").dropna(axis=0,how="all")
            if df.shape[0] < 30 or df.shape[1] < 3:
                st.info("Upload a labelled dataset with multiple feature columns and a final label column to train a classifier.")
            else:
                X=df.iloc[:,:-1]
                y=df.iloc[:,-1]
                valid=y.notna()
                X=X.loc[valid].replace([np.inf,-np.inf],np.nan)
                y=y.loc[valid].astype(str)
                if y.nunique()<2:
                    st.warning("The final column has fewer than two classes. Please check that the last column is the label.")
                else:
                    st.write(f"Dataset detected: **{len(X):,} rows**, **{X.shape[1]} features**, **{y.nunique()} classes**.")
                    if st.button("Train and evaluate Random Forest", type="primary"):
                        with st.spinner("Training baseline model..."):
                            X_train,X_test,y_train,y_test=train_test_split(
                                X,y,test_size=0.2,random_state=42,stratify=y if y.value_counts().min()>=2 else None)
                            model=make_pipeline(SimpleImputer(strategy="median"),
                                RandomForestClassifier(n_estimators=120,random_state=42,class_weight="balanced"))
                            model.fit(X_train,y_train)
                            pred=model.predict(X_test)
                            acc=accuracy_score(y_test,pred)
                        st.metric("Hold-out accuracy",f"{acc*100:.2f}%")
                        st.caption("This is a simple random split baseline, not clinical validation. Accuracy depends on dataset quality and split design.")
                        labels=sorted(y.unique())
                        cm=confusion_matrix(y_test,pred,labels=labels)
                        fig,ax=plt.subplots(figsize=(7,5))
                        im=ax.imshow(cm,aspect="auto")
                        ax.set_xticks(range(len(labels)),labels=labels,rotation=45)
                        ax.set_yticks(range(len(labels)),labels=labels)
                        ax.set_xlabel("Predicted label");ax.set_ylabel("True label");ax.set_title("Confusion matrix")
                        fig.colorbar(im,ax=ax);st.pyplot(fig);plt.close(fig)
                        report=classification_report(y_test,pred,zero_division=0,output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose(),use_container_width=True)
                        st.success("Baseline model training completed.")
        except Exception as e:
            st.error(f"Could not train classifier: {e}")
    else:
        st.info("Upload your original labelled ECG CSV above. The synthetic demo signal is not a labelled training dataset.")

with st.expander("Algorithms and limitations"):
    st.markdown("""
    - **Waveform view:** optional Butterworth band-pass filter and candidate peak detection with `scipy.signal.find_peaks`.
    - **Classifier:** Random Forest using numeric columns as features and the final column as the label.
    - **Evaluation:** hold-out accuracy, classification report and confusion matrix.
    - This app assumes a labelled dataset where the final column is the class label. Verify your dataset documentation before interpreting results.
    - This is a student prototype, not a validated clinical device. Do not use its output for diagnosis.
    """)
