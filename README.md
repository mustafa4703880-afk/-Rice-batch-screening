# Rice Batch Screening

Offline CSV-based rice screening for pilot QA. The app accepts one row of
measurements per rice grain. Supported profiles:

| Profile | Original records | Source |
| --- | ---: | --- |
| Basmati, Jasmine and Karacadag | 45,000 | Three varieties, 15,000 each, from the original [Rice MSC Dataset](https://www.muratkoklu.com/datasets/) |
| Cammeo and Osmancik | 3,810 | [UCI Rice Dataset](https://doi.org/10.24432/C5MW4Z) |

The Rice MSC data file retains the seven shared measurement values and original
labels. Columns have been renamed for the app. See data/Rice_MSC_Citations.txt
for the source publications. Each profile has its own Random Forest trained
on that source, so choose the profile matching how the measurements were taken.
The two sources use separate measurement setups.

## Run on Windows

Use Python 3.15. Open the extracted folder containing app.py in VS Code or CMD.
Run each command separately.

```bat
py -3.15 -m venv .venv315
.\.venv315\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
.\.venv315\Scripts\python.exe -m streamlit run app.py
```

Open the Local URL shown in the terminal. Trained models are included. To
recreate them, run `.\.venv315\Scripts\python.exe train_model.py` before starting
Streamlit.

## Try a batch

1. In **Screen Batch**, pick the measurement profile and declared variety.
2. Select **Use built-in sample** and **Screen batch**. Deselect the sample
   option when uploading your own CSV.
3. Review the match, different variety, uncertain and input issue counts.
   **Flagged rows** lists individual grain results and reasons.
4. Record a decision and note in **QA review** and save. **Batch History**
   supports lot, supplier and date search and saved CSV download.
   The local database is created as data/batch_history.sqlite3.

A user CSV needs seven fields in the supplied template. The program also
recognizes source headings such as MAJOR_AXIS and MINOR_AXIS. An optional
Class column is ignored during screening. Sample CSVs contain held-out grains
from each research dataset. The minimum model score is a pilot review
threshold. The distribution is based on the uploaded sample. Confirm
measurement compatibility and validate on independent supplier lots before
using the results in a live QA decision.
